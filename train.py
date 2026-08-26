import asyncio
import itertools
import logging
import os

from sglang.srt.constants import GPU_MEMORY_TYPE_CUDA_GRAPH, GPU_MEMORY_TYPE_KV_CACHE, GPU_MEMORY_TYPE_WEIGHTS

from miles.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
from miles.utils.arguments import parse_args
from miles.utils.async_utils import eager_create_task
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.debug_utils.periodic_py_spy import maybe_start_periodic_pyspy_dump
from miles.utils.ft_utils.control_server.server import start_control_server
from miles.utils.ft_utils.mini_ft_controller import maybe_start_mini_ft_controller
from miles.utils.logging_utils import configure_logger
from miles.utils.misc import load_function, should_run_periodic_action
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking

logger = logging.getLogger(__name__)


async def _notify_external_policy_publication(
    external_policy_sync,
    *,
    rollout_id,
    actor_model,
    publication_info=None,
):
    if external_policy_sync is None:
        return
    callback = getattr(external_policy_sync, "after_inference_publication", None)
    if callback is not None:
        kwargs = dict(rollout_id=rollout_id, actor_model=actor_model)
        if getattr(
            external_policy_sync,
            "requires_exact_publication_info",
            False,
        ):
            if publication_info is None:
                raise RuntimeError(
                    "external policy identity requires the exact weight-publication engines"
                )
            kwargs["publication_info"] = publication_info
        await callback(**kwargs)


def _load_external_policy_sync(args, *, critic_model):
    """Construct a policy-sync backend and enforce its declared role support."""
    if args.external_policy_sync_path is None:
        return None
    synchronizer = load_function(args.external_policy_sync_path)(args)
    if critic_model is not None and not getattr(synchronizer, "supports_critic", False):
        raise ValueError(
            "the configured external policy synchronizer does not declare critic support"
        )
    return synchronizer


async def _train_actor_and_critic(
    actor_model,
    critic_model,
    rollout_id,
    rollout_data_ref,
):
    """Train both roles concurrently and stop either one when its peer fails."""
    critic_task = await eager_create_task(
        critic_model.train(rollout_id, rollout_data_ref)
    )
    actor_task = await eager_create_task(
        actor_model.train(rollout_id, rollout_data_ref)
    )
    tasks = (critic_task, actor_task)
    try:
        await asyncio.gather(*tasks)
    except BaseException:
        for task in tasks:
            if not task.done():
                task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        raise


async def train(args):
    configure_logger(args, source=MainProcessIdentity())
    maybe_start_periodic_pyspy_dump()
    # allocate the GPUs
    pgs = create_placement_groups(args)
    init_tracking(args)

    # create the rollout manager, with sglang engines inside.
    # need to initialize rollout manager first to calculate num_rollout
    rollout_manager, num_rollout_per_epoch = create_rollout_manager(args, pgs["rollout"])

    # create the actor and critic models
    actor_model, critic_model = await create_training_models(args, pgs, rollout_manager)

    external_policy_sync = _load_external_policy_sync(args, critic_model=critic_model)
    if external_policy_sync is not None:
        if args.offload_train:
            await actor_model.onload()
            if critic_model is not None:
                await critic_model.onload()
        initialize_kwargs = dict(
            actor_model=actor_model,
            rollout_manager=rollout_manager,
        )
        if critic_model is not None:
            initialize_kwargs["critic_model"] = critic_model
        await external_policy_sync.initialize(**initialize_kwargs)

    if args.control_server_port:
        start_control_server(
            actor_model=actor_model,
            rollout_manager=rollout_manager,
            port=args.control_server_port,
            ft_components=args.ft_components,
        )

    maybe_start_mini_ft_controller(args)

    initial_train_offloaded = False
    if (
        external_policy_sync is not None
        and args.offload_train
        and args.offload_rollout
    ):
        # Match the steady-state publication order below.  The external
        # policy has already been applied while the trainer is awake, so
        # retain its prepared LoRA payload before releasing trainer memory.
        # Resuming rollout weights while the trainer is still resident can
        # otherwise exceed GPU capacity before the first rollout.
        await actor_model.prepare_weight_update()
        await actor_model.offload()
        if critic_model is not None:
            await critic_model.offload()
        initial_train_offloaded = True

    if args.offload_rollout:
        await rollout_manager.onload_weights.remote()

    # always update weight first so that sglang has the loaded weights from training.
    publication_info = await actor_model.update_weights()
    await _notify_external_policy_publication(
        external_policy_sync,
        rollout_id=None,
        actor_model=actor_model,
        publication_info=publication_info,
    )
    if (
        external_policy_sync is not None
        and args.offload_train
        and not initial_train_offloaded
    ):
        await actor_model.offload()
        if critic_model is not None:
            await critic_model.offload()

    if args.check_weight_update_equal:
        await rollout_manager.check_weights.remote(
            action="compare",
            allow_quant_error=args.check_weight_update_allow_quant_error,
            selector=args.check_weight_update_selector,
            skip_list=args.check_weight_update_skip_list,
        )

    if args.offload_rollout:
        await rollout_manager.onload_kv.remote()

    # special case for eval-only
    if args.num_rollout == 0 and args.eval_interval is not None:
        await rollout_manager.eval.remote(rollout_id=0)

    async def offload_train():
        if args.offload_train:
            if args.use_critic:
                await critic_model.offload()
                if rollout_id >= args.num_critic_only_steps:
                    await actor_model.offload()
            else:
                await actor_model.offload()
        else:
            await actor_model.clear_memory()

    async def save(rollout_id, force_sync=False):
        force_sync = force_sync or rollout_id == args.num_rollout - 1
        if (not args.use_critic) or (rollout_id >= args.num_critic_only_steps):
            await actor_model.save_model(rollout_id, force_sync=force_sync)
        if args.use_critic:
            await critic_model.save_model(rollout_id, force_sync=force_sync)
        await rollout_manager.save.remote(rollout_id)

    # train loop.
    # note that for async training, one can change the position of the sync operation(ray.get).
    rollout_ids = (
        itertools.count(args.start_rollout_id)
        if external_policy_sync is not None
        and getattr(args, "external_policy_sync_run_until_stop", False)
        else range(args.start_rollout_id, args.num_rollout)
    )
    for rollout_id in rollout_ids:
        if args.eval_interval is not None and rollout_id == args.start_rollout_id and not args.skip_eval_before_train:
            await rollout_manager.eval.remote(rollout_id)

        rollout_data_ref = await rollout_manager.generate.remote(rollout_id)

        if args.offload_rollout:
            offload_tags = [GPU_MEMORY_TYPE_CUDA_GRAPH]
            if "kv_cache" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_KV_CACHE)
            if "weight" in args.offload_rollout_level:
                offload_tags.append(GPU_MEMORY_TYPE_WEIGHTS)
            await rollout_manager.offload.remote(tags=offload_tags)

        if args.use_critic:
            if rollout_id >= args.num_critic_only_steps:
                await _train_actor_and_critic(
                    actor_model,
                    critic_model,
                    rollout_id,
                    rollout_data_ref,
                )
            else:
                critic_task = await eager_create_task(
                    critic_model.train(rollout_id, rollout_data_ref)
                )
                await critic_task
        else:
            await actor_model.train(rollout_id, rollout_data_ref)

        should_stop = False
        if external_policy_sync is not None:
            after_local_train_kwargs = dict(
                rollout_id=rollout_id,
                actor_model=actor_model,
                rollout_data=rollout_data_ref,
            )
            if critic_model is not None:
                after_local_train_kwargs["critic_model"] = critic_model
            should_stop = bool(
                await external_policy_sync.after_local_train(
                    **after_local_train_kwargs
                )
            )

        external_save = args.save_trigger_sentinel is not None and os.path.exists(args.save_trigger_sentinel)
        if external_save or should_run_periodic_action(
            rollout_id, args.save_interval, num_rollout_per_epoch, args.num_rollout
        ):
            await save(rollout_id, force_sync=external_save)
            if external_save:
                os.remove(args.save_trigger_sentinel)

        if args.offload_train and (
            not args.use_critic or rollout_id >= args.num_critic_only_steps
        ):
            await actor_model.prepare_weight_update()
        await offload_train()
        if args.offload_rollout:
            await rollout_manager.onload_weights.remote()
        publication_info = await actor_model.update_weights(rollout_id=rollout_id)
        await _notify_external_policy_publication(
            external_policy_sync,
            rollout_id=rollout_id,
            actor_model=actor_model,
            publication_info=publication_info,
        )
        if args.offload_rollout:
            await rollout_manager.onload_kv.remote()
        if should_stop:
            break

        if should_run_periodic_action(rollout_id, args.eval_interval, num_rollout_per_epoch):
            await rollout_manager.eval.remote(rollout_id)

        if (
            args.debug_exit_after_rollout is not None
            and (rollout_id - args.start_rollout_id + 1) >= args.debug_exit_after_rollout
        ):
            logger.info(
                "debug_exit_after_rollout=%d reached at rollout_id=%d, exiting",
                args.debug_exit_after_rollout,
                rollout_id,
            )
            break

    if external_policy_sync is not None:
        await external_policy_sync.finalize()
    await rollout_manager.dispose.remote()


if __name__ == "__main__":
    args = parse_args()
    try:
        asyncio.run(train(args))
    finally:
        finish_tracking()
