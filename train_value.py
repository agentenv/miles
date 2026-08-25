"""Standalone offline value pretraining for an SAO-compatible Miles critic."""

import asyncio
import logging
from pathlib import Path

from miles.ray.placement_group import create_value_pretraining_group
from miles.ray.rollout.train_data_conversion import split_train_data_by_dp
from miles.utils.arguments import parse_args
from miles.utils.logging_utils import configure_logger
from miles.utils.megatron_args_utils import compute_megatron_world_size_except_dp
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking
from miles.value_pretraining import (
    ValuePretrainDataset,
    VALUE_PRETRAIN_CONTRACT_NAME,
    load_value_pretrain_manifest,
    load_value_pretrain_contract,
    write_value_pretrain_contract,
)

logger = logging.getLogger(__name__)


async def train_value(args) -> None:
    if args.value_pretrain_manifest is None:
        raise ValueError("train_value.py requires --value-pretrain-manifest")
    if args.train_backend != "megatron":
        raise ValueError("offline SAO value pretraining currently requires --train-backend megatron")

    configure_logger(args, source=MainProcessIdentity())
    manifest = load_value_pretrain_manifest(
        args.value_pretrain_manifest,
        expected_sha256=args.value_pretrain_manifest_sha256,
    )
    # This verifies the dataset hash, all rows, IDs, masks, target lengths, and
    # reward range before a GPU placement group is created.
    dataset = ValuePretrainDataset(manifest.train, manifest.objective)
    resume_contract = None
    if args.critic_load is not None and (Path(args.critic_load) / VALUE_PRETRAIN_CONTRACT_NAME).is_file():
        resume_contract = load_value_pretrain_contract(args.critic_load)
        if resume_contract["source_manifest_sha256"] != manifest.sha256:
            raise ValueError("critic resume checkpoint was trained from a different value-pretraining manifest")
        if resume_contract["train_dataset_sha256"] != manifest.train.sha256:
            raise ValueError("critic resume checkpoint was trained from a different value-pretraining dataset")
        if resume_contract["objective"] != manifest.objective.to_dict():
            raise ValueError("critic resume checkpoint used a different value objective")
        expected_batch_plan = {
            "seed": args.seed,
            "global_batch_size": args.global_batch_size,
            "drop_incomplete_batch": True,
        }
        if resume_contract["batch_plan"] != expected_batch_plan:
            raise ValueError("critic resume checkpoint used a different deterministic batch plan")
    model_parallel_size = compute_megatron_world_size_except_dp(args)
    total_gpus = args.critic_num_nodes * args.critic_num_gpus_per_node
    if total_gpus % model_parallel_size != 0:
        raise ValueError(f"critic GPU count {total_gpus} is not divisible by model-parallel size {model_parallel_size}")
    dp_size = total_gpus // model_parallel_size
    if args.global_batch_size % dp_size != 0:
        raise ValueError(f"global_batch_size={args.global_batch_size} must be divisible by critic DP size {dp_size}")
    local_batch_size = args.global_batch_size // dp_size
    if not args.use_dynamic_batch_size and local_batch_size % args.micro_batch_size != 0:
        raise ValueError(f"local critic batch size {local_batch_size} must be divisible by micro_batch_size={args.micro_batch_size}")

    init_tracking(args)
    critic_model = create_value_pretraining_group(args)
    loaded_iterations = await critic_model.init()
    concrete_iterations = [value for value in loaded_iterations if value is not None]
    if not concrete_iterations or len(set(concrete_iterations)) != 1:
        raise RuntimeError("critic ranks disagree about the loaded checkpoint iteration")
    loaded_iteration = concrete_iterations[0]
    resume_after = int(resume_contract["completed_steps"]) if resume_contract is not None else 0
    if resume_contract is not None and loaded_iteration != resume_after:
        raise ValueError(f"critic checkpoint iteration {loaded_iteration} disagrees with contract step {resume_after}")
    if resume_contract is None and loaded_iteration != 0:
        raise ValueError("a non-zero critic checkpoint without a value-pretraining contract is ambiguous; load the base model in finetune mode or resume from a contracted value checkpoint")
    if resume_after > args.num_rollout:
        raise ValueError(f"critic checkpoint step {resume_after} exceeds the {args.num_rollout}-step batch plan")

    logger.info(
        "SAO value pretraining: samples=%d steps=%d resume_after=%d dp=%d objective=%s",
        len(dataset),
        args.num_rollout,
        resume_after,
        dp_size,
        manifest.objective.loss_type,
    )

    def publish_contract(completed_steps: int) -> None:
        contract_path, contract_sha256 = write_value_pretrain_contract(
            args.critic_save,
            manifest=manifest,
            completed_steps=completed_steps,
            model_identity=args.hf_checkpoint or args.critic_load,
            seed=args.seed,
            global_batch_size=args.global_batch_size,
        )
        logger.info(
            "SAO value checkpoint: path=%s contract=%s contract_sha256=%s",
            args.critic_save,
            contract_path,
            contract_sha256,
        )

    last_completed = resume_after
    last_saved = resume_after
    batch_indices = dataset.batch_indices(
        global_batch_size=args.global_batch_size,
        epochs=args.value_pretrain_epochs,
        seed=args.seed,
    )
    for iteration, indices in enumerate(batch_indices, start=1):
        if iteration <= resume_after:
            continue
        rollout_batch = dataset.build_rollout_batch(indices)
        data_ref = split_train_data_by_dp(args, rollout_batch, dp_size=dp_size)
        await critic_model.train(
            iteration,
            {
                "data_ref": data_ref,
                "sample_indices": rollout_batch["sample_indices"],
            },
        )
        last_completed = iteration
        if args.save_interval is not None and iteration % args.save_interval == 0:
            await critic_model.save_model(iteration, force_sync=True)
            publish_contract(iteration)
            last_saved = iteration

    if last_completed < 1:
        raise RuntimeError("value pretraining completed no optimizer steps")
    if last_saved != last_completed:
        await critic_model.save_model(last_completed, force_sync=True)
        publish_contract(last_completed)
    logger.info("SAO value checkpoint ready at %s", args.critic_save)


if __name__ == "__main__":
    parsed_args = parse_args()
    try:
        asyncio.run(train_value(parsed_args))
    finally:
        finish_tracking()
