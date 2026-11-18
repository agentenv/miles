import torch
from megatron.core.optimizer import USING_PYTORCH_OPTIMIZER
from megatron.core.optimizer.cpu_offloading import HybridDeviceOptimizer
from megatron.core.optimizer.optimizer import ChainedOptimizer, MegatronOptimizer


def reset_optimizer_state(
    optimizer: MegatronOptimizer, *, stream_optimizer_state_to_disk: bool, chunked_optimizer_state_offload: bool
) -> None:
    assert isinstance(optimizer, ChainedOptimizer)
    assert not optimizer.config.offload_optimizer_states
    if chunked_optimizer_state_offload:
        # Its CPU store holds the canonical moments (and an optimizer-owned master) only once a step has
        # moved them there. Before that -- the reset of a fresh launch from HF weights -- the store is
        # empty and the optimizer state is everything there is to clear.
        assert not any(_offloader_holds_state(chained) for chained in optimizer.chained_optimizers), (
            "--chunked-optimizer-state-offload keeps the canonical moments outside the optimizer state this "
            "clears"
        )
    assert not optimizer.config.fp16, (
        "an fp16 optimizer keeps a loss scaler with a scale and a growth tracker beside the state this clears, so "
        "the reset would leave the trainer with the scaler of the rollouts it just discarded"
    )
    assert not stream_optimizer_state_to_disk, (
        "--stream-optimizer-state-to-disk keeps the moments in its own store rather than in the master optimizer "
        "state this walks, so the reset would clear nothing and the self-check would pass on an empty walk"
    )

    for chained in optimizer.chained_optimizers:
        if chained.is_stub_optimizer:
            continue

        inner = chained.optimizer
        # HybridDeviceOptimizer (--optimizer-cpu-offload) steps sub-optimizers that own the real moments and
        # only mirrors them into its own state after each step, so clearing the mirror alone would keep them.
        # Its FP32 masters are refreshed from the loaded weights by reload_model_params after the reset.
        for leaf in _leaf_optimizers(inner):
            leaf.state.clear()
            for param_group in leaf.param_groups:
                param_group.pop("step", None)

        # Chunked offload leaves the state empty: TE FusedAdam builds it on the first step, chunk by chunk, from
        # the weights loaded after this reset, and the offloader moves it to the host. Built here it would sit on
        # the GPU in full through the first forward/backward (and a non-remainder master would predate the load).
        if chained.init_state_fn is not None and not USING_PYTORCH_OPTIMIZER and not chunked_optimizer_state_offload:
            chained.init_state_fn(inner, chained.config)

        chained.zero_grad(set_to_none=True)

    _check_optimizer_state_empty(optimizer, lazy_state=chunked_optimizer_state_offload)


def _offloader_holds_state(chained: MegatronOptimizer) -> bool:
    offloader = getattr(chained, "_optimizer_state_offloader", None)  # config-access-exempt: set only by Megatron's chunked optimizer-state offload
    if offloader is None:
        return False
    return bool(getattr(offloader, "_cpu_state", None) or getattr(offloader, "_cpu_master", None))  # config-access-exempt: the offloader's CPU canonical stores


def _check_optimizer_state_empty(optimizer: MegatronOptimizer, *, lazy_state: bool = False) -> None:
    checked_optimizers = 0
    checked_params = 0
    checked_state_keys: set[str] = set()

    for chained in optimizer.chained_optimizers:
        if chained.is_stub_optimizer:
            continue

        checked_optimizers += 1
        inner = chained.optimizer
        for param_group in inner.param_groups:
            assert "step" not in param_group
            for param in param_group["params"]:
                checked_params += 1
                assert param.grad is None

        for leaf in _leaf_optimizers(inner):
            for state in leaf.state.values():
                checked_state_keys |= set(state)
                for value in state.values():
                    if isinstance(value, torch.Tensor):
                        assert torch.count_nonzero(value) == 0
                    else:
                        assert value == 0
            if leaf is not inner:
                assert all("step" not in param_group for param_group in leaf.param_groups)

    assert checked_optimizers > 0
    assert checked_params > 0
    # if you use a non-adam optimizer, add the asserts for its state keys here
    # HybridDeviceOptimizer has no init_state_fn: its sub-optimizers allocate the moments on their first step
    # (lazy_state: the chunked optimizer-state offload, whose reset skips the initializer)
    lazy = USING_PYTORCH_OPTIMIZER or optimizer.config.optimizer_cpu_offload or lazy_state
    assert checked_state_keys == (set() if lazy else {"exp_avg", "exp_avg_sq"})


def _leaf_optimizers(inner: torch.optim.Optimizer) -> list[torch.optim.Optimizer]:
    if isinstance(inner, HybridDeviceOptimizer):
        return [inner, *inner.sub_optimizers]
    return [inner]
