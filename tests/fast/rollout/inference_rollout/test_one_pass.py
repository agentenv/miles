from __future__ import annotations

import argparse
import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from miles.rollout.base_types import RolloutFnTrainInput, RolloutFnTrainOutput
from miles.rollout.data_source import RolloutDataSource, RolloutDataSourceWithBuffer
from miles.rollout.filter_hub.base_types import DynamicFilterOutput
from miles.rollout.inference_rollout import inference_rollout_train as rollout
from miles.rollout.inference_rollout.inference_rollout_common import InferenceRolloutFn
from miles.utils.arguments import (
    _validate_rollout_one_pass_no_replacement,
    _validate_rollout_only_from_checkpoint,
    get_miles_extra_args_provider,
)
from miles.utils.types import Sample

_MODULAR_ROLLOUT = "miles.rollout.inference_rollout.inference_rollout_common.InferenceRolloutFn"


class _Dataset:
    def __init__(self, samples: list[Sample]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)


class _State:
    def __init__(self, args) -> None:
        self.args = args
        self.sampling_params = {"temperature": 0.7}
        self.aborted = False
        self.reset_count = 0

    def reset(self) -> None:
        self.aborted = False
        self.reset_count += 1


def _sample(index: int, *, status: Sample.Status = Sample.Status.COMPLETED) -> Sample:
    return Sample(
        index=index,
        group_index=index,
        prompt=f"prompt-{index}",
        response=f"response-{index}",
        response_length=1,
        reward=1.0,
        status=status,
    )


def _args(*, dynamic_filter_path: str | None = None):
    return SimpleNamespace(
        rollout_global_dataset=True,
        rollout_one_pass_no_replacement=True,
        rollout_batch_size=3,
        n_samples_per_prompt=1,
        dynamic_sampling_filter_path=dynamic_filter_path,
        rollout_sample_filter_path=None,
        rollout_all_samples_process_path=None,
        sglang_router_ip="127.0.0.1",
        sglang_router_port=30000,
    )


def _source_with_samples(args, samples: list[Sample]):
    source = RolloutDataSource.__new__(RolloutDataSource)
    source.args = args
    source.dataset = _Dataset(samples)
    source.sample_offset = 0
    source.epoch_id = 0
    source.sample_group_index = 0
    source.sample_index = 0
    source._one_pass_consumed = False
    source.metadata = {}
    return source


def test_one_pass_data_source_consumes_all_once_without_cycling() -> None:
    args = SimpleNamespace(
        rollout_one_pass_no_replacement=True,
        n_samples_per_prompt=1,
        rollout_shuffle=False,
    )
    source = _source_with_samples(args, [_sample(index) for index in range(3)])

    groups = source.get_samples(3)

    assert [group[0].prompt for group in groups] == [
        "prompt-0",
        "prompt-1",
        "prompt-2",
    ]
    assert source.sample_offset == 3
    with pytest.raises(RuntimeError, match="requested more than once"):
        source.get_samples(3)


def test_one_pass_flag_is_opt_in_and_rejects_refill_configuration() -> None:
    parser = get_miles_extra_args_provider()(argparse.ArgumentParser())
    assert parser.parse_args([]).rollout_one_pass_no_replacement is False
    assert parser.parse_args([]).rollout_only_from_checkpoint is False
    assert parser.parse_args(["--rollout-only-from-checkpoint"]).rollout_only_from_checkpoint is True
    assert parser.parse_args(["--rollout-one-pass-no-replacement"]).rollout_one_pass_no_replacement is True

    args = SimpleNamespace(
        rollout_one_pass_no_replacement=True,
        rollout_global_dataset=True,
        rollout_function_path=_MODULAR_ROLLOUT,
        data_source_path="miles.rollout.data_source.RolloutDataSourceWithBuffer",
        num_epoch=None,
        num_rollout=1,
        start_rollout_id=0,
        rollout_shuffle=False,
        partial_rollout=False,
        rollout_batch_size=2,
        over_sampling_batch_size=2,
        rollout_all_samples_process_path=None,
    )
    _validate_rollout_one_pass_no_replacement(args)

    with pytest.raises(ValueError, match="over_sampling_batch_size must equal"):
        args.over_sampling_batch_size = 3
        _validate_rollout_one_pass_no_replacement(args)


def test_checkpoint_backed_rollout_only_requires_auditable_guards(
    tmp_path,
) -> None:
    checkpoint = tmp_path / "actor-checkpoint"
    checkpoint.mkdir()
    (checkpoint / "latest_checkpointed_iteration.txt").write_text("0\n")
    args = SimpleNamespace(
        rollout_only_from_checkpoint=True,
        train_backend="megatron",
        colocate=True,
        offload_train=True,
        offload_rollout=True,
        rollout_weight_version_format="counter",
        multi_lora=False,
        bridge_distributed_weight_sync=False,
        keep_old_actor=False,
        use_opd=False,
        debug_rollout_only=False,
        debug_train_only=False,
        debug_disable_optimizer=True,
        no_load_optim=True,
        no_load_rng=True,
        finetune=True,
        check_weight_update_equal=False,
        debug_skip_weight_update=False,
        use_critic=False,
        external_policy_sync_path=None,
        eval_interval=None,
        save_interval=None,
        save_trigger_sentinel=None,
        start_rollout_id=0,
        num_rollout=1,
        rollout_one_pass_no_replacement=True,
        load_debug_rollout_data=None,
        lora_rank=0,
        dist_ckpt_strictness="raise_unexpected",
        ckpt_step=None,
        ref_ckpt_step=None,
        control_server_port=0,
        mini_ft_controller_enable=False,
        ft_components=[],
        kl_coef=0,
        kl_loss_coef=0,
        use_kl_loss=False,
        save=None,
        save_hf=None,
        custom_megatron_post_save_hook_path=None,
        ref_load=str(checkpoint),
        load=str(checkpoint),
        rollout_only_publication_evidence=str(tmp_path / "publication.json"),
    )

    _validate_rollout_only_from_checkpoint(args)

    args.check_weight_update_equal = True
    with pytest.raises(ValueError, match="check_weight_update_equal must be disabled"):
        _validate_rollout_only_from_checkpoint(args)


def test_one_pass_data_source_rejects_short_read_and_recycled_buffer() -> None:
    args = SimpleNamespace(
        rollout_one_pass_no_replacement=True,
        n_samples_per_prompt=1,
        rollout_shuffle=False,
    )
    source = _source_with_samples(args, [_sample(index) for index in range(3)])
    with pytest.raises(RuntimeError, match="complete immutable dataset"):
        source.get_samples(2)

    buffered = RolloutDataSourceWithBuffer.__new__(RolloutDataSourceWithBuffer)
    buffered.args = args
    buffered.buffer = [[_sample(9)]]
    with pytest.raises(RuntimeError, match="refuses recycled samples"):
        buffered.get_samples(3)


def test_one_pass_aborted_trajectory_reports_all_available_diagnostics() -> None:
    first = _sample(17, status=Sample.Status.ABORTED)
    first.metadata = {
        "sample_id": "train:task-a:r0",
        "task_id": "task-a",
        "agent_failure": {"diagnostic": "RuntimeError: agent attestation failed"},
    }
    second = _sample(18, status=Sample.Status.ABORTED)
    second.metadata = {
        "task_id": "task-b",
        "agent_failure": {"error_type": "TimeoutError"},
    }
    completed = _sample(19)
    completed.metadata = {"sample_id": "train:task-c:r0", "task_id": "task-c"}

    with pytest.raises(rollout.OnePassRolloutError) as raised:
        rollout._validate_one_pass_generated_group([first, [completed, second]], expected_group_size=2)

    message = str(raised.value)
    assert "sample_id='train:task-a:r0'" in message
    assert "task_id='task-a'" in message
    assert "index=17" in message
    assert "cause=RuntimeError: agent attestation failed" in message
    assert "task_id='task-b'" in message
    assert "index=18" in message
    assert "cause=TimeoutError" in message
    assert "train:task-c:r0" not in message


@pytest.mark.asyncio
async def test_one_pass_submits_each_planned_group_once(monkeypatch) -> None:
    args = _args()
    state = _State(args)
    groups = [[_sample(index)] for index in range(3)]
    source_calls: list[int] = []
    generated: list[int] = []

    def data_source(count: int):
        source_calls.append(count)
        return groups

    def submit(_state, planned):
        async def generate(group):
            generated.append(group[0].index)
            await asyncio.sleep((2 - group[0].index) * 0.001)
            return group

        return [asyncio.create_task(generate(group)) for group in planned]

    recompute = AsyncMock()
    monkeypatch.setattr(rollout.dumper_utils, "configure_sglang", AsyncMock())
    monkeypatch.setattr(rollout, "load_function", lambda _path: None)
    monkeypatch.setattr(rollout, "submit_generate_tasks", submit)
    monkeypatch.setattr(rollout, "recompute_samples_rollout_logprobs_via_prefill", recompute)

    output, aborted = await rollout.generate_rollout_async(state, 0, data_source)

    assert source_calls == [3]
    assert sorted(generated) == [0, 1, 2]
    assert [group[0].index for group in output.samples] == [0, 1, 2]
    assert aborted == []
    assert state.reset_count == 1
    recompute.assert_awaited_once()


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["exception", "aborted", "drop", "short"])
async def test_one_pass_failure_aborts_pending_without_refill(monkeypatch, failure: str) -> None:
    args = _args(dynamic_filter_path="test:drop" if failure == "drop" else None)
    state = _State(args)
    groups = [[_sample(index)] for index in range(3)]
    source_calls: list[int] = []
    abort_calls: list[tuple[int, int]] = []
    wait_forever = asyncio.Event()

    def data_source(count: int):
        source_calls.append(count)
        return groups

    def submit(_state, planned):
        async def generate(group):
            if group[0].index != 0:
                await wait_forever.wait()
                return group
            if failure == "exception":
                raise RuntimeError("agent failed")
            if failure == "aborted":
                group[0].status = Sample.Status.ABORTED
            if failure == "short":
                return []
            return group

        return [asyncio.create_task(generate(group)) for group in planned]

    async def fake_abort(abort_state, tasks, rollout_id):
        abort_calls.append((rollout_id, len(tasks)))
        abort_state.aborted = True
        for task in tasks:
            task.cancel()
        await asyncio.gather(*tasks, return_exceptions=True)
        return []

    def load_function(path):
        if path == "test:drop":
            return lambda _args, _group: DynamicFilterOutput(keep=False, reason="planned-drop")
        return None

    monkeypatch.setattr(rollout.dumper_utils, "configure_sglang", AsyncMock())
    monkeypatch.setattr(rollout, "load_function", load_function)
    monkeypatch.setattr(rollout, "submit_generate_tasks", submit)
    monkeypatch.setattr(rollout, "abort", fake_abort)
    monkeypatch.setattr(
        rollout,
        "recompute_samples_rollout_logprobs_via_prefill",
        AsyncMock(),
    )

    with pytest.raises(rollout.OnePassRolloutError):
        await rollout.generate_rollout_async(state, 7, data_source)

    assert source_calls == [3]
    assert abort_calls == [(7, 2)]


@pytest.mark.asyncio
async def test_one_pass_preserves_task_failure_when_done_batch_and_abort_fail(
    monkeypatch,
) -> None:
    args = _args()
    state = _State(args)
    groups = [[_sample(index)] for index in range(3)]
    source_calls: list[int] = []
    abort_calls: list[int] = []

    def data_source(count: int):
        source_calls.append(count)
        return groups

    def submit(_state, planned):
        futures = []
        for group in planned:
            future = asyncio.get_running_loop().create_future()
            future.set_exception(RuntimeError(f"task-failure-{group[0].index}"))
            futures.append(future)
        return futures

    async def failing_abort(_state, tasks, _rollout_id):
        abort_calls.append(len(tasks))
        raise RuntimeError("abort-cleanup-failure")

    monkeypatch.setattr(rollout.dumper_utils, "configure_sglang", AsyncMock())
    monkeypatch.setattr(rollout, "load_function", lambda _path: None)
    monkeypatch.setattr(rollout, "submit_generate_tasks", submit)
    monkeypatch.setattr(rollout, "abort", failing_abort)

    with pytest.raises(rollout.OnePassRolloutError, match="generation task raised") as raised:
        await rollout.generate_rollout_async(state, 9, data_source)

    assert source_calls == [3]
    assert abort_calls == [2]
    assert isinstance(raised.value.__cause__, RuntimeError)
    assert "task-failure-" in str(raised.value.__cause__)


@pytest.mark.asyncio
async def test_one_pass_short_source_fails_before_submission(monkeypatch) -> None:
    args = _args()
    state = _State(args)
    source_calls: list[int] = []
    submit = AsyncMock()

    def data_source(count: int):
        source_calls.append(count)
        return [[_sample(0)], [_sample(1)]]

    monkeypatch.setattr(rollout.dumper_utils, "configure_sglang", AsyncMock())
    monkeypatch.setattr(rollout, "load_function", lambda _path: None)
    monkeypatch.setattr(rollout, "submit_generate_tasks", submit)

    with pytest.raises(rollout.OnePassRolloutError, match="returned 2 groups"):
        await rollout.generate_rollout_async(state, 0, data_source)

    assert source_calls == [3]
    submit.assert_not_called()


@pytest.mark.asyncio
async def test_one_pass_rollout_does_not_recycle_into_data_source(monkeypatch) -> None:
    args = _args()
    data_source = SimpleNamespace(get_samples=Mock(), add_samples=Mock())
    owner = SimpleNamespace(state=SimpleNamespace(args=args), data_source=data_source)
    expected = RolloutFnTrainOutput(samples=[[_sample(0)]], metrics={})
    generate = AsyncMock(return_value=(expected, []))
    monkeypatch.setattr(rollout, "generate_rollout_async", generate)

    output = await InferenceRolloutFn._call_train(owner, RolloutFnTrainInput(rollout_id=4))

    assert output is expected
    data_source.add_samples.assert_not_called()
