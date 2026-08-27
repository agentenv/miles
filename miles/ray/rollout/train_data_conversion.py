from typing import Any

import ray
import torch

from miles.utils.ray_utils import Box
from miles.utils.seqlen_balancing import get_seqlen_balanced_partitions
from miles.utils.timer import Timer
from miles.utils.types import Sample


def _validate_sao_train_data(args, data):
    from miles.backends.training_utils.sao import validate_sao_train_data

    validate_sao_train_data(args, data)


def convert_samples_to_train_data(
    args,
    samples: list[Sample] | list[list[Sample]],
    metadata: dict[str, Any],
    custom_convert_samples_to_train_data_func,
    custom_reward_post_process_func,
):
    """
    Convert inference generated samples to training data.
    """
    if (f := custom_convert_samples_to_train_data_func) is not None:
        train_data = f(args, samples)
        _validate_sao_train_data(args, train_data)
        return train_data

    raw_rewards, rewards = _post_process_rewards(
        args,
        samples,
        custom_reward_post_process_func=custom_reward_post_process_func,
        prompt_group_sizes=metadata.get("prompt_group_sizes"),
    )

    assert len(raw_rewards) == len(samples)
    assert len(rewards) == len(samples)

    train_data = {
        "tokens": [sample.tokens for sample in samples],
        "response_lengths": [sample.response_length for sample in samples],
        # some reward model, e.g. remote rm, may return multiple rewards,
        # we could use key to select the reward.
        "rewards": rewards,
        "raw_reward": raw_rewards,
        "truncated": [1 if sample.status == Sample.Status.TRUNCATED else 0 for sample in samples],
        "sample_indices": [sample.index for sample in samples],
    }

    # loss mask
    # TODO: compress the loss mask
    loss_masks = []
    for sample in samples:
        # always instantiate loss_mask if not provided
        if sample.loss_mask is None:
            sample.loss_mask = [1] * sample.response_length

        assert (
            len(sample.loss_mask) == sample.response_length
        ), f"loss mask length {len(sample.loss_mask)} != response length {sample.response_length}"
        if sample.remove_sample:
            sample.loss_mask = [0] * sample.response_length
        loss_masks.append(sample.loss_mask)
    train_data["loss_masks"] = loss_masks

    _attach_compaction_train_fields(args, samples, train_data)

    # overwriting the raw reward
    if samples[0].metadata and "raw_reward" in samples[0].metadata:
        train_data["raw_reward"] = [sample.metadata["raw_reward"] for sample in samples]

    # For rollout buffer
    if samples[0].metadata and "round_number" in samples[0].metadata:
        train_data["round_number"] = [sample.metadata["round_number"] for sample in samples]

    # Add rollout log probabilities for off-policy correction
    if samples[0].rollout_log_probs is not None:
        train_data["rollout_log_probs"] = [sample.rollout_log_probs for sample in samples]

    if samples[0].rollout_routed_experts is not None:
        train_data["rollout_routed_experts"] = [sample.rollout_routed_experts for sample in samples]

    if samples[0].rollout_indexer_topk is not None:
        train_data["rollout_indexer_topk"] = [sample.rollout_indexer_topk for sample in samples]

    if samples[0].train_metadata is not None:
        train_data["metadata"] = [sample.train_metadata for sample in samples]

    if any(sample.multimodal_train_inputs is not None for sample in samples):
        train_data["multimodal_train_inputs"] = [sample.multimodal_train_inputs for sample in samples]

    if any(sample.weight_versions for sample in samples):
        train_data["weight_versions"] = [sample.weight_versions for sample in samples]

    if samples[0].teacher_log_probs is not None:
        train_data["teacher_log_probs"] = [sample.teacher_log_probs for sample in samples]

    if any(sample.adapter is not None for sample in samples):
        assert all(sample.adapter is not None for sample in samples), "Cannot mix adapter and adapter-less samples"
        train_data["adapter_slots"] = [sample.adapter.slot for sample in samples]
        # Slots whose adapter batch completes with this batch: the trainer scales their
        # accumulated gradients by 1/adapter-batch-size and advances the LR schedule.
        step_slots = sorted(metadata.get("step_slots", []))
        train_data["step_slots"] = step_slots
        train_data["step_adapter_names"] = sorted(metadata.get("step_adapter_names", []))
        step_slot_set = set(step_slots)
        train_data["step_adapter_batch_sizes"] = {
            sample.adapter.slot: sample.metadata["adapter_global_batch_size"]
            for sample in samples
            if sample.adapter.slot in step_slot_set
        }

    if (prompt_group_sizes := metadata.get("prompt_group_sizes")) is not None:
        train_data["prompt_group_sizes"] = prompt_group_sizes

    if samples[0].opd_reverse_kl is not None:
        train_data["opd_reverse_kl"] = [sample.opd_reverse_kl for sample in samples]

    x = metadata.get("dynamic_global_batch_size")
    assert args.use_dynamic_global_batch_size == (x is not None)
    if x is not None:
        train_data["dynamic_global_batch_size"] = x

    _validate_sao_train_data(args, train_data)
    return train_data


def _attach_compaction_train_fields(args, samples: list[Sample], train_data: dict[str, Any]) -> None:
    """Attach the paper's segment position proof and exact future-token counts."""
    metadata = [sample.metadata or {} for sample in samples]
    marked = [item.get("compaction_schema_version") is not None for item in metadata]
    if not any(marked):
        if getattr(args, "sao_compaction", False):
            raise ValueError("--sao-compaction received rollout data without compaction evidence")
        return
    if not all(marked):
        raise ValueError("rollout batch mixes compacted and unmarked samples")
    if not getattr(args, "sao_compaction", False):
        raise ValueError("compacted rollout data requires --sao-compaction")

    trajectory_ids: list[str] = []
    segment_indices: list[int] = []
    segment_types: list[str] = []
    context_budgets: list[int] = []
    groups: dict[str, list[int]] = {}
    for row, item in enumerate(metadata):
        trajectory_id = item.get("compaction_trajectory_id")
        segment_index = item.get("compaction_segment_index")
        segment_type = item.get("compaction_segment_type")
        context_budget = item.get("compaction_context_budget")
        if item.get("compaction_schema_version") != 1:
            raise ValueError("unsupported compaction sample schema")
        if not isinstance(trajectory_id, str) or not trajectory_id:
            raise ValueError("compaction sample is missing a trajectory ID")
        if isinstance(segment_index, bool) or not isinstance(segment_index, int) or segment_index < 0:
            raise ValueError("compaction sample has an invalid segment index")
        if segment_type not in {"execution", "summary"}:
            raise ValueError("compaction sample has an invalid segment type")
        if (
            isinstance(context_budget, bool)
            or not isinstance(context_budget, int)
            or context_budget <= 0
        ):
            raise ValueError("compaction sample has an invalid context budget")
        trajectory_ids.append(trajectory_id)
        segment_indices.append(segment_index)
        segment_types.append(segment_type)
        context_budgets.append(context_budget)
        groups.setdefault(trajectory_id, []).append(row)

    if len(set(context_budgets)) != 1:
        raise ValueError("rollout batch mixes compaction context budgets")

    subsequent_active_tokens = [0] * len(samples)
    trajectory_active_tokens = [0] * len(samples)
    for trajectory_id, rows in groups.items():
        ordered = sorted(rows, key=segment_indices.__getitem__)
        actual_indices = [segment_indices[row] for row in ordered]
        if actual_indices != list(range(len(ordered))):
            raise ValueError(
                f"compaction trajectory {trajectory_id!r} has non-contiguous or duplicate segments"
            )
        expected_types = ["execution" if index % 2 == 0 else "summary" for index in actual_indices]
        actual_types = [segment_types[row] for row in ordered]
        if actual_types != expected_types or actual_types[-1] != "execution":
            raise ValueError(f"compaction trajectory {trajectory_id!r} has an invalid segment sequence")
        rewards = [float(train_data["rewards"][row]) for row in ordered]
        if any(reward != rewards[0] for reward in rewards[1:]):
            raise ValueError(
                f"compaction trajectory {trajectory_id!r} does not share one final task reward"
            )
        future = 0
        for row in reversed(ordered):
            subsequent_active_tokens[row] = future
            future += sum(bool(value) for value in train_data["loss_masks"][row])
        for row in ordered:
            trajectory_active_tokens[row] = future

    train_data["compaction_trajectory_ids"] = trajectory_ids
    train_data["compaction_segment_indices"] = segment_indices
    train_data["compaction_segment_types"] = segment_types
    train_data["compaction_subsequent_active_tokens"] = subsequent_active_tokens
    train_data["compaction_trajectory_active_tokens"] = trajectory_active_tokens


def _post_process_rewards(
    args,
    samples: list[Sample] | list[list[Sample]],
    custom_reward_post_process_func,
    prompt_group_sizes: list[int] | None = None,
):
    if (f := custom_reward_post_process_func) is not None:
        return f(args, samples)

    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    if args.advantage_estimator in ["grpo", "gspo", "reinforce_plus_plus_baseline"] and args.rewards_normalization:
        # group norm
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        if prompt_group_sizes is not None:
            # Multi-LoRA: groups may have heterogeneous sizes (per-adapter
            # n_samples_per_prompt), so normalize within explicit boundaries.
            assert sum(prompt_group_sizes) == len(
                raw_rewards
            ), f"prompt group sizes sum to {sum(prompt_group_sizes)}, but got {len(raw_rewards)} rewards"
            normalized_groups = []
            for group_rewards in rewards.split(prompt_group_sizes):
                centered = group_rewards - group_rewards.mean()
                if (
                    args.advantage_estimator in ["grpo", "gspo"]
                    and args.grpo_std_normalization
                    and group_rewards.numel() > 1
                ):
                    centered = centered / (group_rewards.std() + 1e-6)
                normalized_groups.append(centered)
            return raw_rewards, torch.cat(normalized_groups).tolist()
        if rewards.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        else:
            # when samples count are not equal in each group
            rewards = rewards.view(-1, rewards.shape[-1])
        mean = rewards.mean(dim=-1, keepdim=True)
        rewards = rewards - mean

        if args.advantage_estimator in ["grpo", "gspo"] and args.grpo_std_normalization:
            std = rewards.std(dim=-1, keepdim=True)
            rewards = rewards / (std + 1e-6)

        return raw_rewards, rewards.flatten().tolist()

    return raw_rewards, raw_rewards


def split_train_data_by_dp(args, data, dp_size):
    """Split the train data by data parallel size."""
    rollout_data_list = split_train_data_by_dp_raw(args, data, dp_size=dp_size)
    return [Box(ray.put(rollout_data)) for rollout_data in rollout_data_list]


def split_train_data_by_dp_raw(args, data: dict[str, Any], *, dp_size: int) -> list[dict[str, Any]]:
    """Split the train data by data parallel size."""
    total_lengths = [len(t) for t in data["tokens"]]
    data["total_lengths"] = total_lengths

    if args.balance_data:
        partitions = get_seqlen_balanced_partitions(total_lengths, dp_size, equal_size=True)
    else:
        partitions = [range(i, len(total_lengths), dp_size) for i in range(dp_size)]

    # Multi-LoRA: sort partitions by adapter slot so each microbatch is
    # contiguous-by-slot (required by the per-adapter token-count math).
    adapter_slots = data.get("adapter_slots")
    if adapter_slots is not None:
        partitions = [sorted(p, key=lambda i: adapter_slots[i]) for p in partitions]

    shards = []

    for i in range(dp_size):
        rollout_data = {}
        partition = partitions[i]
        rollout_data["partition"] = partition
        for key in [
            "tokens",
            "multimodal_train_inputs",
            "response_lengths",
            "rewards",
            "returns",
            "values",
            "truncated",
            "loss_masks",
            "round_number",
            "sample_indices",
            "sample_ids",
            "rollout_log_probs",
            "rollout_routed_experts",
            "rollout_indexer_topk",
            "prompt",
            "teacher_log_probs",
            "opd_reverse_kl",
            "seq_witness_ids",
            "weight_versions",
            "adapter_slots",
            "compaction_trajectory_ids",
            "compaction_segment_indices",
            "compaction_segment_types",
            "compaction_subsequent_active_tokens",
            "compaction_trajectory_active_tokens",
        ]:
            if key not in data:
                continue
            val = [data[key][j] for j in partition]
            rollout_data[key] = val
        # keys that need to be splited at train side
        for key in [
            "raw_reward",
            "total_lengths",
            "dynamic_global_batch_size",
            "step_slots",
            "step_adapter_names",
            "step_adapter_batch_sizes",
            "prompt_group_sizes",
        ]:
            if key not in data:
                continue
            rollout_data[key] = data[key]
        if "adapter_slots" in rollout_data:
            rollout_data["n_adapters"] = args.multi_lora_n_adapters
        shards.append(rollout_data)
    return shards


def process_rollout_data_shard(args, rollout_data):
    """Train-side completion of the DP split: drop the ``partition`` key and
    reorder the batch-global ``total_lengths`` into this shard's row order."""
    partition = rollout_data.pop("partition")
    total_lengths = rollout_data["total_lengths"]

    # save the seqlen of the whole rollout batch
    Timer().seq_lens = total_lengths
    rollout_data["total_lengths"] = [total_lengths[i] for i in partition]

    return rollout_data
