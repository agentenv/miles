"""Standalone offline value pretraining for an SAO-compatible Miles critic."""

import asyncio
import hashlib
import json
import logging
import os
from collections import Counter
from dataclasses import asdict
from itertools import islice
from pathlib import Path

from miles.ray.placement_group import create_value_pretraining_group
from miles.ray.rollout.train_data_conversion import split_train_data_by_dp
from miles.utils.arguments import parse_args
from miles.utils.audit_utils.process_identity import MainProcessIdentity
from miles.utils.logging_utils import configure_logger
from miles.utils.megatron_args_utils import compute_megatron_world_size_except_dp
from miles.utils.tracking_utils.tracking import finish_tracking, init_tracking
from miles.value_evaluation import (
    VALUE_EVAL_REPORT_SCHEMA,
    combine_rank_stats,
    gate_metrics,
    scalar_active_label,
    select_mixed_label_batches,
)
from miles.value_pretraining import (
    VALUE_PRETRAIN_CONTRACT_NAME,
    ValuePretrainDataset,
    load_value_pretrain_contract,
    load_value_pretrain_manifest,
    write_value_pretrain_contract,
)

logger = logging.getLogger(__name__)


def _write_fresh_json(path: Path, payload: dict[str, object]) -> str:
    """Publish a complete report without replacing any prior evidence."""

    encoded = (json.dumps(payload, sort_keys=True, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    digest = hashlib.sha256(encoded).hexdigest()
    temporary = path.parent / f".{path.name}.{os.getpid()}.tmp"
    descriptor = os.open(temporary, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        # Hard-link publication fails instead of overwriting if another process
        # creates the destination after CLI validation.
        os.link(temporary, path)
    finally:
        if os.path.lexists(temporary):
            temporary.unlink()
    return digest


async def _evaluate_value_checkpoint(
    args,
    *,
    manifest,
    dataset: ValuePretrainDataset,
    critic_model,
    dp_size: int,
    loaded_iteration: int,
    checkpoint_contract: dict[str, object],
    batches: tuple[tuple[int, ...], ...],
) -> None:
    batch_reports: list[dict[str, object]] = []
    for batch_number, indices in enumerate(batches):
        rollout_batch = dataset.build_rollout_batch(indices)
        expected_active_targets = sum(sum(sample.loss_mask) for sample in (dataset.get(index) for index in indices))
        data_ref = split_train_data_by_dp(args, rollout_batch, dp_size=dp_size)
        rank_reports = await critic_model.evaluate_critic(
            batch_number,
            {
                "data_ref": data_ref,
                "sample_indices": rollout_batch["sample_indices"],
            },
        )
        token_stats = combine_rank_stats(
            rank_reports,
            expected_dp_size=dp_size,
            expected_sample_indices=indices,
            stats_field="token_weighted",
            expected_active_targets=expected_active_targets,
        )
        trajectory_stats = combine_rank_stats(
            rank_reports,
            expected_dp_size=dp_size,
            expected_sample_indices=indices,
            stats_field="trajectory_weighted",
            expected_active_targets=len(indices),
        )
        token_metrics = token_stats.metrics()
        trajectory_metrics = trajectory_stats.metrics()
        labels = Counter(scalar_active_label(dataset.get(index)) for index in indices)
        report = {
            "batch": batch_number,
            "sample_indices": list(indices),
            "sample_ids": list(rollout_batch["sample_ids"]),
            "label_counts": {format(label, ".17g"): count for label, count in sorted(labels.items())},
            "expected_active_targets": expected_active_targets,
            "trajectory_weighted": {
                "sufficient_stats": asdict(trajectory_stats),
                "metrics": trajectory_metrics,
            },
            "token_weighted": {
                "sufficient_stats": asdict(token_stats),
                "metrics": token_metrics,
            },
        }
        batch_reports.append(report)
        logger.info(
            "SAO critic EV batch=%d active_targets=%d trajectory_ev=%.8g token_ev=%.8g",
            batch_number,
            token_metrics["active_targets"],
            trajectory_metrics["explained_variance"],
            token_metrics["explained_variance"],
        )

    trajectory_aggregate, trajectory_gate = gate_metrics([report["trajectory_weighted"] for report in batch_reports])
    token_aggregate, token_gate = gate_metrics([report["token_weighted"] for report in batch_reports])
    passed = bool(trajectory_gate["passed"] and token_gate["passed"])
    report_path = Path(args.value_pretrain_eval_report)
    report = {
        "schema": VALUE_EVAL_REPORT_SCHEMA,
        "passed": passed,
        "manifest": {
            "path": str(manifest.path),
            "sha256": manifest.sha256,
            "split": args.value_pretrain_eval_split,
            "dataset_path": str(dataset.spec.path),
            "dataset_sha256": dataset.spec.sha256,
            "dataset_num_samples": len(dataset),
        },
        "checkpoint": {
            "path": str(Path(args.critic_load).resolve()),
            "contract_sha256": args.critic_value_pretrain_contract_sha256,
            "contract_completed_steps": checkpoint_contract["completed_steps"],
            "loaded_iteration": loaded_iteration,
        },
        "execution": {
            "forward_only": True,
            "optimizer_initialized": False,
            "optimizer_steps": 0,
            "pure_data_parallel": True,
            "dp_size": dp_size,
        },
        "batch_plan": {
            "seed": args.seed,
            "global_batch_size": args.global_batch_size,
            "num_batches": len(batches),
            "disjoint_rows": True,
            "mixed_label_batches": True,
        },
        "batches": batch_reports,
        "aggregate": {
            "trajectory_weighted": {
                "sufficient_stats": asdict(trajectory_aggregate),
                "metrics": trajectory_gate.pop("aggregate_metrics"),
            },
            "token_weighted": {
                "sufficient_stats": asdict(token_aggregate),
                "metrics": token_gate.pop("aggregate_metrics"),
            },
        },
        "gate": {
            "passed": passed,
            "trajectory_weighted": trajectory_gate,
            "token_weighted": token_gate,
            "criteria": {
                "trajectory_weighted_aggregate_explained_variance": "> 0",
                "trajectory_weighted_positive_batch_explained_variance": ("strict majority"),
                "token_weighted_aggregate_explained_variance": "> 0",
                "token_weighted_positive_batch_explained_variance": ("strict majority"),
                "minimum_target_variance_per_batch": "> 1e-8",
            },
        },
    }
    digest = _write_fresh_json(report_path, report)
    logger.info(
        "SAO critic EV gate: passed=%s trajectory_ev=%.8g "
        "trajectory_positive_batches=%d/%d "
        "report=%s report_sha256=%s",
        report["passed"],
        report["aggregate"]["trajectory_weighted"]["metrics"]["explained_variance"],
        report["gate"]["trajectory_weighted"]["positive_batches"],
        report["gate"]["trajectory_weighted"]["evaluable_batches"],
        report_path,
        digest,
    )
    if not report["passed"]:
        raise RuntimeError(
            "critic EV gate failed: require both trajectory- and token-weighted "
            "aggregate explained variance > 0 and a strict majority of their "
            "mixed-label batch EVs > 0"
        )


async def train_value(args) -> None:
    if args.value_pretrain_manifest is None:
        raise ValueError("train_value.py requires --value-pretrain-manifest")
    if args.train_backend != "megatron":
        raise ValueError("offline SAO value pretraining currently requires --train-backend megatron")
    eval_only = bool(args.value_pretrain_eval_only)
    canary_one_step = bool(args.value_pretrain_canary_one_step)
    output_contract = None if eval_only else Path(args.critic_save) / VALUE_PRETRAIN_CONTRACT_NAME
    if canary_one_step and output_contract is not None and os.path.lexists(output_contract):
        raise RuntimeError("one-step value-pretraining canary output already contains a production contract")

    configure_logger(args, source=MainProcessIdentity())
    manifest = load_value_pretrain_manifest(
        args.value_pretrain_manifest,
        expected_sha256=args.value_pretrain_manifest_sha256,
    )
    # This verifies the dataset hash, all rows, IDs, masks, target lengths, and
    # reward range before a GPU placement group is created.
    dataset_spec = manifest.train if args.value_pretrain_eval_split == "train" else manifest.heldout
    if dataset_spec is None:
        raise ValueError("value-pretraining manifest has no heldout split")
    dataset = ValuePretrainDataset(dataset_spec, manifest.objective)
    evaluation_batches = (
        select_mixed_label_batches(
            dataset,
            global_batch_size=args.global_batch_size,
            num_batches=args.value_pretrain_eval_num_batches,
            seed=args.seed,
        )
        if eval_only
        else None
    )
    resume_contract = None
    if args.critic_load is not None and (Path(args.critic_load) / VALUE_PRETRAIN_CONTRACT_NAME).is_file():
        resume_contract = load_value_pretrain_contract(
            args.critic_load,
            expected_sha256=(args.critic_value_pretrain_contract_sha256 if eval_only else None),
        )
        if resume_contract["source_manifest_sha256"] != manifest.sha256:
            raise ValueError("critic resume checkpoint was trained from a different value-pretraining manifest")
        if resume_contract["train_dataset_sha256"] != manifest.train.sha256:
            raise ValueError("critic resume checkpoint was trained from a different value-pretraining dataset")
        if resume_contract["objective"] != manifest.objective.to_dict():
            raise ValueError("critic resume checkpoint used a different value objective")
        if not eval_only:
            expected_batch_plan = {
                "seed": args.seed,
                "global_batch_size": args.global_batch_size,
                "drop_incomplete_batch": True,
            }
            if resume_contract["batch_plan"] != expected_batch_plan:
                raise ValueError("critic resume checkpoint used a different deterministic batch plan")
    if eval_only and resume_contract is None:
        raise ValueError("critic evaluation requires a contracted value checkpoint")
    # A fresh offline critic starts from an actor checkpoint: load the backbone
    # strictly while retaining the newly initialized value head.  Contracted
    # resumes must load the trained critic head as part of the checkpoint.
    args._critic_bootstrap_from_actor_checkpoint = resume_contract is None
    model_parallel_size = compute_megatron_world_size_except_dp(args)
    total_gpus = args.critic_num_nodes * args.critic_num_gpus_per_node
    if total_gpus % model_parallel_size != 0:
        raise ValueError(
            f"critic GPU count {total_gpus} is not divisible by model-parallel size {model_parallel_size}"
        )
    dp_size = total_gpus // model_parallel_size
    if args.global_batch_size % dp_size != 0:
        raise ValueError(f"global_batch_size={args.global_batch_size} must be divisible by critic DP size {dp_size}")
    local_batch_size = args.global_batch_size // dp_size
    if not args.use_dynamic_batch_size and local_batch_size % args.micro_batch_size != 0:
        raise ValueError(
            f"local critic batch size {local_batch_size} must be divisible by micro_batch_size={args.micro_batch_size}"
        )

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
        raise ValueError(
            "a non-zero critic checkpoint without a value-pretraining contract is ambiguous; load the base model in finetune mode or resume from a contracted value checkpoint"
        )
    if resume_after > args.num_rollout:
        if not eval_only:
            raise ValueError(f"critic checkpoint step {resume_after} exceeds the {args.num_rollout}-step batch plan")

    if eval_only:
        assert resume_contract is not None
        assert evaluation_batches is not None
        await _evaluate_value_checkpoint(
            args,
            manifest=manifest,
            dataset=dataset,
            critic_model=critic_model,
            dp_size=dp_size,
            loaded_iteration=loaded_iteration,
            checkpoint_contract=resume_contract,
            batches=evaluation_batches,
        )
        return

    logger.info(
        "SAO value pretraining: samples=%d steps=%d resume_after=%d dp=%d objective=%s",
        len(dataset),
        args.num_rollout,
        resume_after,
        dp_size,
        manifest.objective.loss_type,
    )

    def publish_contract(completed_steps: int) -> None:
        if canary_one_step:
            raise RuntimeError("one-step value-pretraining canaries must not publish a production contract")
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
    if canary_one_step:
        batch_indices = islice(batch_indices, 1)
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
            if not canary_one_step:
                publish_contract(iteration)
            last_saved = iteration

    if last_completed < 1:
        raise RuntimeError("value pretraining completed no optimizer steps")
    if last_saved != last_completed:
        await critic_model.save_model(last_completed, force_sync=True)
        if not canary_one_step:
            publish_contract(last_completed)
    if canary_one_step:
        if last_completed != 1:
            raise RuntimeError(f"one-step value-pretraining canary completed {last_completed} optimizer steps")
        assert output_contract is not None
        if os.path.lexists(output_contract):
            raise RuntimeError("one-step value-pretraining canary published a production contract")
    logger.info("SAO value checkpoint ready at %s", args.critic_save)


if __name__ == "__main__":
    parsed_args = parse_args()
    try:
        asyncio.run(train_value(parsed_args))
    finally:
        finish_tracking()
