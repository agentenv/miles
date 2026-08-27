"""Run one exact, immutable Terminal-Bench baseline ramp island.

This is deliberately separate from the production 44/45-row baseline runner.
It accepts only the two pre-full-wave gates: one planned trajectory per GPU
(eight total) or eight per GPU (64 total).  Every row must be an unchanged
prefix row from the bound, immutable plan-v2 baseline shard.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import miles.utils.external_utils.command_utils as U
from tools.probes import run_tbench21_compaction_baseline as baseline


RAMP_SCHEMA = "miles.tbench21-baseline-ramp-plan.v1"
PLAN_SCHEMA = "yeto.tbench21-sao-diloco-plan.v1"
GATE_SIZES = {8: 1, 64: 8}
RUN_ID = re.compile(r"[a-z0-9][a-z0-9-]{0,39}\Z")


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_json(path: Path, name: str) -> dict[str, Any]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ValueError(f"{name} must be an absolute regular non-symlink")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is unreadable or malformed") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _load_bound_rows(args: RampArgs) -> list[dict[str, Any]]:
    per_island = GATE_SIZES.get(args.gate_size)
    if per_island is None:
        raise ValueError("ramp gate_size must be exactly 8 or 64")
    if RUN_ID.fullmatch(args.run_id) is None:
        raise ValueError("ramp run_id is not a bounded lowercase identifier")

    source_root = Path(args.source_plan)
    if not source_root.is_absolute() or source_root.is_symlink() or not source_root.is_dir():
        raise ValueError("source plan must be an absolute real directory")
    source_root = source_root.resolve()
    ramp_manifest_path = Path(args.ramp_manifest)
    ramp = _load_json(ramp_manifest_path, "ramp manifest")
    source_manifest_path = source_root / "manifest.json"
    source = _load_json(source_manifest_path, "source plan manifest")
    topology = source.get("topology")
    rollouts = source.get("rollouts")
    source_files = source.get("files")
    ramp_files = ramp.get("files")
    selected = ramp.get("selected_sample_ids")
    if (
        source.get("schema") != PLAN_SCHEMA
        or not isinstance(topology, dict)
        or topology.get("islands") != 8
        or topology.get("one_physical_gpu_per_island") is not True
        or not isinstance(rollouts, dict)
        or rollouts.get("baseline") != 356
        or rollouts.get("per_task") != 4
        or rollouts.get("episode_timeout_seconds") != 1800
        or rollouts.get("seed_base") != baseline.ROLLOUT_SEED_BASE
        or rollouts.get("per_island_seeds") != [baseline.ROLLOUT_SEED_BASE + island_id for island_id in range(8)]
        or not isinstance(source_files, dict)
    ):
        raise ValueError("source plan differs from the immutable full baseline")
    if ramp.get("schema") != RAMP_SCHEMA or ramp.get("gate_size") != args.gate_size or ramp.get("per_island") != per_island or ramp.get("run_id") != args.run_id or ramp.get("source_plan_manifest_sha256") != _sha256(source_manifest_path) or not isinstance(ramp_files, dict) or not isinstance(selected, dict):
        raise ValueError("ramp manifest is not bound to this gate/source plan")

    all_ids: set[str] = set()
    selected_rows: list[list[dict[str, Any]]] = []
    ramp_root = ramp_manifest_path.resolve().parent
    for island_id in range(8):
        relative = f"baseline/island-{island_id}.jsonl"
        source_shard = source_root / relative
        ramp_shard = ramp_root / relative
        if not source_shard.is_file() or source_shard.is_symlink() or source_files.get(relative) != _sha256(source_shard) or not ramp_shard.is_file() or ramp_shard.is_symlink() or ramp_files.get(relative) != _sha256(ramp_shard):
            raise ValueError(f"ramp/source shard identity changed: {relative}")
        try:
            source_rows = [json.loads(line) for line in source_shard.read_text(encoding="utf-8").splitlines() if line]
            rows = [json.loads(line) for line in ramp_shard.read_text(encoding="utf-8").splitlines() if line]
        except (OSError, UnicodeError, json.JSONDecodeError) as error:
            raise ValueError(f"ramp/source shard is malformed: {relative}") from error
        if len(source_rows) not in {44, 45} or len(rows) != per_island or rows != source_rows[:per_island]:
            raise ValueError(f"ramp shard is not the exact planned prefix: {relative}")
        ids: list[str] = []
        for row in rows:
            metadata = row.get("metadata") if isinstance(row, dict) else None
            messages = row.get("messages") if isinstance(row, dict) else None
            sample_id = metadata.get("sample_id") if isinstance(metadata, dict) else None
            if (
                not isinstance(sample_id, str)
                or not sample_id
                or sample_id in all_ids
                or metadata.get("island_id") != island_id
                or metadata.get("rollout_seed") != baseline.ROLLOUT_SEED_BASE + island_id
                or metadata.get("split") != "baseline"
                or metadata.get("episode_timeout_seconds") != 1800
                or not isinstance(metadata.get("task_id"), str)
                or not isinstance(messages, list)
                or not messages
            ):
                raise ValueError(f"ramp row identity changed: {relative}")
            ids.append(sample_id)
            all_ids.add(sample_id)
        if selected.get(str(island_id)) != ids:
            raise ValueError(f"ramp selected-ID ledger changed: {relative}")
        selected_rows.append(rows)
    if len(all_ids) != args.gate_size:
        raise ValueError("ramp does not contain the exact requested unique trajectory count")

    expected_prompt = ramp_root / f"baseline/island-{args.island_id}.jsonl"
    prompt_path = Path(args.prompt_data)
    if not prompt_path.is_absolute() or prompt_path.is_symlink() or prompt_path.resolve() != expected_prompt:
        raise ValueError("prompt_data is not this island's bound ramp shard")
    return selected_rows[args.island_id]


@dataclass
class RampArgs(baseline.ScriptArgs):
    ramp_manifest: str = "/root/ramp-plan/manifest.json"
    source_plan: str = "/root/source-plan"
    gate_size: int = 8
    run_id: str = "gate8-v1"
    prompt_data: str = "/root/ramp-plan/baseline/island-0.jsonl"
    dump_details: str = "/root/run-output/details"
    concurrency: int = 1


def _preflight(args: RampArgs) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if not 0 <= args.island_id < 8:
        raise ValueError("island_id must be in [0, 7]")
    if args.phase != "baseline":
        raise ValueError("ramp supports only the baseline phase")
    if args.rollout_seed != baseline.ROLLOUT_SEED_BASE:
        raise ValueError("rollout_seed must equal the immutable plan seed base")
    if args.max_seq_len != 8192:
        raise ValueError("the pinned Qwen3.5-0.8B ramp requires max_seq_len=8192")
    if not (1 <= args.compaction_summary_max_tokens < args.compaction_trigger_tokens < args.max_seq_len) or args.compaction_trigger_tokens + args.compaction_summary_max_tokens > args.max_seq_len:
        raise ValueError("compaction summary/trigger/context budgets are inconsistent")
    expected_concurrency = GATE_SIZES[args.gate_size]
    if args.concurrency != expected_concurrency:
        raise ValueError("ramp concurrency must equal the exact per-island gate size")
    baseline._hmac_key_file()
    rows = _load_bound_rows(args)

    output = Path(args.dump_details).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("ramp dump_details must be a fresh path")
    for name, value in {
        "HF checkpoint": args.hf_checkpoint,
        "Megatron checkpoint": args.ref_load,
        "Miles root": args.miles_root,
        "Yeto root": args.yeto_root,
    }.items():
        if not Path(value).resolve().is_dir():
            raise FileNotFoundError(f"{name} is missing: {value}")
    binary = Path(args.codex_binary).resolve()
    python = baseline._validated_openenv_python(args.openenv_agent_python)
    if not binary.is_file() or binary.is_symlink() or not os.access(binary, os.X_OK):
        raise FileNotFoundError("the pinned Codex binary is missing or not executable")
    version = subprocess.run([str(binary), "--version"], check=True, capture_output=True, text=True).stdout.strip()
    if version != baseline.CODEX_VERSION:
        raise ValueError(f"Codex version drifted: {version!r}")
    if binary.stat().st_size != baseline.CODEX_SIZE_BYTES or _sha256(binary) != baseline.CODEX_SHA256:
        raise ValueError("the pinned Codex 0.145.0 binary identity drifted")
    identity = baseline._isolated_codex_identity(
        python=python,
        miles_root=Path(args.miles_root).resolve(),
        yeto_root=Path(args.yeto_root).resolve(),
    )
    return rows, {
        **{str(key): str(value) for key, value in identity["stock"].items()},
        "YETO_CODEX_OPENENV_BACKEND_PROFILE": "qwen35_08b",
        "YETO_CODEX_OPENENV_MODEL_ID": baseline.MODEL,
        "YETO_CODEX_OPENENV_MODEL_REVISION": baseline.MODEL_REVISION,
        "YETO_CODEX_OPENENV_BASE_INSTRUCTIONS_SHA256": identity["openenv"]["base_instructions_sha256"],
        "YETO_CODEX_OPENENV_TERMINAL_EXEC_TOOL_SCHEMA_SHA256": identity["openenv"]["terminal_exec_tool_schema_sha256"],
        "YETO_CODEX_OPENENV_SUBMIT_TOOL_SCHEMA_SHA256": identity["openenv"]["submit_tool_schema_sha256"],
        "YETO_CODEX_OPENENV_DYNAMIC_TOOLS_SCHEMA_SHA256": identity["openenv"]["dynamic_tools_schema_sha256"],
        "YETO_CODEX_BINARY_PATH": str(binary),
        "YETO_CODEX_BINARY_SIZE_BYTES": str(binary.stat().st_size),
        "YETO_CODEX_BINARY_SHA256": _sha256(binary),
        "YETO_CODEX_VERSION": baseline.CODEX_VERSION,
    }


def execute(args: RampArgs) -> None:
    rows, codex_env = _preflight(args)
    hmac_key_file = baseline._hmac_key_file()
    prompt_path = Path(args.prompt_data).resolve()
    dump_path = Path(args.dump_details).resolve()
    island_seed = args.rollout_seed + args.island_id

    checkpoint_args = f"--hf-checkpoint {_q(args.hf_checkpoint)} --ref-load {_q(args.ref_load)} --model-name qwen3_5 --megatron-to-hf-mode raw --no-load-optim --no-load-rng --finetune "
    rollout_args = (
        f"--prompt-data {_q(prompt_path)} --input-key messages --metadata-key metadata "
        f"--num-rollout 1 --rollout-batch-size {len(rows)} --n-samples-per-prompt 1 "
        "--num-steps-per-rollout 1 --rollout-one-pass-no-replacement "
        f"--global-batch-size {len(rows)} --micro-batch-size 1 "
        "--use-dynamic-global-batch-size --sao-compaction "
        "--rollout-temperature 0.7 --rollout-top-p 0.95 "
        f"--rollout-max-context-len {args.max_seq_len} "
        f"--rollout-max-response-len {args.model_call_max_tokens} "
        f"--max-seq-len {args.max_seq_len} --dump-details {_q(dump_path)} "
    )
    topology_args = "--num-gpus-per-node 1 --actor-num-nodes 1 --actor-num-gpus-per-node 1 --rollout-num-gpus 1 --rollout-num-gpus-per-engine 1 --tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 --context-parallel-size 1 --expert-model-parallel-size 1 --expert-tensor-parallel-size 1 "
    sglang_args = (
        "--sglang-mem-fraction-static 0.90 "
        "--sglang-reasoning-parser qwen3 --sglang-tool-call-parser qwen3_coder "
        f"--sglang-context-length {args.max_seq_len} "
        f"--sglang-server-concurrency {args.concurrency} "
        f"--sglang-max-running-requests {args.concurrency} "
        f"--async-max-concurrent-samples {args.concurrency} "
        "--sglang-disable-cuda-graph --sglang-cuda-graph-backend-prefill disabled "
        "--sglang-chunked-prefill-size 4096 "
    )
    agent_args = (
        "--custom-generate-function-path miles.rollout.generate_hub.agentic_tool_call.generate "
        "--custom-agent-function-path codex_openenv_subprocess_agent_function.run "
        "--custom-rm-path openenv_generate.reward_func "
        "--dynamic-sampling-filter-path openenv_generate.check_terminal_bench_episode "
        "--tito-model qwen35 --use-session-server --session-server-port 30000 "
        "--session-server-startup-timeout-secs 120 "
        "--tito-allowed-append-roles user tool "
    )
    runtime_args = (
        "--debug-rollout-only --max-position-embeddings 262144 "
        f"--seq-length {args.max_seq_len} "
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--attention-backend flash --attention-softmax-in-fp32 --bf16 "
        f"--seed {island_seed} --rollout-seed {island_seed} "
        "--distributed-timeout-minutes 30 --pin-rollout-manager-to-head "
        "--wandb-mode disabled "
        f"--apply-chat-template-kwargs {_q('{"clear_thinking":false}')} "
    )
    extra_env = {
        **codex_env,
        "PYTHONPATH": os.pathsep.join([args.megatron_path, str(baseline.SCRIPT_DIR), args.miles_root, args.yeto_root]),
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "OPENENV_ENV_URL": args.openenv_env_url,
        "OPENENV_NATIVE_EVALUATE": "1",
        "OPENENV_AGENT_PYTHON": str(baseline._validated_openenv_python(args.openenv_agent_python)),
        "OPENENV_MAX_ROLLOUT_TIME_SECONDS": "1800",
        "SECRLENV_MAX_ROLLOUT_TIME_SECONDS": "1800",
        "OPENENV_MESSAGE_TIMEOUT_S": "1800",
        "SECRLENV_MAX_TURNS": str(args.max_turns),
        "YETO_CODEX_CHAT_TEMPLATE": "qwen35_08b",
        "YETO_CODEX_BACKEND_MAX_TOKENS": str(args.model_call_max_tokens),
        "YETO_CODEX_COMPACTION_ENABLED": "1",
        "YETO_CODEX_COMPACTION_TRIGGER_TOKENS": str(args.compaction_trigger_tokens),
        "YETO_CODEX_COMPACTION_SUMMARY_MAX_TOKENS": str(args.compaction_summary_max_tokens),
        "YETO_CODEX_MAX_COMPACTIONS": str(args.max_compactions),
        baseline.HMAC_FILE_ENV: str(hmac_key_file),
        "NCCL_NVLS_ENABLE": "0",
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "RAY_DEDUP_LOGS": "0",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    U.execute_train(
        train_args=(checkpoint_args + rollout_args + topology_args + sglang_args + agent_args + runtime_args),
        config=args,
        num_gpus_per_node=1,
        megatron_model_type="qwen3.5-0.8B",
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env,
    )


@U.dataclass_cli
def main(args: RampArgs) -> None:
    execute(args)


if __name__ == "__main__":
    typer.run(main)
