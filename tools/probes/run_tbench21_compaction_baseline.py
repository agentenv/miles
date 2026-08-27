"""Run one immutable Terminal-Bench 2.1 CompactionRL baseline island.

The full baseline is eight sibling invocations of this entrypoint, one per
physical GPU and one plan shard per invocation.  It is rollout-only: the
resulting native Miles shards are later consumed in full by offline SAO value
pretraining.  Training and evaluation use separate entrypoints.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

import miles.utils.external_utils.command_utils as U


SCRIPT_DIR = Path(__file__).resolve().parents[2] / "examples" / "experimental" / "openenv"
MODEL = "Qwen/Qwen3.5-0.8B"
MODEL_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
CODEX_VERSION = "codex-cli 0.145.0"
CODEX_SHA256 = "a2a05dafaa1acb002a45eaec0a462de5b13694fcfcd7bc43305f14781ce7be14"
CODEX_SIZE_BYTES = 310730800
OPENENV_PYTHON = Path("/root/openenv-venv/bin/python")
OPENENV_PYVENV_CFG_SHA256 = "d9a08e91e4d3c59990a1795a0c3db0deef3f94a6eb5633938d960e6fae2f2eed"
HMAC_FILE_ENV = "TBENCH_REWARD_HMAC_KEY_FILE"
HMAC_DIRECT_ENV = "TBENCH_REWARD_HMAC_KEY"
ROLLOUT_SEED_BASE = 82621


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _validated_openenv_python(value: str | Path) -> Path:
    """Return the exact pinned venv launcher without resolving away the venv."""

    source = Path(value).expanduser()
    if not source.is_absolute() or source != OPENENV_PYTHON:
        raise ValueError("OpenEnv Python must be the pinned venv launcher")
    venv = source.parent.parent
    config = venv / "pyvenv.cfg"
    if not source.is_symlink() or venv.is_symlink() or source.parent.is_symlink():
        raise ValueError("the pinned OpenEnv venv layout drifted")
    try:
        target = source.resolve(strict=True)
        config_stat = config.stat()
    except OSError as error:
        raise ValueError("the pinned OpenEnv venv is incomplete") from error
    if (
        not target.is_file()
        or target.is_symlink()
        or not os.access(target, os.X_OK)
        or config.is_symlink()
        or not config.is_file()
        or config_stat.st_uid != 0
        or config_stat.st_gid != 0
        or stat.S_IMODE(config_stat.st_mode) & 0o022
        or _sha256(config) != OPENENV_PYVENV_CFG_SHA256
    ):
        raise ValueError("the pinned OpenEnv venv identity drifted")
    return source


def _load_rows(path: Path, island_id: int, phase: str) -> list[dict[str, Any]]:
    if phase not in {"baseline", "eval"}:
        raise ValueError("rollout-only phase must be baseline or eval")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON") from error
        metadata = row.get("metadata") if isinstance(row, dict) else None
        messages = row.get("messages") if isinstance(row, dict) else None
        if (
            not isinstance(metadata, dict)
            or metadata.get("island_id") != island_id
            or metadata.get("rollout_seed") != ROLLOUT_SEED_BASE + island_id
            or metadata.get("split") != phase
            or metadata.get("episode_timeout_seconds") != 1800
            or not isinstance(metadata.get("task_id"), str)
            or not isinstance(metadata.get("sample_id"), str)
            or not isinstance(messages, list)
            or not messages
        ):
            raise ValueError(f"{path}:{line_number}: row violates the baseline-island contract")
        rows.append(row)
    sample_ids = [row["metadata"]["sample_id"] for row in rows]
    expected_counts = {"baseline": {44, 45}, "eval": {22, 23}}
    if len(rows) not in expected_counts[phase] or len(set(sample_ids)) != len(sample_ids):
        expected = " or ".join(str(value) for value in sorted(expected_counts[phase]))
        raise ValueError(f"{phase} island must contain {expected} unique planned trajectories")
    return rows


def _hmac_key_file() -> Path:
    if os.getenv(HMAC_DIRECT_ENV):
        raise ValueError("direct Terminal-Bench HMAC keys are forbidden for this launch")
    raw = os.getenv(HMAC_FILE_ENV)
    if not raw:
        raise ValueError(f"{HMAC_FILE_ENV} is required")
    candidate = Path(raw).expanduser()
    if not candidate.is_absolute() or candidate.is_symlink():
        raise ValueError("Terminal-Bench HMAC key file must be absolute and non-symlink")
    path = candidate.resolve()
    try:
        information = path.stat()
        value = path.read_bytes().rstrip(b"\r\n")
    except OSError as error:
        raise ValueError("Terminal-Bench HMAC key file is unreadable") from error
    if (
        not stat.S_ISREG(information.st_mode)
        or stat.S_IMODE(information.st_mode) not in {0o400, 0o600}
        or not 32 <= len(value) <= 4096
    ):
        raise ValueError("Terminal-Bench HMAC key file is not private and bounded")
    return path


def _isolated_codex_identity(*, python: Path, miles_root: Path, yeto_root: Path) -> dict[str, str]:
    code = r"""
import json
from yeto_miles_secrlenv import codex_harness_agent as stock
from codex_openenv_agent_function import codex_openenv_harness_identity
print(json.dumps({"stock": stock._IDENTITY_ENV,
                  "openenv": codex_openenv_harness_identity()}, sort_keys=True))
"""
    env = os.environ.copy()
    env.update(
        {
            "PYTHONDONTWRITEBYTECODE": "1",
            "YETO_CODEX_CHAT_TEMPLATE": "qwen35_08b",
            "PYTHONPATH": os.pathsep.join(
                dict.fromkeys(
                    [
                        str(SCRIPT_DIR),
                        str(miles_root),
                        str(yeto_root),
                        *env.get("PYTHONPATH", "").split(os.pathsep),
                    ]
                )
            ),
        }
    )
    completed = subprocess.run(
        [str(python), "-c", code],
        check=True,
        capture_output=True,
        text=True,
        env=env,
    )
    try:
        payload = json.loads(completed.stdout)
    except json.JSONDecodeError as error:
        raise RuntimeError("isolated Codex identity preflight returned invalid JSON") from error
    if (
        not isinstance(payload, dict)
        or not isinstance(payload.get("stock"), dict)
        or not isinstance(payload.get("openenv"), dict)
    ):
        raise RuntimeError("isolated Codex identity preflight returned an invalid contract")
    return payload


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    island_id: int = 0
    phase: str = "baseline"
    prompt_data: str = "/root/data/tbench21-plan/baseline/island-0.jsonl"
    dump_details: str = "/root/runs/tbench21-baseline/island-0/details"
    hf_checkpoint: str = "/root/models/Qwen3.5-0.8B"
    ref_load: str = "/root/checkpoints/Qwen3.5-0.8B_torch_dist"
    codex_binary: str = "/root/codex-cli/vendor/x86_64-unknown-linux-musl/bin/codex"
    openenv_agent_python: str = "/root/openenv-venv/bin/python"
    openenv_env_url: str = "http://host.docker.internal:8003"
    miles_root: str = "/root/miles"
    yeto_root: str = "/root/yeto"
    megatron_path: str = "/root/Megatron-LM"
    max_seq_len: int = 8192
    model_call_max_tokens: int = 2048
    compaction_trigger_tokens: int = 6144
    compaction_summary_max_tokens: int = 1024
    max_compactions: int = 3
    concurrency: int = 38
    max_turns: int = 40
    rollout_seed: int = ROLLOUT_SEED_BASE
    serve_ref_checkpoint: bool = False


def _validated_eval_checkpoint(value: str | Path) -> Path:
    root = Path(value).resolve()
    marker = root / "latest_checkpointed_iteration.txt"
    if marker.is_symlink() or not marker.is_file():
        raise ValueError("eval ref_load has no safe checkpoint iteration marker")
    try:
        iteration = marker.read_text(encoding="utf-8").strip()
    except (OSError, UnicodeError) as error:
        raise ValueError("eval ref_load checkpoint marker is unreadable") from error
    if iteration != "0":
        raise ValueError(f"eval ref_load checkpoint marker must be rollout 0, got {iteration!r}")
    iteration_dir = root / "iter_0000000"
    metadata = iteration_dir / ".metadata"
    shards = sorted(iteration_dir.glob("*.distcp"))
    if (
        iteration_dir.is_symlink()
        or not iteration_dir.is_dir()
        or metadata.is_symlink()
        or not metadata.is_file()
        or not shards
        or any(path.is_symlink() or not path.is_file() for path in shards)
    ):
        raise ValueError("eval ref_load checkpoint iteration is incomplete")
    return root


def _preflight(args: ScriptArgs) -> tuple[list[dict[str, Any]], dict[str, str]]:
    if not 0 <= args.island_id < 8:
        raise ValueError("island_id must be in [0, 7]")
    if args.rollout_seed != ROLLOUT_SEED_BASE:
        raise ValueError("rollout_seed must equal the immutable plan seed base")
    if args.max_seq_len != 8192:
        raise ValueError("the pinned Qwen3.5-0.8B baseline requires max_seq_len=8192")
    if not (1 <= args.compaction_summary_max_tokens < args.compaction_trigger_tokens < args.max_seq_len):
        raise ValueError("compaction summary/trigger/context budgets are inconsistent")
    if args.compaction_trigger_tokens + args.compaction_summary_max_tokens > args.max_seq_len:
        raise ValueError("compaction trigger does not reserve its summary budget")
    if not 1 <= args.concurrency <= 64:
        raise ValueError("one-island concurrency must be in [1, 64]")
    if args.serve_ref_checkpoint is not (args.phase == "eval"):
        raise ValueError("--serve-ref-checkpoint is required exactly for the eval phase")
    _hmac_key_file()

    prompt_path = Path(args.prompt_data).resolve()
    rows = _load_rows(prompt_path, args.island_id, args.phase)
    output = Path(args.dump_details).resolve()
    if output.exists() or output.is_symlink():
        raise FileExistsError("baseline dump_details must be a fresh path")
    for name, value in {
        "HF checkpoint": args.hf_checkpoint,
        "Megatron checkpoint": args.ref_load,
        "Miles root": args.miles_root,
        "Yeto root": args.yeto_root,
    }.items():
        if not Path(value).resolve().is_dir():
            raise FileNotFoundError(f"{name} is missing: {value}")
    if args.phase == "eval":
        _validated_eval_checkpoint(args.ref_load)
    binary = Path(args.codex_binary).resolve()
    python = _validated_openenv_python(args.openenv_agent_python)
    if not binary.is_file() or binary.is_symlink() or not os.access(binary, os.X_OK):
        raise FileNotFoundError("the pinned Codex binary is missing or not executable")
    version = subprocess.run(
        [str(binary), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if version != CODEX_VERSION:
        raise ValueError(f"Codex version drifted: {version!r}")
    if binary.stat().st_size != CODEX_SIZE_BYTES or _sha256(binary) != CODEX_SHA256:
        raise ValueError("the pinned Codex 0.145.0 binary identity drifted")
    identity = _isolated_codex_identity(
        python=python,
        miles_root=Path(args.miles_root).resolve(),
        yeto_root=Path(args.yeto_root).resolve(),
    )
    return rows, {
        **{str(key): str(value) for key, value in identity["stock"].items()},
        "YETO_CODEX_OPENENV_BACKEND_PROFILE": "qwen35_08b",
        "YETO_CODEX_OPENENV_MODEL_ID": MODEL,
        "YETO_CODEX_OPENENV_MODEL_REVISION": MODEL_REVISION,
        "YETO_CODEX_OPENENV_BASE_INSTRUCTIONS_SHA256": identity["openenv"]["base_instructions_sha256"],
        "YETO_CODEX_OPENENV_TERMINAL_EXEC_TOOL_SCHEMA_SHA256": identity["openenv"]["terminal_exec_tool_schema_sha256"],
        "YETO_CODEX_OPENENV_SUBMIT_TOOL_SCHEMA_SHA256": identity["openenv"]["submit_tool_schema_sha256"],
        "YETO_CODEX_OPENENV_DYNAMIC_TOOLS_SCHEMA_SHA256": identity["openenv"]["dynamic_tools_schema_sha256"],
        "YETO_CODEX_BINARY_PATH": str(binary),
        "YETO_CODEX_BINARY_SIZE_BYTES": str(binary.stat().st_size),
        "YETO_CODEX_BINARY_SHA256": _sha256(binary),
        "YETO_CODEX_VERSION": CODEX_VERSION,
    }


def execute(args: ScriptArgs) -> None:
    rows, codex_env = _preflight(args)
    hmac_key_file = _hmac_key_file()
    prompt_path = Path(args.prompt_data).resolve()
    dump_path = Path(args.dump_details).resolve()
    island_seed = args.rollout_seed + args.island_id

    checkpoint_args = (
        f"--train-backend megatron --hf-checkpoint {_q(args.hf_checkpoint)} "
        f"--ref-load {_q(args.ref_load)} --model-name qwen3_5 "
        "--megatron-to-hf-mode raw --dist-ckpt-strictness raise_unexpected "
        "--no-load-optim --no-load-rng --finetune "
    )
    rollout_args = (
        f"--prompt-data {_q(prompt_path)} --input-key messages --metadata-key metadata "
        f"--start-rollout-id 0 --num-rollout 1 --rollout-batch-size {len(rows)} --n-samples-per-prompt 1 "
        "--num-steps-per-rollout 1 --rollout-one-pass-no-replacement "
        f"--global-batch-size {len(rows)} --micro-batch-size 1 "
        "--use-dynamic-global-batch-size --sao-compaction "
        "--rollout-temperature 0.7 --rollout-top-p 0.95 "
        f"--rollout-max-context-len {args.max_seq_len} "
        f"--rollout-max-response-len {args.model_call_max_tokens} "
        f"--max-seq-len {args.max_seq_len} --dump-details {_q(dump_path)} "
    )
    topology_args = (
        ("--colocate " if args.serve_ref_checkpoint else "")
        + "--num-gpus-per-node 1 --actor-num-nodes 1 --actor-num-gpus-per-node 1 "
        "--rollout-num-gpus 1 --rollout-num-gpus-per-engine 1 "
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 --expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
    )
    eval_sglang_args = (
        "--sglang-mem-fraction-static 0.15 " "--sglang-max-total-tokens 393216 " "--sglang-max-mamba-cache-size 256 "
    )
    sglang_args = (
        (eval_sglang_args if args.serve_ref_checkpoint else "--sglang-mem-fraction-static 0.90 ")
        + "--sglang-reasoning-parser qwen3 --sglang-tool-call-parser qwen3_coder "
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
    checkpoint_rollout_args = (
        "--rollout-only-from-checkpoint --debug-disable-optimizer "
        f"--rollout-only-publication-evidence {_q(dump_path.parent / 'checkpoint-publication.json')} "
        "--rollout-weight-version-format counter "
        "--update-weight-buffer-size 1073741824 "
    )
    chat_template_kwargs = _q('{"clear_thinking":false}')
    runtime_args = (
        (checkpoint_rollout_args if args.serve_ref_checkpoint else "--debug-rollout-only ")
        + "--max-position-embeddings 262144 "
        f"--seq-length {args.max_seq_len} "
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--attention-backend flash --attention-softmax-in-fp32 --bf16 "
        f"--seed {island_seed} --rollout-seed {island_seed} "
        "--distributed-timeout-minutes 30 --pin-rollout-manager-to-head "
        "--wandb-mode disabled "
        f"--apply-chat-template-kwargs {chat_template_kwargs} "
    )
    extra_env = {
        **codex_env,
        "PYTHONPATH": os.pathsep.join([args.megatron_path, str(SCRIPT_DIR), args.miles_root, args.yeto_root]),
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "OPENENV_ENV_URL": args.openenv_env_url,
        "OPENENV_NATIVE_EVALUATE": "1",
        "OPENENV_AGENT_PYTHON": str(_validated_openenv_python(args.openenv_agent_python)),
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
        HMAC_FILE_ENV: str(hmac_key_file),
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
def main(args: ScriptArgs) -> None:
    execute(args)


if __name__ == "__main__":
    typer.run(main)
