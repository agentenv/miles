"""Run one Qwen3.5-0.8B Terminal-Bench SAO streaming-DiLoCo island.

The production online wave consists of eight sibling containers, each with
exactly one visible H200 and one invocation of this launcher.  Every process
owns 22 immutable Terminal-Bench 2.1 training trajectories and colocates its
full-parameter actor, full-parameter critic, and TP1 SGLang worker on that one
GPU.  Actor and critic updates are sent to separate streaming-DiLoCo sessions
through the hash-bound benchmark-neutral v2 runtime contracts.

This launcher deliberately does not start either syncer or construct runtime
contracts.  Those are shared, separately attested services/inputs and must be
ready before any island is allowed to allocate a GPU.
"""

from __future__ import annotations

import json
import os
import re
import shlex
import stat
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import typer

import miles.utils.external_utils.command_utils as U
from tools.probes.run_tbench21_compaction_baseline import (
    CODEX_VERSION,
    MODEL,
    MODEL_REVISION,
    SCRIPT_DIR,
    _isolated_codex_identity,
    _sha256,
    _validated_openenv_python,
)


_HASH = re.compile(r"[0-9a-f]{64}\Z")
_CONTEXT_SCHEMA = "miles.sao-runtime.v2"
_STREAMING_SCHEMA = "yeto.sao-streaming-runtime.v2"
_BENCHMARK = "terminal-bench-2.1"
_CODEX_AGENT = "codex_openenv_subprocess_agent_function.run"
_GENERATE = "miles.rollout.generate_hub.agentic_tool_call.generate"
_REWARD = "openenv_generate.reward_func"
_FILTER = "openenv_generate.check_terminal_bench_episode"
_ROLLOUT = "yeto.rl.miles.generate_rollout"
_QUEUE = "yeto.rl.miles.queue_completed_groups"
_EXPECTED_ROWS = 22
_EXPECTED_ISLANDS = 8
_EPISODE_TIMEOUT_SECONDS = 1800
_MAX_SEQ_LEN = 8192
_MODEL_CALL_MAX_TOKENS = 2048
_COMPACTION_TRIGGER_TOKENS = 6144
_COMPACTION_SUMMARY_MAX_TOKENS = 1024
_MAX_COMPACTIONS = 3
_PER_ISLAND_CONCURRENCY = 38
_SGLANG_MEM_FRACTION_STATIC = 0.15
_SGLANG_MAX_TOTAL_TOKENS = 393216
_SGLANG_MAX_MAMBA_CACHE_SIZE = 256
_ROLLOUT_SEED_BASE = 82621


class LaunchContractError(RuntimeError):
    """The local island differs from its immutable online-run contract."""


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


def _island_path(template: str, island_id: int) -> Path:
    try:
        rendered = template.format(island_id=island_id)
    except (IndexError, KeyError, ValueError) as error:
        raise LaunchContractError("invalid {island_id} path template") from error
    path = Path(rendered).expanduser()
    if not path.is_absolute() or ".." in path.parts:
        raise LaunchContractError("island paths must be absolute and traversal-free")
    if path.is_symlink():
        raise LaunchContractError("island paths must not be symlinks")
    return path.resolve()


def _exact_hash(name: str, value: str) -> str:
    if _HASH.fullmatch(value) is None:
        raise LaunchContractError(f"{name} must be a lowercase SHA256")
    return value


def _load_private_json(path: Path, expected_sha256: str, *, name: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise LaunchContractError(f"{name} must be a regular non-symlink file")
    if stat.S_IMODE(path.stat().st_mode) not in {0o400, 0o600}:
        raise LaunchContractError(f"{name} must have mode 0400 or 0600")
    if _sha256(path) != _exact_hash(f"{name} SHA256", expected_sha256):
        raise LaunchContractError(f"{name} SHA256 mismatch")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise LaunchContractError(f"{name} is unreadable or malformed") from error
    if not isinstance(value, dict):
        raise LaunchContractError(f"{name} must contain a JSON object")
    return value


def _load_train_rows(path: Path, island_id: int) -> list[dict[str, Any]]:
    if path.is_symlink() or not path.is_file():
        raise LaunchContractError("training shard must be a regular non-symlink file")
    rows: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as error:
            raise LaunchContractError(
                f"{path}:{line_number}: invalid JSON"
            ) from error
        metadata = row.get("metadata") if isinstance(row, dict) else None
        messages = row.get("messages") if isinstance(row, dict) else None
        if not isinstance(metadata, dict) or not isinstance(messages, list) or not messages:
            raise LaunchContractError(
                f"{path}:{line_number}: invalid messages/metadata"
            )
        task_id = metadata.get("task_id")
        replica = metadata.get("rollout_replica")
        sample_id = metadata.get("sample_id")
        expected_sample_id = (
            f"train:{task_id}:r{replica}"
            if isinstance(task_id, str) and type(replica) is int
            else None
        )
        if (
            not isinstance(task_id, str)
            or not task_id
            or type(replica) is not int
            or replica not in range(4)
            or not isinstance(sample_id, str)
            or sample_id != expected_sample_id
            or type(metadata.get("island_id")) is not int
            or metadata.get("island_id") != island_id
            or metadata.get("rollout_seed") != _ROLLOUT_SEED_BASE + island_id
            or metadata.get("split") != "train"
            or metadata.get("prompt_tier") != "l2"
            or metadata.get("max_seq_len") != _MAX_SEQ_LEN
            or metadata.get("episode_timeout_seconds")
            != _EPISODE_TIMEOUT_SECONDS
        ):
            raise LaunchContractError(
                f"{path}:{line_number}: row violates the online-island contract"
            )
        rows.append(row)
    sample_ids = [row["metadata"]["sample_id"] for row in rows]
    task_ids = [row["metadata"]["task_id"] for row in rows]
    if (
        len(rows) != _EXPECTED_ROWS
        or len(set(sample_ids)) != _EXPECTED_ROWS
        or len(set(task_ids)) != _EXPECTED_ROWS
    ):
        raise LaunchContractError(
            "one online island requires exactly 22 unique planned train rows"
        )
    return rows


def _validate_hmac_key_file(path: Path) -> None:
    if os.getenv("TBENCH_REWARD_HMAC_KEY"):
        raise LaunchContractError("direct Terminal-Bench HMAC keys are forbidden")
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise LaunchContractError(
            "Terminal-Bench HMAC key must be an absolute regular non-symlink file"
        )
    if stat.S_IMODE(path.stat().st_mode) not in {0o400, 0o600}:
        raise LaunchContractError("Terminal-Bench HMAC key must have mode 0400 or 0600")
    size = len(path.read_bytes().rstrip(b"\r\n"))
    if not 32 <= size <= 4096:
        raise LaunchContractError(
            "Terminal-Bench HMAC key must contain 32..4096 private bytes"
        )


def _fresh_output(path: Path, *, name: str, run_root: Path) -> None:
    if path.exists() or path.is_symlink():
        raise LaunchContractError(f"{name} must be a fresh path")
    try:
        path.relative_to(run_root)
    except ValueError as error:
        raise LaunchContractError(f"{name} must stay under the island run root") from error
    parent = path.parent
    if parent.is_symlink():
        raise LaunchContractError(f"{name} parent must not be a symlink")


def _validate_one_h200() -> None:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise LaunchContractError("cannot attest the island's physical GPU") from error
    rows = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise LaunchContractError(
            "one island container must expose exactly one physical GPU"
        )
    try:
        name, memory_raw = (part.strip() for part in rows[0].rsplit(",", 1))
        memory_mib = int(memory_raw)
    except (TypeError, ValueError) as error:
        raise LaunchContractError("nvidia-smi returned malformed GPU evidence") from error
    if "H200" not in name.upper() or memory_mib < 120 * 1024:
        raise LaunchContractError(
            "one-GPU SAO island requires an H200 with at least 120 GiB"
        )


def _codex_env(args: ScriptArgs) -> dict[str, str]:
    binary_source = Path(args.codex_binary).expanduser()
    if binary_source.is_symlink():
        raise LaunchContractError("the Codex executable must not be a symlink")
    binary = binary_source.resolve()
    try:
        python = _validated_openenv_python(args.openenv_agent_python)
    except ValueError as error:
        raise LaunchContractError(str(error)) from error
    if not binary.is_file() or binary.is_symlink() or not os.access(binary, os.X_OK):
        raise LaunchContractError("the pinned stock Codex binary is unavailable")
    version = subprocess.run(
        [str(binary), "--version"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if version != CODEX_VERSION:
        raise LaunchContractError(f"stock Codex version drifted: {version!r}")
    identity = _isolated_codex_identity(
        python=python,
        miles_root=Path(args.miles_root).resolve(),
        yeto_root=Path(args.yeto_root).resolve(),
    )
    environment = {
        **{str(key): str(value) for key, value in identity["stock"].items()},
        "YETO_CODEX_OPENENV_BACKEND_PROFILE": "qwen35_08b",
        "YETO_CODEX_OPENENV_MODEL_ID": MODEL,
        "YETO_CODEX_OPENENV_MODEL_REVISION": MODEL_REVISION,
        "YETO_CODEX_OPENENV_BASE_INSTRUCTIONS_SHA256": identity["openenv"][
            "base_instructions_sha256"
        ],
        "YETO_CODEX_OPENENV_TERMINAL_EXEC_TOOL_SCHEMA_SHA256": identity["openenv"][
            "terminal_exec_tool_schema_sha256"
        ],
        "YETO_CODEX_OPENENV_SUBMIT_TOOL_SCHEMA_SHA256": identity["openenv"][
            "submit_tool_schema_sha256"
        ],
        "YETO_CODEX_OPENENV_DYNAMIC_TOOLS_SCHEMA_SHA256": identity["openenv"][
            "dynamic_tools_schema_sha256"
        ],
        "YETO_CODEX_BINARY_PATH": str(binary),
        "YETO_CODEX_BINARY_SIZE_BYTES": str(binary.stat().st_size),
        "YETO_CODEX_BINARY_SHA256": _sha256(binary),
        "YETO_CODEX_VERSION": CODEX_VERSION,
        "YETO_CODEX_BACKEND_MAX_TOKENS": str(_MODEL_CALL_MAX_TOKENS),
    }
    if (
        environment.get("YETO_CODEX_REASONING_EFFORT") != "xhigh"
        or environment.get("YETO_CODEX_BACKEND_REASONING_EFFORT") != "xhigh"
        or environment.get("YETO_CODEX_BACKEND_THINKING") != "enabled"
        or environment.get("YETO_CODEX_CHAT_TEMPLATE") != "qwen35_08b"
    ):
        raise LaunchContractError("stock Codex xhigh/thinking profile drifted")
    return environment


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    island_id: int = 0
    prompt_data: str = "/root/data/tbench21-plan/train/island-{island_id}.jsonl"
    sao_context: str = "/root/data/tbench21-contracts/island-{island_id}/sao-context.json"
    sao_context_sha256: str = os.environ.get("YETO_SAO_CONTEXT_SHA256", "")
    streaming_context: str = (
        "/root/data/tbench21-contracts/island-{island_id}/sao-streaming-context.json"
    )
    streaming_context_sha256: str = os.environ.get(
        "YETO_SAO_STREAMING_CONTEXT_SHA256", ""
    )
    run_root: str = "/root/runs/tbench21-sao-online/island-{island_id}"
    hf_checkpoint: str = "/root/models/Qwen3.5-0.8B"
    ref_load: str = "/root/checkpoints/Qwen3.5-0.8B_torch_dist"
    critic_load: str = "/root/checkpoints/Qwen3.5-0.8B_value_tbench21_full"
    critic_contract_sha256: str = os.environ.get(
        "YETO_CRITIC_CONTRACT_SHA256", ""
    )
    codex_binary: str = "/root/codex-cli/vendor/x86_64-unknown-linux-musl/bin/codex"
    openenv_agent_python: str = "/root/openenv-venv/bin/python"
    openenv_env_url: str = "http://host.docker.internal:8003"
    tbench_hmac_key_file: str = "/run/secrets/tbench-reward-hmac.key"
    miles_root: str = "/root/miles"
    yeto_root: str = "/root/yeto"
    megatron_path: str = "/root/Megatron-LM"
    rollout_seed: int = _ROLLOUT_SEED_BASE


@dataclass(frozen=True)
class _Prepared:
    rows: tuple[dict[str, Any], ...]
    prompt_path: Path
    sao_context_path: Path
    streaming_context_path: Path
    run_root: Path
    dump_details: Path
    actor_save: Path
    critic_save: Path
    openenv_python: Path
    codex_env: dict[str, str]


def _preflight(args: ScriptArgs) -> _Prepared:
    if args.island_id not in range(_EXPECTED_ISLANDS):
        raise LaunchContractError("island_id must be in [0, 7]")
    if args.rollout_seed != _ROLLOUT_SEED_BASE:
        raise LaunchContractError("rollout_seed must equal the immutable plan seed base")
    if args.num_nodes != 1:
        raise LaunchContractError("one online island requires num_nodes=1")
    if args.extra_env_vars.strip() not in {"", "{}"}:
        raise LaunchContractError(
            "extra environment overrides are forbidden for the attested online run"
        )
    if os.getenv("RAY_ADDRESS") or os.getenv(
        "MILES_SCRIPT_EXTERNAL_RAY", "0"
    ).lower() in {"1", "true"}:
        raise LaunchContractError("one online island requires its own local Ray cluster")
    if os.getenv("MILES_SCRIPT_ENABLE_RAY_SUBMIT", "1").lower() not in {
        "1",
        "true",
    }:
        raise LaunchContractError("Ray job submission must remain enabled")
    _validate_one_h200()
    for name, value in {
        "HF checkpoint": args.hf_checkpoint,
        "actor checkpoint": args.ref_load,
        "critic checkpoint": args.critic_load,
        "Miles root": args.miles_root,
        "Yeto root": args.yeto_root,
        "Megatron root": args.megatron_path,
    }.items():
        source = Path(value).expanduser()
        path = source.resolve()
        if source.is_symlink() or not path.is_dir():
            raise LaunchContractError(f"{name} is missing or unsafe: {value}")
    for checkpoint, name in (
        (Path(args.ref_load).resolve(), "actor"),
        (Path(args.critic_load).resolve(), "critic"),
    ):
        if not (checkpoint / "latest_checkpointed_iteration.txt").is_file():
            raise LaunchContractError(f"{name} checkpoint has no iteration marker")

    critic_contract = Path(args.critic_load).resolve() / "value_pretrain_contract.json"
    if (
        critic_contract.is_symlink()
        or not critic_contract.is_file()
        or _sha256(critic_contract)
        != _exact_hash("critic value-pretraining contract SHA256", args.critic_contract_sha256)
    ):
        raise LaunchContractError("critic value-pretraining contract SHA256 mismatch")

    prompt_path = _island_path(args.prompt_data, args.island_id)
    rows = _load_train_rows(prompt_path, args.island_id)
    sao_path = _island_path(args.sao_context, args.island_id)
    streaming_path = _island_path(args.streaming_context, args.island_id)
    sao = _load_private_json(
        sao_path,
        args.sao_context_sha256,
        name="generic SAO v2 context",
    )
    streaming = _load_private_json(
        streaming_path,
        args.streaming_context_sha256,
        name="SAO streaming v2 context",
    )
    prompt_digest = _sha256(prompt_path)
    reward_digest = _sha256(SCRIPT_DIR / "openenv_generate.py")
    expected_context = {
        "schema": _CONTEXT_SCHEMA,
        "benchmark": _BENCHMARK,
        "model": MODEL,
        "data_sha256": prompt_digest,
        "base_model_revision": MODEL_REVISION,
        "rollout_model_revision": MODEL_REVISION,
        "reward_sha256": reward_digest,
        "dynamic_sampling_max_replacements": 0,
        "learner_id": args.island_id,
    }
    mismatched_context = [
        name for name, expected in expected_context.items() if sao.get(name) != expected
    ]
    try:
        context_data = Path(sao.get("data", "")).resolve()
    except (OSError, TypeError, ValueError):
        context_data = Path("/")
    if context_data != prompt_path:
        mismatched_context.append("data")
    if mismatched_context:
        raise LaunchContractError(
            "generic SAO v2 context drifted: " + ", ".join(mismatched_context)
        )

    evidence = streaming.get("trajectory_evidence")
    actor = streaming.get("actor")
    critic = streaming.get("critic")
    actor_component = actor.get("component") if isinstance(actor, dict) else None
    critic_component = critic.get("component") if isinstance(critic, dict) else None
    actor_syncer = actor.get("syncer") if isinstance(actor, dict) else None
    critic_syncer = critic.get("syncer") if isinstance(critic, dict) else None
    profile = streaming.get("syncer_profile")
    actor_fragments = (
        actor.get("expected_fragments") if isinstance(actor, dict) else None
    )
    critic_fragments = (
        critic.get("expected_fragments") if isinstance(critic, dict) else None
    )
    if (
        streaming.get("schema") != _STREAMING_SCHEMA
        or streaming.get("sao_context_sha256") != args.sao_context_sha256
        or not isinstance(evidence, dict)
        or evidence.get("kind") != _BENCHMARK
        or evidence.get("schema_version") != 2
        or not isinstance(actor, dict)
        or not isinstance(critic, dict)
        or actor.get("role") != "actor"
        or critic.get("role") != "critic"
        or not isinstance(actor_component, dict)
        or not isinstance(critic_component, dict)
        or actor_component.get("model_revision") != MODEL_REVISION
        or critic_component.get("model_revision") != MODEL_REVISION
        or not isinstance(actor_syncer, dict)
        or not isinstance(critic_syncer, dict)
        or actor_syncer.get("host") != critic_syncer.get("host")
        or actor_syncer.get("port") != 29400
        or critic_syncer.get("port") != 29401
        or actor.get("learner_id") != args.island_id
        or critic.get("learner_id") != args.island_id
        or actor.get("learner_generation") != 0
        or critic.get("learner_generation") != 0
        or actor.get("learner_generations") != [0] * _EXPECTED_ISLANDS
        or critic.get("learner_generations") != [0] * _EXPECTED_ISLANDS
        or actor.get("local_horizon") != 1
        or critic.get("local_horizon") != 1
        or actor.get("optimizer_steps_per_round") != 1
        or critic.get("optimizer_steps_per_round") != 2
        or type(actor_fragments) is not int
        or not 1 <= actor_fragments <= 2
        or critic_fragments != actor_fragments
        or actor.get("total_fragment_steps") != actor_fragments
        or critic.get("total_fragment_steps") != actor_fragments
        or not isinstance(profile, dict)
        or profile.get("learners") != _EXPECTED_ISLANDS
        or profile.get("quorum") != _EXPECTED_ISLANDS
        or profile.get("pipeline") != 2
        or profile.get("total_steps") != actor_fragments
        or actor.get("parameter_layout_sha256") != sao.get("layout_hash")
        or actor.get("parameter_layout_sha256")
        == critic.get("parameter_layout_sha256")
        or actor.get("training_contract_sha256")
        != critic.get("training_contract_sha256")
    ):
        raise LaunchContractError("SAO streaming v2 context drifted")

    run_root = _island_path(args.run_root, args.island_id)
    if run_root.exists() and (run_root.is_symlink() or not run_root.is_dir()):
        raise LaunchContractError("island run root exists but is not a real directory")
    dump_details = run_root / "details"
    actor_save = run_root / "actor-checkpoint"
    critic_save = run_root / "critic-checkpoint"
    for output, name in (
        (dump_details, "dump-details output"),
        (actor_save, "actor full-parameter save"),
        (critic_save, "critic full-parameter save"),
    ):
        _fresh_output(output, name=name, run_root=run_root)
    for field in ("completed_groups_path", "event_tape"):
        raw_output = sao.get(field)
        if not isinstance(raw_output, str):
            raise LaunchContractError(f"generic SAO context has no {field}")
        output = Path(raw_output).expanduser()
        if not output.is_absolute() or output.is_symlink():
            raise LaunchContractError(f"generic SAO {field} is unsafe")
        _fresh_output(
            output.resolve(),
            name=f"generic SAO {field}",
            run_root=run_root,
        )
    evidence_directory = evidence.get("directory")
    if not isinstance(evidence_directory, str):
        raise LaunchContractError("streaming evidence directory is missing")
    try:
        evidence_path = Path(evidence_directory).resolve()
        evidence_path.relative_to(run_root)
    except (OSError, TypeError, ValueError) as error:
        raise LaunchContractError(
            "streaming evidence directory must stay under this island run root"
        ) from error
    if evidence_path.exists() or evidence_path.is_symlink():
        raise LaunchContractError("streaming trajectory-evidence output must be fresh")

    hmac_path = Path(args.tbench_hmac_key_file).expanduser()
    _validate_hmac_key_file(hmac_path)
    return _Prepared(
        rows=tuple(rows),
        prompt_path=prompt_path,
        sao_context_path=sao_path,
        streaming_context_path=streaming_path,
        run_root=run_root,
        dump_details=dump_details,
        actor_save=actor_save,
        critic_save=critic_save,
        openenv_python=_validated_openenv_python(args.openenv_agent_python),
        codex_env=_codex_env(args),
    )


def execute(args: ScriptArgs) -> None:
    prepared = _preflight(args)
    base_port = 32000 + args.island_id * 100
    session_port = 31000 + args.island_id
    island_seed = args.rollout_seed + args.island_id

    checkpoint_args = (
        "--train-backend megatron "
        f"--hf-checkpoint {_q(args.hf_checkpoint)} "
        f"--ref-load {_q(args.ref_load)} "
        f"--critic-load {_q(args.critic_load)} "
        f"--critic-value-pretrain-contract-sha256 {_q(args.critic_contract_sha256)} "
        f"--save {_q(prepared.actor_save)} "
        f"--critic-save {_q(prepared.critic_save)} "
        "--save-interval 1 --model-name qwen3_5 --megatron-to-hf-mode raw "
        "--dist-ckpt-strictness raise_unexpected --no-load-optim --no-load-rng "
        "--no-save-optim --no-save-rng --finetune "
    )
    rollout_args = (
        f"--prompt-data {_q(prepared.prompt_path)} "
        "--input-key messages --metadata-key metadata "
        "--start-rollout-id 0 --num-rollout 1 --num-critic-only-steps 0 "
        "--rollout-batch-size 22 --n-samples-per-prompt 1 "
        "--over-sampling-batch-size 22 --num-steps-per-rollout 1 "
        "--num-critic-epochs 2 --global-batch-size 22 --micro-batch-size 1 "
        "--use-dynamic-global-batch-size --sao-online-recipe coding --sao-compaction "
        "--rollout-temperature 0.7 --rollout-top-p 0.95 "
        f"--rollout-max-context-len {_MAX_SEQ_LEN} "
        f"--rollout-max-response-len {_MODEL_CALL_MAX_TOKENS} "
        f"--max-seq-len {_MAX_SEQ_LEN} --dump-details {_q(prepared.dump_details)} "
    )
    topology_args = (
        "--sao-one-gpu-island --colocate --offload --num-gpus-per-node 1 "
        "--actor-num-nodes 1 --actor-num-gpus-per-node 1 "
        "--critic-num-nodes 1 --critic-num-gpus-per-node 1 "
        "--rollout-num-gpus 1 --rollout-num-gpus-per-engine 1 "
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 --expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
    )
    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.0 "
        "--adam-beta1 0.9 --adam-beta2 0.98 --recompute-granularity full "
        "--recompute-method uniform --recompute-num-layers 1 "
    )
    sglang_args = (
        f"--sglang-mem-fraction-static {_SGLANG_MEM_FRACTION_STATIC} "
        f"--sglang-max-total-tokens {_SGLANG_MAX_TOTAL_TOKENS} "
        f"--sglang-max-mamba-cache-size {_SGLANG_MAX_MAMBA_CACHE_SIZE} "
        "--sglang-reasoning-parser qwen3 --sglang-tool-call-parser qwen3_coder "
        f"--sglang-context-length {_MAX_SEQ_LEN} "
        f"--sglang-server-concurrency {_PER_ISLAND_CONCURRENCY} "
        f"--sglang-max-running-requests {_PER_ISLAND_CONCURRENCY} "
        f"--async-max-concurrent-samples {_PER_ISLAND_CONCURRENCY} "
        "--sglang-disable-cuda-graph --sglang-cuda-graph-backend-prefill disabled "
        "--sglang-chunked-prefill-size 4096 "
        f"--rollout-engine-base-port {base_port} --sglang-router-port {base_port + 20} "
        f"--sglang-router-prometheus-port {base_port + 21} "
        f"--train-master-base-port {base_port + 40} "
    )
    agent_args = (
        f"--rollout-function-path {_ROLLOUT} "
        f"--rollout-all-samples-process-path {_QUEUE} "
        f"--custom-generate-function-path {_GENERATE} "
        f"--custom-agent-function-path {_CODEX_AGENT} "
        f"--custom-rm-path {_REWARD} --dynamic-sampling-filter-path {_FILTER} "
        f"--use-session-server --session-server-ip 127.0.0.1 --session-server-port {session_port} "
        "--tito-model qwen35 --tito-allowed-append-roles tool user "
    )
    runtime_args = (
        "--max-position-embeddings 262144 "
        f"--seq-length {_MAX_SEQ_LEN} "
        "--attention-dropout 0.0 --hidden-dropout 0.0 --attention-backend flash "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 --bf16 "
        f"--seed {island_seed} --rollout-seed {island_seed} "
        "--distributed-timeout-minutes 30 --pin-rollout-manager-to-head "
        "--update-weight-buffer-size 1073741824 --update-weight-transfer-mode broadcast "
        "--rollout-weight-version-format counter "
        "--wandb-mode disabled "
        f"--apply-chat-template-kwargs {_q('{\"clear_thinking\":false}')} "
    )
    context_args = (
        f"--sao-secrlenv-context {_q(prepared.sao_context_path)} "
        f"--sao-secrlenv-context-sha256 {_q(args.sao_context_sha256)} "
        f"--sao-streaming-context {_q(prepared.streaming_context_path)} "
        f"--sao-streaming-context-sha256 {_q(args.streaming_context_sha256)} "
    )
    extra_env = {
        **prepared.codex_env,
        "PYTHONPATH": os.pathsep.join(
            [args.megatron_path, str(SCRIPT_DIR), args.miles_root, args.yeto_root]
        ),
        "MILES_EXPERIMENTAL_FT_TRAINER": "0",
        "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR": "1",
        "OPENENV_ENV_URL": args.openenv_env_url,
        "OPENENV_NATIVE_EVALUATE": "1",
        "OPENENV_AGENT_PYTHON": str(prepared.openenv_python),
        "OPENENV_MAX_ROLLOUT_TIME_SECONDS": str(_EPISODE_TIMEOUT_SECONDS),
        "SECRLENV_MAX_ROLLOUT_TIME_SECONDS": str(_EPISODE_TIMEOUT_SECONDS),
        "OPENENV_MESSAGE_TIMEOUT_S": str(_EPISODE_TIMEOUT_SECONDS),
        "OPENENV_MAX_TURNS": "40",
        "SECRLENV_MAX_TURNS": "40",
        "YETO_CODEX_COMPACTION_ENABLED": "1",
        "YETO_CODEX_COMPACTION_TRIGGER_TOKENS": str(
            _COMPACTION_TRIGGER_TOKENS
        ),
        "YETO_CODEX_COMPACTION_SUMMARY_MAX_TOKENS": str(
            _COMPACTION_SUMMARY_MAX_TOKENS
        ),
        "YETO_CODEX_MAX_COMPACTIONS": str(_MAX_COMPACTIONS),
        "TBENCH_REWARD_HMAC_KEY_FILE": str(
            Path(args.tbench_hmac_key_file).resolve()
        ),
        "NCCL_NVLS_ENABLE": "0",
        # Colocated rollout offload enables SGLang's TorchMemorySaver.  That
        # allocator cannot run with expandable segments, so this path must use
        # PyTorch's default CUDA allocator.
        "PYTHONDONTWRITEBYTECODE": "1",
        "PYTHONUNBUFFERED": "1",
        "RAY_DEDUP_LOGS": "0",
        "HF_HUB_OFFLINE": "1",
        "TRANSFORMERS_OFFLINE": "1",
        "HF_HUB_DISABLE_TELEMETRY": "1",
    }
    U.execute_train(
        train_args=(
            context_args
            + checkpoint_args
            + rollout_args
            + topology_args
            + optimizer_args
            + sglang_args
            + agent_args
            + runtime_args
        ),
        config=args,
        num_gpus_per_node=1,
        megatron_model_type="qwen3.5-0.8B",
        megatron_path=args.megatron_path,
        train_script=str(
            Path(args.miles_root).resolve()
            / "tools"
            / "probes"
            / "train_sao_streaming_secrlenv.py"
        ),
        extra_env_vars=extra_env,
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    execute(args)


if __name__ == "__main__":
    typer.run(main)
