"""Build a train-only Miles value dataset from the Qwen3.8-27B TB2.1 corpus.

The input is the legacy Harbor layout under ``/data/sft/tb_traces``: one
ATIF-v1.7 trajectory, one Harbor result, and one native Codex session per
Terminal-Bench task.  The converter is intentionally corpus-specific and
fails closed on any identity, coverage, reward, or tool-binding drift.

The two known zero-reward traces whose ATIF export contains no visible agent
step are recovered from their source-faithful native Codex reasoning item.
No assistant text or reward is synthesized.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from collections import Counter
from pathlib import Path
from typing import Any


MANIFEST_SCHEMA = "miles.value-pretrain.v1"
PROVENANCE_SCHEMA = "miles.tbench21-qwen38-teacher-value-provenance.v1"
REPORT_SCHEMA = "miles.tbench21-qwen38-teacher-value-conversion.v1"

TERMINAL_BENCH_VERSION = "2.1"
TERMINAL_BENCH_TASK_DIR = "terminal-bench-2-1"
EXPECTED_TASKS = 89
EXPECTED_REWARDS = {"0": 54, "1": 35}
EXPECTED_PLAN_SCHEMA = "yeto.tbench21-sao-diloco-plan.v1"

SOURCE_MODEL = "Qwen/Qwen3.8-27B"
SOURCE_AGENT = "codex"
SOURCE_AGENT_VERSION = "0.146.0"
SOURCE_TRAJECTORY_SCHEMA = "ATIF-v1.7"
SOURCE_REASONING_EFFORT = "xhigh"

TARGET_MODEL = "Qwen/Qwen3.5-0.8B"
TARGET_REVISION = "2fc06364715b967f1860aea9cf38778875588b17"
# Qwen3.5 pads the model embedding table beyond the tokenizer vocabulary.
# Keep the three identities separate so a valid tokenizer is not rejected as
# incomplete and emitted IDs are still proven to come from that tokenizer.
TARGET_MODEL_VOCAB_SIZE = 248320
TARGET_TOKENIZER_BASE_VOCAB_SIZE = 248044
TARGET_TOKENIZER_LENGTH = 248077
MAX_SEQ_LEN = 8192

MISSING_REWARD_REJECT = "reject"
MISSING_REWARD_TIMEOUT_ZERO = "map-known-agent-timeout-to-zero"
KNOWN_TIMEOUT_TASK = "terminal-bench/torch-pipeline-parallelism"
KNOWN_TIMEOUT_TYPE = "AgentTimeoutError"
KNOWN_TIMEOUT_MESSAGE = "Agent execution timed out after 900.0 seconds"

NATIVE_FALLBACK_TASKS = (
    "terminal-bench/gpt2-codegolf",
    "terminal-bench/polyglot-c-py",
)


class ConversionError(RuntimeError):
    pass


def _canonical(value: object) -> bytes:
    return (json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True) + "\n").encode("utf-8")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _loads(payload: str, *, location: str) -> Any:
    try:
        return json.loads(
            payload,
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except (TypeError, ValueError, json.JSONDecodeError) as error:
        raise ConversionError(f"{location}: invalid JSON") from error


def _load_json(path: Path) -> Any:
    try:
        return _loads(path.read_text(encoding="utf-8"), location=str(path))
    except (OSError, UnicodeError) as error:
        raise ConversionError(f"{path}: unreadable UTF-8 JSON") from error


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _is_sha256(value: object) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(character in "0123456789abcdef" for character in value)


def _exclusive_write(path: Path, payload: bytes) -> None:
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    try:
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
    except BaseException:
        path.unlink(missing_ok=True)
        raise


def _require_real_file(path: Path, *, root: Path) -> Path:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ConversionError(f"source file escapes root: {path}") from error
    cursor = root
    for part in relative.parts:
        cursor = cursor / part
        if cursor.is_symlink():
            raise ConversionError(f"source path contains a symlink: {cursor}")
    if not path.is_file():
        raise ConversionError(f"source file is missing: {path}")
    return path


def _validate_source_root(path: Path) -> Path:
    if not path.is_absolute() or path.is_symlink() or not path.is_dir():
        raise ConversionError("--source-root must be an absolute real directory")
    return path.resolve()


def _file_record(root: Path, path: Path) -> dict[str, object]:
    path = _require_real_file(path, root=root)
    return {
        "path": path.relative_to(root).as_posix(),
        "bytes": path.stat().st_size,
        "sha256": _sha256(path),
    }


def _inventory(root: Path, paths: list[Path]) -> dict[str, object]:
    records = [_file_record(root, path) for path in sorted(paths, key=lambda item: item.relative_to(root).as_posix())]
    aggregate = hashlib.sha256()
    for record in records:
        aggregate.update(str(record["path"]).encode("utf-8"))
        aggregate.update(b"\0")
        aggregate.update(str(record["sha256"]).encode("ascii"))
        aggregate.update(b"\n")
    return {
        "count": len(records),
        "total_bytes": sum(int(record["bytes"]) for record in records),
        "aggregate_sha256": aggregate.hexdigest(),
        "aggregate_algorithm": "sha256(relpath + NUL + file_sha256 + LF)",
        "files": records,
    }


def _inventory_summary(inventory: dict[str, object]) -> dict[str, object]:
    return {
        "count": inventory["count"],
        "total_bytes": inventory["total_bytes"],
        "aggregate_sha256": inventory["aggregate_sha256"],
    }


def _trial_sources(source_root: Path) -> list[tuple[Path, Path, Path]]:
    trajectories = sorted(source_root.glob("**/agent/trajectory.json"))
    if len(trajectories) != EXPECTED_TASKS:
        raise ConversionError(f"expected exactly {EXPECTED_TASKS} ATIF trajectories, found {len(trajectories)}")

    sources: list[tuple[Path, Path, Path]] = []
    seen_trials: set[Path] = set()
    for trajectory in trajectories:
        trajectory = _require_real_file(trajectory, root=source_root)
        if trajectory.parent.name != "agent":
            raise ConversionError(f"unexpected ATIF path: {trajectory}")
        trial_dir = trajectory.parent.parent
        if trial_dir in seen_trials:
            raise ConversionError(f"trial contains more than one ATIF trajectory: {trial_dir}")
        seen_trials.add(trial_dir)

        result = _require_real_file(trial_dir / "result.json", root=source_root)
        session_root = trajectory.parent / "CODEX_HOME" / "sessions"
        if session_root.is_symlink() or not session_root.is_dir():
            raise ConversionError(f"trial has no real native Codex session directory: {trial_dir}")
        sessions = sorted(session_root.glob("**/rollout-*.jsonl"))
        if len(sessions) != 1:
            raise ConversionError(f"trial must contain exactly one native Codex session: {trial_dir}")
        session = _require_real_file(sessions[0], root=source_root)
        sources.append((trajectory, result, session))
    return sources


def _expected_plan(path: Path) -> tuple[set[str], dict[str, str]]:
    if not path.is_absolute() or path.is_symlink() or not path.is_file():
        raise ConversionError("--expected-plan-manifest must be an absolute real file")
    raw = _load_json(path)
    if not isinstance(raw, dict) or raw.get("schema") != EXPECTED_PLAN_SCHEMA:
        raise ConversionError("expected plan manifest schema changed")
    terminal_bench = raw.get("terminal_bench")
    split = raw.get("split")
    if not isinstance(terminal_bench, dict) or not isinstance(split, dict):
        raise ConversionError("expected plan has no Terminal-Bench inventory/split")
    task_contracts = terminal_bench.get("task_contracts")
    task_inventory_sha256 = terminal_bench.get("task_inventory_sha256")
    if (
        terminal_bench.get("version") != TERMINAL_BENCH_VERSION
        or terminal_bench.get("task_count") != EXPECTED_TASKS
        or not isinstance(task_contracts, dict)
        or len(task_contracts) != EXPECTED_TASKS
        or any(
            not isinstance(task_id, str)
            or not task_id
            or "/" in task_id
            or not isinstance(contract, dict)
            for task_id, contract in task_contracts.items()
        )
        or not _is_sha256(task_inventory_sha256)
        or _canonical_sha256(task_contracts) != task_inventory_sha256
    ):
        raise ConversionError("expected plan Terminal-Bench task inventory changed")

    split_payload = {
        "seed": split.get("seed"),
        "algorithm": split.get("algorithm"),
        "train_task_ids": split.get("train_task_ids"),
        "eval_task_ids": split.get("eval_task_ids"),
    }
    train_tasks = split_payload["train_task_ids"]
    eval_tasks = split_payload["eval_task_ids"]
    split_sha256 = split.get("sha256")
    if (
        not isinstance(split_payload["seed"], str)
        or not split_payload["seed"]
        or split_payload["algorithm"] != "sha256-domain-ranked-first-44"
        or not isinstance(train_tasks, list)
        or not isinstance(eval_tasks, list)
        or len(train_tasks) != 44
        or len(eval_tasks) != 45
        or any(not isinstance(task_id, str) for task_id in train_tasks + eval_tasks)
        or len(set(train_tasks)) != 44
        or len(set(eval_tasks)) != 45
        or set(train_tasks) & set(eval_tasks)
        or set(train_tasks) | set(eval_tasks) != set(task_contracts)
        or split.get("train_task_count") != 44
        or split.get("eval_task_count") != 45
        or not _is_sha256(split_sha256)
        or _canonical_sha256(split_payload) != split_sha256
    ):
        raise ConversionError("expected plan 44/45 Terminal-Bench split changed")
    return set(task_contracts), {
        "plan_manifest_sha256": _sha256(path),
        "task_inventory_sha256": task_inventory_sha256,
        "split_sha256": split_sha256,
    }


def _terminal_bench_task_path(value: object, *, slug: str, location: str) -> None:
    if not isinstance(value, str):
        raise ConversionError(f"{location}: Terminal-Bench task path is missing")
    parts = Path(value).parts
    if len(parts) < 3 or parts[-3:] != (TERMINAL_BENCH_TASK_DIR, "tasks", slug):
        raise ConversionError(f"{location}: task is not from Terminal-Bench 2.1")


def _validate_result(
    path: Path,
    *,
    missing_reward_policy: str,
) -> tuple[str, str, float, bool]:
    raw = _load_json(path)
    if not isinstance(raw, dict):
        raise ConversionError(f"{path}: Harbor result must be an object")

    task_id = raw.get("task_name")
    trial_name = raw.get("trial_name")
    if not isinstance(task_id, str) or not task_id.startswith("terminal-bench/") or task_id.count("/") != 1:
        raise ConversionError(f"{path}: invalid Terminal-Bench task name")
    if not isinstance(trial_name, str) or trial_name != path.parent.name or trial_name.count("__") != 1:
        raise ConversionError(f"{path}: trial name/path binding changed")
    slug = task_id.removeprefix("terminal-bench/")
    if trial_name.split("__", 1)[0] != slug:
        raise ConversionError(f"{path}: task name does not bind the trial name")

    task_id_obj = raw.get("task_id")
    config = raw.get("config")
    agent_info = raw.get("agent_info")
    if not isinstance(task_id_obj, dict) or not isinstance(config, dict) or not isinstance(agent_info, dict):
        raise ConversionError(f"{path}: Harbor identity metadata is incomplete")
    _terminal_bench_task_path(task_id_obj.get("path"), slug=slug, location=str(path))
    config_task = config.get("task")
    config_agent = config.get("agent")
    model_info = agent_info.get("model_info")
    if not isinstance(config_task, dict) or not isinstance(config_agent, dict) or not isinstance(model_info, dict):
        raise ConversionError(f"{path}: Harbor config/agent metadata is incomplete")
    _terminal_bench_task_path(config_task.get("path"), slug=slug, location=str(path))
    if (
        raw.get("source") != "tasks"
        or config_agent.get("name") != "tb_codex_agent:TerminalBenchCodex"
        or config_agent.get("model_name") != SOURCE_MODEL
        or agent_info.get("name") != SOURCE_AGENT
        or agent_info.get("version") != SOURCE_AGENT_VERSION
        or model_info.get("name") != SOURCE_MODEL.split("/", 1)[1]
        or model_info.get("provider") != "openai"
    ):
        raise ConversionError(f"{path}: source model, Codex, or harness identity changed")
    if not _is_sha256(raw.get("task_checksum")):
        raise ConversionError(f"{path}: task checksum is missing")

    verifier = raw.get("verifier_result")
    mapped_timeout = False
    if isinstance(verifier, dict):
        rewards = verifier.get("rewards")
        if not isinstance(rewards, dict) or set(rewards) != {"reward"}:
            raise ConversionError(f"{path}: verifier reward schema changed")
        reward = rewards["reward"]
        if isinstance(reward, bool) or not isinstance(reward, (int, float)) or not math.isfinite(float(reward)):
            raise ConversionError(f"{path}: reward is not finite numeric data")
        reward = float(reward)
        if reward not in {0.0, 1.0}:
            raise ConversionError(f"{path}: teacher reward must be binary")
    elif verifier is None:
        exception = raw.get("exception_info")
        allowed = (
            missing_reward_policy == MISSING_REWARD_TIMEOUT_ZERO
            and task_id == KNOWN_TIMEOUT_TASK
            and isinstance(exception, dict)
            and exception.get("exception_type") == KNOWN_TIMEOUT_TYPE
            and exception.get("exception_message") == KNOWN_TIMEOUT_MESSAGE
            and isinstance(exception.get("exception_traceback"), str)
            and "harbor.trial.errors.AgentTimeoutError" in exception["exception_traceback"]
        )
        if not allowed:
            raise ConversionError(
                f"{path}: missing reward is forbidden; the sole known timeout requires "
                f"--missing-reward-policy {MISSING_REWARD_TIMEOUT_ZERO}"
            )
        reward = 0.0
        mapped_timeout = True
    else:
        raise ConversionError(f"{path}: verifier result schema changed")
    return task_id, trial_name, reward, mapped_timeout


def _native_session(
    path: Path,
    *,
    expected_session_id: str,
    fallback_task: str | None,
) -> str | None:
    session_meta: list[dict[str, Any]] = []
    turn_contexts: list[dict[str, Any]] = []
    reasoning_items: list[str] = []
    raw_reasoning_events: list[str] = []
    assistant_messages = 0
    task_complete: list[dict[str, Any]] = []

    try:
        stream = path.open("r", encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ConversionError(f"{path}: native Codex session is unreadable") from error
    with stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                raise ConversionError(f"{path}:{line_number}: blank JSONL record")
            raw = _loads(line, location=f"{path}:{line_number}")
            if not isinstance(raw, dict) or not isinstance(raw.get("type"), str) or not isinstance(raw.get("payload"), dict):
                raise ConversionError(f"{path}:{line_number}: malformed native Codex event")
            event_type = raw["type"]
            payload = raw["payload"]
            if event_type == "session_meta":
                session_meta.append(payload)
            elif event_type == "turn_context":
                turn_contexts.append(payload)
            elif event_type == "response_item" and payload.get("type") == "message":
                if payload.get("role") == "assistant":
                    assistant_messages += 1
            elif event_type == "response_item" and payload.get("type") == "reasoning":
                content = payload.get("content")
                if (
                    not isinstance(content, list)
                    or len(content) != 1
                    or not isinstance(content[0], dict)
                    or content[0].get("type") != "reasoning_text"
                    or not isinstance(content[0].get("text"), str)
                    or not content[0]["text"]
                ):
                    if fallback_task is not None:
                        raise ConversionError(f"{path}: fallback reasoning item schema changed")
                else:
                    reasoning_items.append(content[0]["text"])
            elif event_type == "event_msg" and payload.get("type") == "agent_reasoning_raw_content":
                text = payload.get("text")
                if isinstance(text, str) and text:
                    raw_reasoning_events.append(text)
            elif event_type == "event_msg" and payload.get("type") == "task_complete":
                task_complete.append(payload)

    if len(session_meta) != 1 or len(turn_contexts) < 1:
        raise ConversionError(f"{path}: native Codex identity records are incomplete")
    metadata = session_meta[0]
    if (
        metadata.get("session_id") != expected_session_id
        or metadata.get("id") != expected_session_id
        or metadata.get("cli_version") != SOURCE_AGENT_VERSION
        or metadata.get("model_provider") != "openai"
        or metadata.get("originator") != "codex_exec"
    ):
        raise ConversionError(f"{path}: native Codex session identity changed")
    for context in turn_contexts:
        collaboration = context.get("collaboration_mode")
        settings = collaboration.get("settings") if isinstance(collaboration, dict) else None
        if (
            context.get("model") != SOURCE_MODEL
            or context.get("effort") != SOURCE_REASONING_EFFORT
            or not isinstance(settings, dict)
            or settings.get("model") != SOURCE_MODEL
            or settings.get("reasoning_effort") != SOURCE_REASONING_EFFORT
        ):
            raise ConversionError(f"{path}: native model or xhigh reasoning contract changed")

    if fallback_task is None:
        return None
    if (
        len(reasoning_items) != 1
        or raw_reasoning_events != reasoning_items
        or assistant_messages != 0
        or len(task_complete) != 1
        or task_complete[0].get("last_agent_message") is not None
    ):
        raise ConversionError(f"{path}: {fallback_task} no-visible-output fallback is no longer exact")
    return reasoning_items[0]


def _atif_blocks(path: Path) -> tuple[str, list[list[dict[str, Any]]], int, int]:
    raw = _load_json(path)
    if not isinstance(raw, dict) or set(raw) != {"agent", "final_metrics", "schema_version", "session_id", "steps"}:
        raise ConversionError(f"{path}: ATIF top-level schema changed")
    if raw.get("schema_version") != SOURCE_TRAJECTORY_SCHEMA:
        raise ConversionError(f"{path}: expected {SOURCE_TRAJECTORY_SCHEMA}")
    session_id = raw.get("session_id")
    agent = raw.get("agent")
    steps = raw.get("steps")
    if not isinstance(session_id, str) or not session_id or not isinstance(agent, dict) or not isinstance(steps, list) or not steps:
        raise ConversionError(f"{path}: ATIF identity or steps are incomplete")
    if (
        agent.get("name") != SOURCE_AGENT
        or agent.get("version") != SOURCE_AGENT_VERSION
        or agent.get("model_name") != SOURCE_MODEL
    ):
        raise ConversionError(f"{path}: ATIF source identity changed")

    blocks: list[list[dict[str, Any]]] = []
    tool_call_count = 0
    agent_step_count = 0
    global_call_ids: set[str] = set()
    seen_agent_step = False
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict) or step.get("step_id") != index or not isinstance(step.get("message"), str):
            raise ConversionError(f"{path}: ATIF step IDs/messages are malformed")
        source = step.get("source")
        if source in {"system", "user"}:
            expected_keys = {"message", "source", "step_id", "timestamp"}
            if set(step) != expected_keys:
                raise ConversionError(f"{path}: {source} step schema changed")
            if source == "system" and index != 1:
                raise ConversionError(f"{path}: system step must be first")
            if source == "user" and seen_agent_step:
                raise ConversionError(f"{path}: later user turn would make target rendering non-append-only")
            blocks.append([{"role": source, "content": step["message"]}])
            continue
        if source != "agent" or step.get("model_name") != SOURCE_MODEL:
            raise ConversionError(f"{path}: unknown ATIF step source/model")
        seen_agent_step = True
        agent_step_count += 1
        assistant: dict[str, Any] = {"role": "assistant", "content": step["message"]}
        calls_raw = step.get("tool_calls")
        observation = step.get("observation")
        block: list[dict[str, Any]] = [assistant]
        if calls_raw is None:
            if observation is not None:
                raise ConversionError(f"{path}: agent observation has no tool calls")
        else:
            if not isinstance(calls_raw, list) or not calls_raw:
                raise ConversionError(f"{path}: tool_calls must be a non-empty list when present")
            calls: list[dict[str, Any]] = []
            call_ids: list[str] = []
            for call in calls_raw:
                if not isinstance(call, dict) or set(call) != {"arguments", "function_name", "tool_call_id"}:
                    raise ConversionError(f"{path}: ATIF tool-call schema changed")
                call_id = call.get("tool_call_id")
                name = call.get("function_name")
                arguments = call.get("arguments")
                if (
                    not isinstance(call_id, str)
                    or not call_id
                    or call_id in global_call_ids
                    or not isinstance(name, str)
                    or not name
                    or not isinstance(arguments, dict)
                ):
                    raise ConversionError(f"{path}: invalid or duplicate ATIF tool call")
                global_call_ids.add(call_id)
                call_ids.append(call_id)
                calls.append(
                    {
                        "id": call_id,
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                )
            assistant["tool_calls"] = calls
            if not isinstance(observation, dict) or set(observation) != {"results"}:
                raise ConversionError(f"{path}: tool observation schema changed")
            results = observation.get("results")
            if not isinstance(results, list) or len(results) != len(calls):
                raise ConversionError(f"{path}: tool results do not match calls")
            result_ids: list[str] = []
            for result in results:
                if not isinstance(result, dict) or set(result) != {"content", "source_call_id"}:
                    raise ConversionError(f"{path}: ATIF tool-result schema changed")
                source_call_id = result.get("source_call_id")
                content = result.get("content")
                if not isinstance(source_call_id, str) or not isinstance(content, str):
                    raise ConversionError(f"{path}: invalid ATIF tool result")
                result_ids.append(source_call_id)
                block.append({"role": "tool", "tool_call_id": source_call_id, "content": content})
            if result_ids != call_ids:
                raise ConversionError(f"{path}: tool results are not explicitly order-bound to calls")
            tool_call_count += len(calls)
        blocks.append(block)

    if not blocks or blocks[0][0].get("role") != "system" or not any(block[0].get("role") == "user" for block in blocks):
        raise ConversionError(f"{path}: ATIF prompt prefix is incomplete")
    return session_id, blocks, agent_step_count, tool_call_count


def _target_tokenizer(
    *,
    tokenizer_source: str,
    chat_template_path: Path,
    model: str,
    revision: str,
) -> tuple[Any, str, dict[str, object]]:
    if model != TARGET_MODEL or revision != TARGET_REVISION:
        raise ConversionError("pinned Qwen3.5-0.8B target model/revision changed")
    if not chat_template_path.is_absolute() or chat_template_path.is_symlink() or not chat_template_path.is_file():
        raise ConversionError("--target-chat-template must be an absolute real file")
    try:
        chat_template = chat_template_path.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        raise ConversionError("target chat template is unreadable UTF-8") from error
    if not chat_template:
        raise ConversionError("target chat template is empty")

    tokenizer_path = Path(tokenizer_source)
    load_source = tokenizer_source
    if tokenizer_path.exists():
        if not tokenizer_path.is_absolute() or tokenizer_path.is_symlink() or not tokenizer_path.is_dir():
            raise ConversionError("local --target-tokenizer must be an absolute real directory")
        load_source = str(tokenizer_path.resolve())
    elif tokenizer_source != TARGET_MODEL:
        raise ConversionError(f"remote --target-tokenizer must equal {TARGET_MODEL}")

    try:
        from miles.utils.processing_utils import load_tokenizer

        tokenizer = load_tokenizer(
            load_source,
            chat_template_path=str(chat_template_path),
            revision=revision,
            trust_remote_code=True,
        )
    except Exception as error:
        raise ConversionError("failed to load the supplied target tokenizer/chat template") from error
    if not getattr(tokenizer, "is_fast", False):
        raise ConversionError("target tokenizer must be fast so assistant spans can be proven")
    vocabulary = tokenizer.get_vocab()
    if (
        tokenizer.vocab_size != TARGET_TOKENIZER_BASE_VOCAB_SIZE
        or len(tokenizer) != TARGET_TOKENIZER_LENGTH
        or len(vocabulary) != TARGET_TOKENIZER_LENGTH
        or max(vocabulary.values()) != TARGET_TOKENIZER_LENGTH - 1
    ):
        raise ConversionError("target tokenizer vocabulary differs from pinned Qwen3.5-0.8B")
    if tokenizer.chat_template != chat_template:
        raise ConversionError("loaded target chat template differs from the supplied file")

    try:
        backend_serialization = tokenizer.backend_tokenizer.to_str()
    except Exception as error:
        raise ConversionError("target tokenizer backend cannot be serialized for provenance") from error
    component_files: dict[str, str] = {}
    local_source = Path(load_source)
    if local_source.is_dir():
        for name in ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "added_tokens.json"):
            candidate = local_source / name
            if candidate.is_file() and not candidate.is_symlink():
                component_files[name] = _sha256(candidate)
    digest_payload = {
        "schema": "miles.tokenizer-semantic-digest.v2",
        "model": model,
        "revision": revision,
        "backend_serialization_sha256": hashlib.sha256(backend_serialization.encode("utf-8")).hexdigest(),
        "special_tokens_map": tokenizer.special_tokens_map,
        "chat_template_sha256": _sha256(chat_template_path),
        "component_files": component_files,
    }
    tokenizer_digest = hashlib.sha256(_canonical(digest_payload)).hexdigest()
    details = {
        "source": load_source,
        "model": model,
        "revision": revision,
        "model_vocab_size": TARGET_MODEL_VOCAB_SIZE,
        "vocab_size": tokenizer.vocab_size,
        "tokenizer_length": len(tokenizer),
        "digest_schema": digest_payload["schema"],
        "digest_sha256": tokenizer_digest,
        "backend_serialization_sha256": digest_payload["backend_serialization_sha256"],
        "component_files": component_files,
        "chat_template_path": str(chat_template_path),
        "chat_template_sha256": digest_payload["chat_template_sha256"],
    }
    return tokenizer, tokenizer_digest, details


def _render(
    messages: list[dict[str, Any]],
    *,
    tokenizer: Any,
    add_generation_prompt: bool,
    tokenize: bool,
) -> str | list[int]:
    try:
        from miles.utils.chat_template_utils import apply_chat_template

        return apply_chat_template(
            messages,
            tokenizer=tokenizer,
            add_generation_prompt=add_generation_prompt,
            tokenize=tokenize,
            clear_thinking=False,
            enable_thinking=True,
        )
    except Exception as error:
        raise ConversionError("target Qwen3.5 chat-template rendering failed") from error


def _tokenize_with_assistant_mask(
    blocks: list[list[dict[str, Any]]],
    *,
    tokenizer: Any,
    location: str,
) -> tuple[list[int], list[int], int]:
    messages = [message for block in blocks for message in block]
    full_text = _render(messages, tokenizer=tokenizer, add_generation_prompt=False, tokenize=False)
    canonical_ids = _render(messages, tokenizer=tokenizer, add_generation_prompt=False, tokenize=True)
    if not isinstance(full_text, str) or not isinstance(canonical_ids, list):
        raise ConversionError(f"{location}: chat template returned an unexpected type")

    try:
        first_assistant = next(
            index for index, block in enumerate(blocks) if block[0].get("role") == "assistant"
        )
    except (IndexError, StopIteration) as error:
        raise ConversionError(f"{location}: conversation has no assistant block") from error
    preamble = blocks[:first_assistant]
    if not preamble or any(
        len(block) != 1 or block[0].get("role") not in {"system", "user"}
        for block in preamble
    ):
        raise ConversionError(f"{location}: conversation preamble is invalid")

    active_spans: list[tuple[int, int]] = []
    prefix = [message for block in preamble for message in block]
    rendered_prefix = _render(
        prefix,
        tokenizer=tokenizer,
        add_generation_prompt=False,
        tokenize=False,
    )
    if not isinstance(rendered_prefix, str):
        raise ConversionError(f"{location}: preamble rendering returned an unexpected type")
    for block in blocks[first_assistant:]:
        if not block:
            raise ConversionError(f"{location}: empty conversation block")
        first_role = block[0].get("role")
        if first_role == "assistant":
            if any(message.get("role") != "tool" for message in block[1:]):
                raise ConversionError(f"{location}: assistant block contains a non-tool result")
            generation = _render(prefix, tokenizer=tokenizer, add_generation_prompt=True, tokenize=False)
            after_assistant = _render(
                prefix + [block[0]],
                tokenizer=tokenizer,
                add_generation_prompt=False,
                tokenize=False,
            )
            if (
                not isinstance(generation, str)
                or not isinstance(after_assistant, str)
                or not generation.startswith(rendered_prefix)
                or not after_assistant.startswith(generation)
            ):
                raise ConversionError(f"{location}: assistant rendering is not append-only")
            active_spans.append((len(generation), len(after_assistant)))
        else:
            raise ConversionError(f"{location}: non-assistant turn follows the first assistant")

        after_block = _render(
            prefix + block,
            tokenizer=tokenizer,
            add_generation_prompt=False,
            tokenize=False,
        )
        if not isinstance(after_block, str) or not after_block.startswith(rendered_prefix):
            raise ConversionError(f"{location}: conversation rendering is not append-only")
        prefix.extend(block)
        rendered_prefix = after_block
    if rendered_prefix != full_text or not active_spans:
        raise ConversionError(f"{location}: full conversation or assistant coverage changed")

    encoded = tokenizer(full_text, add_special_tokens=False, return_offsets_mapping=True)
    token_ids = encoded.get("input_ids")
    offsets = encoded.get("offset_mapping")
    if token_ids != canonical_ids or not isinstance(offsets, list) or len(offsets) != len(token_ids):
        raise ConversionError(f"{location}: tokenizer/chat-template tokenization mismatch")

    boundaries = {boundary for span in active_spans for boundary in span}
    loss_mask: list[int] = []
    boundary_tokens = 0
    for ordinal, offset in enumerate(offsets):
        if (
            not isinstance(offset, (list, tuple))
            or len(offset) != 2
            or type(offset[0]) is not int
            or type(offset[1]) is not int
            or offset[0] < 0
            or offset[1] <= offset[0]
        ):
            raise ConversionError(f"{location}: token {ordinal} has no exact source-text span")
        start, end = offset
        if any(start < boundary < end for boundary in boundaries):
            boundary_tokens += 1
        # BPE tokens can straddle a rendered assistant boundary. Such a token
        # cannot be split; assign it to the assistant iff any of its source
        # span overlaps assistant-authored text.
        matches = sum(
            span_start < end and start < span_end
            for span_start, span_end in active_spans
        )
        if matches > 1:
            raise ConversionError(f"{location}: overlapping assistant spans")
        loss_mask.append(int(matches == 1))
    if not any(loss_mask):
        raise ConversionError(f"{location}: conversation has no trainable assistant token")
    return token_ids, loss_mask, boundary_tokens


def _suffix_window(
    tokens: list[int],
    full_loss_mask: list[int],
    *,
    max_seq_len: int,
    location: str,
) -> tuple[list[int], int, list[int], dict[str, int | bool]]:
    if max_seq_len < 2 or len(tokens) != len(full_loss_mask) or not any(full_loss_mask):
        raise ConversionError(f"{location}: invalid full-sequence window input")
    first_active = full_loss_mask.index(1)
    if first_active < 1:
        raise ConversionError(f"{location}: full sequence has no prompt token")
    response_mask = full_loss_mask[first_active:]
    response_capacity = max_seq_len - 1
    response_end = len(response_mask)
    response_start = max(0, response_end - response_capacity)
    if not any(response_mask[response_start:response_end]):
        last_active = max(index for index, value in enumerate(response_mask) if value)
        response_end = last_active + 1
        response_start = max(0, response_end - response_capacity)

    absolute_response_start = first_active + response_start
    absolute_end = first_active + response_end
    absolute_start = max(0, absolute_end - max_seq_len)
    if absolute_start >= absolute_response_start:
        absolute_start = absolute_response_start - 1
    window_tokens = tokens[absolute_start:absolute_end]
    window_mask = response_mask[response_start:response_end]
    if (
        absolute_start < 0
        or len(window_tokens) > max_seq_len
        or len(window_mask) >= len(window_tokens)
        or not any(window_mask)
    ):
        raise ConversionError(f"{location}: could not preserve a prompt prefix and assistant target")
    return (
        window_tokens,
        len(window_mask),
        window_mask,
        {
            "original_tokens": len(tokens),
            "window_tokens": len(window_tokens),
            "original_active_tokens": sum(full_loss_mask),
            "window_active_tokens": sum(window_mask),
            "response_window_start": response_start,
            "response_window_end": response_end,
            "windowed": len(window_tokens) != len(tokens),
        },
    )


def build(
    *,
    source_root: Path,
    expected_plan_manifest: Path,
    target_tokenizer_source: str,
    target_chat_template: Path,
    output_dir: Path,
    missing_reward_policy: str,
    model: str,
    revision: str,
    max_seq_len: int,
) -> Path:
    source_root = _validate_source_root(source_root)
    if output_dir.exists() or output_dir.is_symlink():
        raise ConversionError("output directory must be fresh")
    if missing_reward_policy not in {MISSING_REWARD_REJECT, MISSING_REWARD_TIMEOUT_ZERO}:
        raise ConversionError("unknown missing-reward policy")
    if max_seq_len != MAX_SEQ_LEN:
        raise ConversionError(f"max sequence length must remain pinned to {MAX_SEQ_LEN}")

    expected_plan_tasks, plan_binding = _expected_plan(expected_plan_manifest)

    tokenizer, tokenizer_digest, tokenizer_details = _target_tokenizer(
        tokenizer_source=target_tokenizer_source,
        chat_template_path=target_chat_template,
        model=model,
        revision=revision,
    )
    sources = _trial_sources(source_root)
    trajectory_paths = [source[0] for source in sources]
    result_paths = [source[1] for source in sources]
    session_paths = [source[2] for source in sources]

    rows: list[dict[str, object]] = []
    task_reports: list[dict[str, object]] = []
    seen_tasks: set[str] = set()
    seen_trials: set[str] = set()
    reward_counter: Counter[str] = Counter()
    mapped_timeout_tasks: list[str] = []
    fallback_tasks: list[str] = []
    total_tool_calls = 0
    total_agent_steps = 0

    for trajectory_path, result_path, session_path in sources:
        task_id, trial_name, reward, mapped_timeout = _validate_result(
            result_path,
            missing_reward_policy=missing_reward_policy,
        )
        if task_id in seen_tasks or trial_name in seen_trials:
            raise ConversionError("Terminal-Bench task/trial is not unique")
        seen_tasks.add(task_id)
        seen_trials.add(trial_name)

        session_id, blocks, agent_steps, tool_calls = _atif_blocks(trajectory_path)
        fallback_task = task_id if agent_steps == 0 else None
        if fallback_task is not None and fallback_task not in NATIVE_FALLBACK_TASKS:
            raise ConversionError(f"{task_id}: unexpected ATIF trace with no visible assistant step")
        fallback_reasoning = _native_session(
            session_path,
            expected_session_id=session_id,
            fallback_task=fallback_task,
        )
        if fallback_task is not None:
            if reward != 0.0 or fallback_reasoning is None:
                raise ConversionError(f"{task_id}: native fallback must be source-faithful zero-reward reasoning")
            blocks.append(
                [
                    {
                        "role": "assistant",
                        "content": "",
                        "reasoning_content": fallback_reasoning,
                    }
                ]
            )
            fallback_tasks.append(task_id)

        tokens, full_loss_mask, boundary_tokens = _tokenize_with_assistant_mask(
            blocks,
            tokenizer=tokenizer,
            location=task_id,
        )
        window_tokens, response_length, loss_mask, window_report = _suffix_window(
            tokens,
            full_loss_mask,
            max_seq_len=max_seq_len,
            location=task_id,
        )
        if any(
            type(token) is not int or token < 0 or token >= TARGET_TOKENIZER_LENGTH
            for token in window_tokens
        ):
            raise ConversionError(f"{task_id}: target token ID is outside the pinned vocabulary")

        reward_key = str(int(reward))
        reward_counter[reward_key] += 1
        if mapped_timeout:
            mapped_timeout_tasks.append(task_id)
        total_tool_calls += tool_calls
        total_agent_steps += agent_steps
        rows.append(
            {
                "sample_id": f"tb21-qwen38-teacher:{task_id}",
                "tokens": window_tokens,
                "response_length": response_length,
                "returns": [reward] * response_length,
                "loss_mask": loss_mask,
            }
        )
        task_reports.append(
            {
                "task_id": task_id,
                "trial_name": trial_name,
                "reward": reward,
                "mapped_missing_reward": mapped_timeout,
                "native_session_fallback": fallback_task is not None,
                "assistant_boundary_tokens": boundary_tokens,
                "agent_steps": agent_steps,
                "tool_calls": tool_calls,
                **window_report,
            }
        )

    if (
        len(seen_tasks) != EXPECTED_TASKS
        or len(rows) != EXPECTED_TASKS
        or dict(sorted(reward_counter.items())) != EXPECTED_REWARDS
        or mapped_timeout_tasks != [KNOWN_TIMEOUT_TASK]
        or sorted(fallback_tasks) != list(NATIVE_FALLBACK_TASKS)
        or {task_id.removeprefix("terminal-bench/") for task_id in seen_tasks} != expected_plan_tasks
    ):
        raise ConversionError(
            "teacher corpus coverage changed: expected 89 tasks, rewards 35/54, one known timeout, "
            "exactly two native-session fallbacks, and the exact planned task ledger"
        )

    inventories = {
        "trajectory": _inventory(source_root, trajectory_paths),
        "result": _inventory(source_root, result_paths),
        "session": _inventory(source_root, session_paths),
    }
    if any(inventory["count"] != EXPECTED_TASKS for inventory in inventories.values()):
        raise ConversionError("source inventory does not contain exactly one file of each kind per task")

    output_dir.mkdir(mode=0o700, parents=True)
    train_path = output_dir / "train.jsonl"
    train_payload = b"".join(_canonical(row) for row in sorted(rows, key=lambda row: str(row["sample_id"])))
    _exclusive_write(train_path, train_payload)
    train_sha256 = _sha256(train_path)

    report = {
        "schema": REPORT_SCHEMA,
        "provenance_schema": PROVENANCE_SCHEMA,
        "source": {
            "root": str(source_root),
            "expected_plan_manifest": str(expected_plan_manifest),
            **plan_binding,
            "terminal_bench_version": TERMINAL_BENCH_VERSION,
            "model": SOURCE_MODEL,
            "agent": SOURCE_AGENT,
            "agent_version": SOURCE_AGENT_VERSION,
            "trajectory_schema": SOURCE_TRAJECTORY_SCHEMA,
            "reasoning_effort": SOURCE_REASONING_EFFORT,
            "outcome_authentication_status": "legacy_unsigned",
            "inventories": inventories,
        },
        "target": tokenizer_details,
        "conversion": {
            "max_seq_len": max_seq_len,
            "window_policy": "one_contiguous_response_suffix_with_one_or_more_prompt_tokens",
            "loss_mask": "assistant_only",
            "assistant_boundary_token_policy": "overlap_to_assistant",
            "tool_structure_preserved": True,
            "tool_definitions_included": False,
            "tool_definition_source": "absent_in_atif",
            "missing_reward_policy": missing_reward_policy,
            "known_timeout_zero_tasks": mapped_timeout_tasks,
            "native_session_fallback_tasks": sorted(fallback_tasks),
            "native_session_fallback_count": len(fallback_tasks),
            "split": "train_only",
        },
        "coverage": {
            "planned_tasks": EXPECTED_TASKS,
            "included_tasks": len(seen_tasks),
            "included_trajectories": len(rows),
            "included_samples": len(rows),
            "one_trace_per_task": True,
            "all_tasks_exactly_once": True,
            "reward_counts": dict(sorted(reward_counter.items())),
            "agent_steps": total_agent_steps,
            "tool_calls": total_tool_calls,
            "windowed_samples": sum(bool(report["windowed"]) for report in task_reports),
            "active_tokens": sum(int(report["window_active_tokens"]) for report in task_reports),
            "assistant_boundary_tokens": sum(
                int(report["assistant_boundary_tokens"]) for report in task_reports
            ),
            "task_ids": sorted(seen_tasks),
        },
        "tasks": sorted(task_reports, key=lambda report: str(report["task_id"])),
        "outputs": {
            "train_path": train_path.name,
            "train_sha256": train_sha256,
            "train_bytes": train_path.stat().st_size,
            "train_samples": len(rows),
        },
    }
    report_path = output_dir / "report.json"
    _exclusive_write(report_path, _canonical(report))
    report_sha256 = _sha256(report_path)

    inventory_summaries = {name: _inventory_summary(inventory) for name, inventory in inventories.items()}
    provenance = {
        "schema": PROVENANCE_SCHEMA,
        "model": model,
        "revision": revision,
        "tokenizer_digest": tokenizer_digest,
        "chat_template_sha256": tokenizer_details["chat_template_sha256"],
        "max_seq_len": max_seq_len,
        "source_model": SOURCE_MODEL,
        "source_agent": SOURCE_AGENT,
        "source_agent_version": SOURCE_AGENT_VERSION,
        "source_trajectory_schema": SOURCE_TRAJECTORY_SCHEMA,
        "reasoning_effort": SOURCE_REASONING_EFFORT,
        "terminal_bench_version": TERMINAL_BENCH_VERSION,
        **plan_binding,
        "planned_tasks": EXPECTED_TASKS,
        "included_tasks": len(seen_tasks),
        "included_trajectories": len(rows),
        "included_samples": len(rows),
        "one_trace_per_task": True,
        "all_tasks_exactly_once": True,
        "reward_counts": dict(sorted(reward_counter.items())),
        "known_timeout_zero_mappings": len(mapped_timeout_tasks),
        "missing_reward_policy": missing_reward_policy,
        "native_session_fallback_tasks": sorted(fallback_tasks),
        "native_session_fallback_count": len(fallback_tasks),
        "outcome_authentication_status": "legacy_unsigned",
        "legacy_unsigned_outcomes": True,
        "all_terminal_bench_outcomes_authenticated": False,
        "assistant_only_loss_mask": True,
        "assistant_boundary_token_policy": "overlap_to_assistant",
        "tool_structure_preserved": True,
        "tool_definitions_included": False,
        "tool_definition_source": "absent_in_atif",
        "source_inventories": inventory_summaries,
        "conversion_report_path": report_path.name,
        "conversion_report_sha256": report_sha256,
    }
    manifest = {
        "schema": MANIFEST_SCHEMA,
        "train": {
            "path": train_path.name,
            "sha256": train_sha256,
            "num_samples": len(rows),
        },
        "heldout": None,
        "objective": {
            "loss_type": "classification",
            "num_bins": 51,
            "reward_range": [0.0, 1.0],
            "target_type": "hl_gauss",
            "hl_gauss_sigma_ratio": 0.75,
        },
        "provenance": provenance,
    }
    manifest_path = output_dir / "manifest.json"
    _exclusive_write(manifest_path, _canonical(manifest))
    return manifest_path


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--expected-plan-manifest", type=Path, required=True)
    parser.add_argument("--target-tokenizer", required=True)
    parser.add_argument("--target-chat-template", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--missing-reward-policy",
        choices=(MISSING_REWARD_REJECT, MISSING_REWARD_TIMEOUT_ZERO),
        default=MISSING_REWARD_REJECT,
    )
    parser.add_argument("--model", default=TARGET_MODEL)
    parser.add_argument("--revision", default=TARGET_REVISION)
    parser.add_argument("--max-seq-len", type=int, default=MAX_SEQ_LEN)
    args = parser.parse_args(argv)

    manifest = build(
        source_root=args.source_root,
        expected_plan_manifest=args.expected_plan_manifest,
        target_tokenizer_source=args.target_tokenizer,
        target_chat_template=args.target_chat_template,
        output_dir=args.output_dir.resolve(),
        missing_reward_policy=args.missing_reward_policy,
        model=args.model,
        revision=args.revision,
        max_seq_len=args.max_seq_len,
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest),
                "manifest_sha256": _sha256(manifest),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
