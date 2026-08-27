"""Signed stock-Codex bridge for canonical Terminal-Bench 2.1 episodes.

This module runs only in the isolated OpenEnv child interpreter.  It reuses
Yeto's attested stock-Codex app-server/Responses/Miles bridge, replacing only
the model-facing task/tool descriptions and the environment client.  Miles
therefore remains the sampler and owner of exact token IDs, behavior logprobs,
and action masks; OpenEnv never enters the Ray learner process.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import hmac
import json
import math
import os
import time
import uuid
from pathlib import Path
from typing import Any

import openenv_agent_function as openenv
from yeto.rl.codex_backend import QWEN35_08B_MODEL, QWEN35_08B_REVISION
from yeto.rl.tbench_outcome import (
    NATIVE_VERIFIER,
    TEST_SH_VERIFIER,
    TIMEOUT_VERIFIER,
    build_signed_metadata,
)
from yeto_miles_secrlenv import agent as legacy
from yeto_miles_secrlenv import codex_harness_agent as stock

BASE_INSTRUCTIONS = """You are an autonomous terminal agent solving one
authorized Terminal-Bench task inside its isolated task container. Use only the
two model-facing tools, `terminal.exec` and `submit`. `terminal.exec` executes a
shell command in the task's persistent workspace. Inspect the environment,
implement the requested change, and run appropriate checks. When the task is
complete, call `submit` exactly once with concise, concrete evidence. The
`submit` call is terminal: make no later model or tool calls. Do not merely
describe commands, invent tool results, or claim success without evidence."""

TERMINAL_EXEC_TOOL = copy.deepcopy(stock.TERMINAL_EXEC_TOOL)
TERMINAL_EXEC_TOOL["description"] = (
    "Execute one shell command in the persistent isolated Terminal-Bench task "
    "workspace and return its exit status and bounded output."
)
SUBMIT_TOOL = copy.deepcopy(stock.SUBMIT_TOOL)
SUBMIT_TOOL["description"] = (
    "Finish the Terminal-Bench episode after the requested work has been "
    "implemented and checked. Supply concise evidence; canonical evaluation "
    "runs after this terminal call."
)
DYNAMIC_TOOLS = [TERMINAL_EXEC_TOOL, SUBMIT_TOOL]
_EXEC_RESPONSE_GRACE_S = 30.0


def _canonical_json(value: Any) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _sha256_json(value: Any) -> str:
    return hashlib.sha256(_canonical_json(value)).hexdigest()


BASE_INSTRUCTIONS_SHA256 = hashlib.sha256(BASE_INSTRUCTIONS.encode()).hexdigest()
TERMINAL_EXEC_TOOL_SCHEMA_SHA256 = _sha256_json(TERMINAL_EXEC_TOOL)
SUBMIT_TOOL_SCHEMA_SHA256 = _sha256_json(SUBMIT_TOOL)
DYNAMIC_TOOLS_SCHEMA_SHA256 = _sha256_json(DYNAMIC_TOOLS)

_OPENENV_IDENTITY_ENV = {
    "YETO_CODEX_OPENENV_BACKEND_PROFILE": "qwen35_08b",
    "YETO_CODEX_OPENENV_MODEL_ID": QWEN35_08B_MODEL,
    "YETO_CODEX_OPENENV_MODEL_REVISION": QWEN35_08B_REVISION,
    "YETO_CODEX_OPENENV_BASE_INSTRUCTIONS_SHA256": BASE_INSTRUCTIONS_SHA256,
    "YETO_CODEX_OPENENV_TERMINAL_EXEC_TOOL_SCHEMA_SHA256": (
        TERMINAL_EXEC_TOOL_SCHEMA_SHA256
    ),
    "YETO_CODEX_OPENENV_SUBMIT_TOOL_SCHEMA_SHA256": SUBMIT_TOOL_SCHEMA_SHA256,
    "YETO_CODEX_OPENENV_DYNAMIC_TOOLS_SCHEMA_SHA256": DYNAMIC_TOOLS_SCHEMA_SHA256,
}


def codex_openenv_harness_identity() -> dict[str, str]:
    """Return recomputed identities for launch-time attestation."""

    values = {
        "base_instructions_sha256": hashlib.sha256(
            BASE_INSTRUCTIONS.encode()
        ).hexdigest(),
        "terminal_exec_tool_schema_sha256": _sha256_json(TERMINAL_EXEC_TOOL),
        "submit_tool_schema_sha256": _sha256_json(SUBMIT_TOOL),
        "dynamic_tools_schema_sha256": _sha256_json(DYNAMIC_TOOLS),
    }
    expected = {
        "base_instructions_sha256": BASE_INSTRUCTIONS_SHA256,
        "terminal_exec_tool_schema_sha256": TERMINAL_EXEC_TOOL_SCHEMA_SHA256,
        "submit_tool_schema_sha256": SUBMIT_TOOL_SCHEMA_SHA256,
        "dynamic_tools_schema_sha256": DYNAMIC_TOOLS_SCHEMA_SHA256,
    }
    if values != expected:
        raise stock.CodexHarnessError("Codex OpenEnv surface identity drifted")
    return values


def _attest_and_install_openenv_surface() -> Path:
    """Attest stock Codex first, then specialize this one child process.

    Every Miles rollout invokes this module in a fresh subprocess.  Rebinding
    the bridge globals is therefore process-local and cannot race another
    episode.  Keeping the stock implementation intact preserves its exact
    Responses history validation, app-server protocol checks, and TITO path.
    """

    binary = stock._attest_runtime()
    codex_openenv_harness_identity()
    for name, expected in _OPENENV_IDENTITY_ENV.items():
        if not hmac.compare_digest(os.getenv(name, ""), expected):
            raise stock.CodexHarnessError(
                f"{name} does not match the signed Codex OpenEnv surface"
            )
    if (
        stock._BACKEND_PROFILE.get("model_identifier") != QWEN35_08B_MODEL
        or stock._BACKEND_PROFILE.get("model_revision") != QWEN35_08B_REVISION
        or stock._BACKEND_PROFILE.get("tito_model") != "qwen35"
        or stock.BACKEND_MODEL != "qwen35"
    ):
        raise stock.CodexHarnessError(
            "Codex OpenEnv Qwen3.5-0.8B backend identity drifted"
        )

    stock.BASE_INSTRUCTIONS = BASE_INSTRUCTIONS
    stock.TERMINAL_EXEC_TOOL = copy.deepcopy(TERMINAL_EXEC_TOOL)
    stock.SUBMIT_TOOL = copy.deepcopy(SUBMIT_TOOL)
    stock.DYNAMIC_TOOLS = copy.deepcopy(DYNAMIC_TOOLS)
    stock._MILES_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "terminal.exec",
                "description": TERMINAL_EXEC_TOOL["description"],
                "parameters": copy.deepcopy(TERMINAL_EXEC_TOOL["inputSchema"]),
            },
        },
        {
            "type": "function",
            "function": {
                "name": "submit",
                "description": SUBMIT_TOOL["description"],
                "parameters": copy.deepcopy(SUBMIT_TOOL["inputSchema"]),
            },
        },
    ]
    stock._EXPECTED_RESPONSE_TOOLS = [
        stock._response_tool(tool) for tool in stock.DYNAMIC_TOOLS
    ]
    return binary


def _observation(result: Any) -> Any:
    return getattr(result, "observation", result)


def _field(result: Any, name: str, default: Any = None) -> Any:
    observation = _observation(result)
    return getattr(observation, name, getattr(result, name, default))


def _bounded_output(value: Any) -> tuple[str, bool]:
    output = value if isinstance(value, str) else str(value or "")
    raw = output.encode("utf-8")
    if len(raw) <= stock.MAX_TOOL_OUTPUT_BYTES:
        return output, False
    marker = b"\n[output truncated by Codex OpenEnv adapter]"
    prefix = raw[: max(0, stock.MAX_TOOL_OUTPUT_BYTES - len(marker))]
    output = prefix.decode("utf-8", errors="ignore") + marker.decode()
    return output, True


class _OpenEnvEpisodeClient:
    """Duck-typed environment client consumed by the stock app-server driver."""

    def __init__(self, env: Any, action_cls: Any, *, native_evaluate: bool) -> None:
        self._env = env
        self._action_cls = action_cls
        self._native_evaluate = native_evaluate
        self._submitted: dict[str, Any] | None = None

    async def execute(
        self,
        episode_id: str,
        command: str,
        *,
        timeout_seconds: float,
        output_bytes: int,
    ) -> dict[str, Any]:
        del episode_id
        if output_bytes != stock.MAX_TOOL_OUTPUT_BYTES:
            raise stock.CodexHarnessError("OpenEnv tool output boundary drifted")
        actual = command if self._native_evaluate else openenv._apply_workdir(command)
        timed_out = False
        try:
            result = await asyncio.wait_for(
                self._env.step(
                    self._action_cls(
                        action_type="exec",
                        command=actual,
                        wait_seconds=timeout_seconds,
                    )
                ),
                timeout=timeout_seconds + _EXEC_RESPONSE_GRACE_S,
            )
        except asyncio.TimeoutError:
            timed_out = True
            result = None
        error = _field(result, "error", "") if result is not None else ""
        raw_output = _field(result, "output", "") if result is not None else ""
        if error and not raw_output:
            raw_output = f"environment error: {error}"
        output, truncated = _bounded_output(raw_output)
        info = _field(result, "info", {}) if result is not None else {}
        info = info if isinstance(info, dict) else {}
        exit_code = (
            info.get("exit_code", _field(result, "exit_code"))
            if result is not None
            else None
        )
        return {
            "exit_code": exit_code,
            "output": output,
            "timed_out": timed_out
            or bool(info.get("timed_out", _field(result, "timed_out", False))),
            "truncated": truncated or bool(_field(result, "truncated", False)),
        }

    async def submit(
        self, episode_id: str, submission: dict[str, Any]
    ) -> dict[str, Any]:
        del episode_id
        if self._submitted is not None:
            raise stock.EpisodeAPIError(400, "duplicate_submit", "already submitted")
        self._submitted = copy.deepcopy(submission)
        return {"accepted": True}

    async def evaluate(self, episode_id: str) -> dict[str, Any]:
        del episode_id
        if self._native_evaluate:
            result = await self._env.step(self._action_cls(action_type="evaluate"))
            reward = getattr(result, "reward", None)
            error = _field(result, "error", "")
            if reward is None or error:
                raise RuntimeError(
                    f"Terminal-Bench canonical evaluate produced no verdict: {error!r}"
                )
            return {"reward": float(reward), "testsh_rc": None}

        result = await self._env.step(
            self._action_cls(
                action_type="exec", command=openenv._CANONICAL_EVAL_CMD
            )
        )
        output = str(_field(result, "output", "") or "")
        reward = openenv._parse_reward_marker(output)
        if reward is None:
            raise RuntimeError("Terminal-Bench canonical test.sh produced no verdict")
        return {
            "reward": reward,
            "testsh_rc": openenv._parse_testsh_rc(output),
        }

    @property
    def submission(self) -> dict[str, Any] | None:
        return copy.deepcopy(self._submitted)


def _task_id(metadata: dict[str, Any]) -> str:
    value = metadata.get("task_id") or metadata.get("task_name")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Terminal-Bench metadata must contain task_id")
    return value


def _sample_id(metadata: dict[str, Any]) -> str:
    value = metadata.get("sample_id")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Terminal-Bench metadata must contain sample_id")
    return value


async def _run_admitted_episode(
    env: Any,
    *,
    binary: Path,
    base_url: str,
    task_id: str,
    sample_id: str,
    episode_id: str,
    request_kwargs: dict[str, Any],
    metadata: dict[str, Any],
    native_evaluate: bool,
    metrics: legacy.AgentMetrics,
) -> dict[str, Any] | None:
    classes = openenv._load_tbench2()
    action_cls = classes["action"]
    started = time.monotonic()
    try:
        reset = await env.reset(task_id=task_id)
    finally:
        metrics.create_time = time.monotonic() - started
    instruction = openenv._obs_field(reset, "instruction")
    if not instruction:
        raise RuntimeError("Terminal-Bench reset returned no task instruction")

    episode = {"episode_id": episode_id, "prompt": instruction}
    client = _OpenEnvEpisodeClient(
        env, action_cls, native_evaluate=native_evaluate
    )
    max_seq_len = legacy._metadata_max_seq_len(metadata)
    outcome_status = await stock._drive_codex(
        binary,
        base_url,
        client,
        episode,
        request_kwargs,
        metrics,
        max_seq_len=max_seq_len,
    )

    started = time.monotonic()
    try:
        evaluation = await client.evaluate(episode_id)
    finally:
        metrics.evaluate_time = time.monotonic() - started
    metrics.total_tool_time += metrics.create_time + metrics.evaluate_time
    reward = evaluation.get("reward")
    if (
        isinstance(reward, bool)
        or not isinstance(reward, (int, float))
        or not math.isfinite(float(reward))
        or float(reward) not in {0.0, 1.0}
    ):
        raise RuntimeError("Terminal-Bench verifier returned a non-binary reward")
    return {
        "reward": float(reward),
        "exit_status": outcome_status,
        "eval_report": {},
        "agent_metrics": {
            **metrics.to_dict(),
            "testsh_rc": evaluation.get("testsh_rc"),
            "terminal_submission": client.submission,
        },
        **build_signed_metadata(
            task_id=task_id,
            sample_id=sample_id,
            episode_id=episode_id,
            status=outcome_status,
            reward=float(reward),
            verifier=(NATIVE_VERIFIER if native_evaluate else TEST_SH_VERIFIER),
            testsh_rc=evaluation.get("testsh_rc"),
        ),
    }


async def run(
    base_url: str,
    prompt: Any,
    request_kwargs: dict[str, Any] | None = None,
    metadata: dict[str, Any] | None = None,
    **_kwargs: Any,
) -> dict[str, Any] | None:
    """Run one canonical Terminal-Bench episode under stock Codex."""

    del prompt
    request_kwargs = dict(request_kwargs or {})
    metadata = dict(metadata or {})
    try:
        task_id = _task_id(metadata)
        sample_id = _sample_id(metadata)
        binary = _attest_and_install_openenv_surface()
        classes = openenv._load_tbench2()
        native_evaluate = openenv._shared_native_evaluate_enabled()
        episode_id = f"openenv-{uuid.uuid4().hex}"
        metrics = legacy.AgentMetrics()
        timeout = legacy._positive_env(
            "OPENENV_MAX_ROLLOUT_TIME_SECONDS", 1800.0
        )

        async def body(env: Any) -> dict[str, Any] | None:
            try:
                return await _run_admitted_episode(
                    env,
                    binary=binary,
                    base_url=base_url,
                    task_id=task_id,
                    sample_id=sample_id,
                    episode_id=episode_id,
                    request_kwargs=request_kwargs,
                    metadata=metadata,
                    native_evaluate=native_evaluate,
                    metrics=metrics,
                )
            finally:
                await openenv._purge_trial_dirs(env, classes["action"])

        try:
            # One deadline owns shared-server admission, reset, all Codex/tool
            # turns, canonical evaluation, and the shared-server purge.  In
            # particular, a queued capacity retry no longer gets a fresh 30
            # minutes before the actual episode begins.
            return await asyncio.wait_for(
                openenv._shared_run_body(classes["env"], metadata, body),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            metrics.timed_out = 1
            return {
                "reward": 0.0,
                "exit_status": "timeout",
                "eval_report": {},
                "agent_metrics": metrics.to_dict(),
                **build_signed_metadata(
                    task_id=task_id,
                    sample_id=sample_id,
                    episode_id=episode_id,
                    status="timeout",
                    reward=0.0,
                    verifier=TIMEOUT_VERIFIER,
                    testsh_rc=None,
                ),
            }
    except asyncio.CancelledError:
        raise
    except Exception:
        stock.LOGGER.exception(
            "Codex OpenEnv episode failed before a trustworthy verifier verdict"
        )
        # The outer subprocess/agentic boundary converts this failure into one
        # bounded-diagnostic ABORTED sample. Returning ``None`` here would make
        # already-recorded model turns look trainable until reward verification.
        raise


run._miles_fail_closed_result = True
