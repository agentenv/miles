"""Centralized SAO/SecRLEnv entrypoint with a hash-bound Yeto context.

This is the temporary online probe entrypoint. It intentionally does not
configure an external policy synchronizer: Miles publishes the trained actor
directly to SGLang. Streaming DiLoCo later replaces that boundary.
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import re
import sys
from pathlib import Path
from typing import Any


_SCHEMA = "miles.sao-secrlenv.v1"
_HASH = re.compile(r"[0-9a-f]{64}")
_REVISION = re.compile(r"[0-9a-f]{40,64}")
_REQUIRED = {
    "schema",
    "model",
    "data",
    "data_sha256",
    "base_model_revision",
    "rollout_model_revision",
    "data_revision",
    "layout_hash",
    "lora_config_hash",
    "reward_sha256",
    "dynamic_sampling_max_replacements",
    "secrlenv_max_infrastructure_replacements",
    "completed_groups_path",
    "event_tape",
    "learner_id",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_context(path: str | Path, expected_sha256: str | None = None) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if expected_sha256 is not None and (not _HASH.fullmatch(expected_sha256) or _sha256(source) != expected_sha256):
        raise ValueError("SAO SecRLEnv context SHA256 mismatch")
    try:
        context = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError("SAO SecRLEnv context is unreadable or malformed") from error
    if not isinstance(context, dict) or set(context) != _REQUIRED:
        raise ValueError("SAO SecRLEnv context fields do not match the v1 schema")
    if context["schema"] != _SCHEMA:
        raise ValueError(f"unsupported SAO SecRLEnv context schema: {context['schema']!r}")
    if not isinstance(context["model"], str) or not context["model"]:
        raise ValueError("SAO SecRLEnv model identity is missing")
    for name in ("base_model_revision", "rollout_model_revision"):
        value = context[name]
        if not isinstance(value, str) or not _REVISION.fullmatch(value):
            raise ValueError(f"SAO SecRLEnv {name} must be an immutable lowercase revision")
    for name in ("data_sha256", "layout_hash", "reward_sha256"):
        value = context[name]
        if not isinstance(value, str) or not _HASH.fullmatch(value):
            raise ValueError(f"SAO SecRLEnv {name} must be a lowercase SHA256")
    if context["data_revision"] is not None and not isinstance(context["data_revision"], str):
        raise ValueError("SAO SecRLEnv data_revision must be a string or null")
    if not isinstance(context["lora_config_hash"], str) or not context["lora_config_hash"]:
        raise ValueError("SAO SecRLEnv lora_config_hash is missing")
    for name in (
        "dynamic_sampling_max_replacements",
        "secrlenv_max_infrastructure_replacements",
        "learner_id",
    ):
        value = context[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError(f"SAO SecRLEnv {name} must be a non-negative integer")

    data = Path(context["data"])
    if not data.is_absolute() or not data.is_file() or data.is_symlink():
        raise ValueError("SAO SecRLEnv data must be an absolute regular non-symlink file")
    if _sha256(data) != context["data_sha256"]:
        raise ValueError("SAO SecRLEnv dataset SHA256 mismatch")
    for name in ("completed_groups_path", "event_tape"):
        output = Path(context[name])
        if not output.is_absolute() or output.is_symlink():
            raise ValueError(f"SAO SecRLEnv {name} must be an absolute non-symlink path")
        output.parent.mkdir(parents=True, exist_ok=True)
    return context


def bind_context(args: Any, context: dict[str, Any]) -> None:
    args.yeto_rl_model = context["model"]
    args.yeto_rl_data = context["data"]
    args.yeto_rl_base_model_revision = context["base_model_revision"]
    args.yeto_rl_rollout_model_revision = context["rollout_model_revision"]
    args.yeto_rl_data_revision = context["data_revision"]
    args.yeto_rl_layout_hash = context["layout_hash"]
    args.yeto_rl_lora_config_hash = context["lora_config_hash"]
    args.yeto_rl_reward_sha256 = context["reward_sha256"]
    args.yeto_rl_dynamic_sampling_max_replacements = context[
        "dynamic_sampling_max_replacements"
    ]
    args.yeto_rl_secrlenv_max_infrastructure_replacements = context[
        "secrlenv_max_infrastructure_replacements"
    ]
    args.yeto_rl_completed_groups_path = context["completed_groups_path"]
    args.yeto_rl_event_tape = context["event_tape"]
    args.yeto_rl_learner_id = context["learner_id"]
    args.yeto_rl_sync_preset = "strict-avg"
    args.wandb = bool(getattr(args, "use_wandb", False))
    args.wandb_entity = getattr(args, "wandb_team", None)


def main(argv: list[str] | None = None) -> None:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--sao-secrlenv-context", required=True)
    parser.add_argument("--sao-secrlenv-context-sha256")
    context_args, miles_argv = parser.parse_known_args(raw_argv)
    context = load_context(
        context_args.sao_secrlenv_context,
        context_args.sao_secrlenv_context_sha256,
    )

    previous = sys.argv
    try:
        sys.argv = [previous[0], *miles_argv]
        from miles.utils.arguments import parse_args

        args = parse_args()
    finally:
        sys.argv = previous
    if args.sao_online_recipe is None:
        raise ValueError("centralized SAO probe requires --sao-online-recipe")
    if args.external_policy_sync_path is not None:
        raise ValueError("centralized SAO probe must not configure --external-policy-sync-path")
    bind_context(args, context)

    from miles.utils.tracking_utils.tracking import finish_tracking
    from train import train

    try:
        asyncio.run(train(args))
    finally:
        finish_tracking()


if __name__ == "__main__":
    main()
