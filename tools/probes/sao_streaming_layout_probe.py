"""Metadata-only actor/critic layout attestation for streaming SAO.

The probe constructs both real Miles training groups, asks each group for its
topology-owned FP32 master manifests, and derives the exact role-domain-
separated Yeto layouts.  It never installs plans, exports parameter payloads,
runs a rollout, or executes an optimizer step.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import tempfile
from dataclasses import asdict
from pathlib import Path
from typing import Any

_EVIDENCE_ENV = "YETO_SAO_STREAMING_LAYOUT_EVIDENCE"
_SECRLENV_CONTEXT_SHA_ENV = "YETO_SAO_SECRLENV_CONTEXT_SHA256"
_ACTOR_REVISION_ENV = "YETO_SAO_ACTOR_MODEL_REVISION"
_ACTOR_CONFIG_ENV = "YETO_SAO_ACTOR_CONFIG_SHA256"
_CRITIC_REVISION_ENV = "YETO_SAO_CRITIC_MODEL_REVISION"
_CRITIC_CONFIG_ENV = "YETO_SAO_CRITIC_CONFIG_SHA256"
_MAX_FRAGMENT_ENV = "YETO_SAO_STREAMING_MAX_FRAGMENT_BYTES"
_MAX_CHUNK_ENV = "YETO_SAO_STREAMING_MAX_CHUNK_BYTES"
_SYNCER_PROFILE_ENV = "YETO_SAO_SYNCER_PROFILE_JSON"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_REVISION = re.compile(r"(?:[0-9a-f]{40}|[0-9a-f]{64})\Z")


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"SAO streaming layout probe is missing {name}")
    return value


def _hash_env(name: str) -> str:
    value = _required_env(name)
    if not _SHA256.fullmatch(value):
        raise RuntimeError(f"SAO streaming layout probe {name} is not a SHA256")
    return value


def _revision_env(name: str) -> str:
    value = _required_env(name)
    if not _REVISION.fullmatch(value):
        raise RuntimeError(f"SAO streaming layout probe {name} is not a commit")
    return value


def _integer_env(name: str) -> int:
    raw = _required_env(name)
    if not raw.isdecimal():
        raise RuntimeError(f"SAO streaming layout probe {name} is not an integer")
    return int(raw)


def _write_private_json(path: Path, payload: dict[str, object]) -> str:
    if not path.is_absolute() or path.exists() or path.is_symlink() or not path.parent.is_dir() or path.parent.is_symlink():
        raise RuntimeError("SAO streaming layout evidence path is not fresh and safe")
    descriptor, raw = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    temporary = Path(raw)
    try:
        os.fchmod(descriptor, 0o600)
        encoded = (json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n").encode()
        with os.fdopen(descriptor, "wb", closefd=True) as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
    return hashlib.sha256(encoded).hexdigest()


def _role_evidence(adapter, component) -> dict[str, object]:
    fragments = []
    for fragment_id, fragment in enumerate(adapter.layout.fragments.fragments):
        role, shard_id = adapter.layout.fragment_owner(fragment_id)
        fragments.append(
            {
                "fragment_id": fragment_id,
                "role": role,
                "shard_id": shard_id,
                "parameter_count": len(adapter.layout.fragment_specs(fragment_id)),
                "scalar_count": fragment.numel,
                "payload_bytes": fragment.numel * 4,
            }
        )
    manifests = [
        {
            "topology": asdict(manifest.topology),
            "source_layout_sha256": manifest.layout_hash,
            "parameter_count": len(manifest.specs),
            "scalar_count": sum(spec.numel for spec in manifest.specs),
        }
        for manifest in adapter.manifests
    ]
    return {
        "role": component.role,
        "component": {
            "model_revision": component.model_revision,
            "config_sha256": component.config_hash,
        },
        "parameter_layout_sha256": adapter.layout.layout_hash,
        "expected_fragments": adapter.layout.fragments.num_fragments,
        "parameter_tensor_count": adapter.expected_parameter_tensor_count,
        "parameter_scalar_count": adapter.expected_parameter_scalar_count,
        "fragments": fragments,
        "manifests": manifests,
    }


class SAOStreamingLayoutProbe:
    """Derive both real role layouts and stop before any online work."""

    supports_critic = True

    def __init__(self, args: Any) -> None:
        self.args = args
        self.evidence_path = Path(_required_env(_EVIDENCE_ENV))
        self.secrlenv_context_sha256 = _hash_env(_SECRLENV_CONTEXT_SHA_ENV)
        self.actor_revision = _revision_env(_ACTOR_REVISION_ENV)
        self.actor_config_sha256 = _hash_env(_ACTOR_CONFIG_ENV)
        self.critic_revision = _revision_env(_CRITIC_REVISION_ENV)
        self.critic_config_sha256 = _hash_env(_CRITIC_CONFIG_ENV)
        self.max_fragment_bytes = _integer_env(_MAX_FRAGMENT_ENV)
        self.max_chunk_bytes = _integer_env(_MAX_CHUNK_ENV)
        if not 4 <= self.max_fragment_bytes <= 2 << 30 or self.max_fragment_bytes % 4 or not 4 <= self.max_chunk_bytes < 1 << 30 or self.max_chunk_bytes % 4:
            raise RuntimeError("SAO streaming layout transport bounds are invalid")
        from yeto.syncer_profile import SyncerSemanticProfile

        try:
            raw_profile = json.loads(_required_env(_SYNCER_PROFILE_ENV))
            self.syncer_profile = SyncerSemanticProfile.from_mapping(raw_profile)
            self.syncer_profile.validate_sao()
        except (TypeError, ValueError, json.JSONDecodeError) as error:
            raise RuntimeError(
                "SAO streaming layout probe syncer profile is invalid"
            ) from error
        self.evidence: dict[str, object] | None = None
        self.evidence_sha256: str | None = None

    async def initialize(self, *, actor_model, critic_model, rollout_manager) -> None:
        del rollout_manager
        if getattr(self.args, "debug_train_only", None) is not True or not getattr(self.args, "use_critic", False) or int(getattr(self.args, "lora_rank", 0)) > 0 or int(getattr(self.args, "start_rollout_id", 0)) != 0 or int(getattr(self.args, "num_critic_only_steps", 0)) != 0:
            raise RuntimeError("SAO streaming layout probe requires full-parameter lockstep actor/critic debug training from version zero")

        from yeto.rl.local_learner import ComponentIdentity
        from yeto.rl.miles_chunked_full_parameter import (
            MilesChunkedFullParameterAdapter,
        )
        from yeto.rl.sao_streaming_runtime import streaming_layout_attestation_schema

        actor_component = ComponentIdentity(
            role="actor",
            model_revision=self.actor_revision,
            config_hash=self.actor_config_sha256,
        )
        critic_component = ComponentIdentity(
            role="critic",
            model_revision=self.critic_revision,
            config_hash=self.critic_config_sha256,
        )
        actor_manifests, critic_manifests = await asyncio.gather(
            actor_model.full_parameter_shard_manifests(),
            critic_model.full_parameter_shard_manifests(),
        )

        def create_adapter(manifests, component, role, minimum_fragments):
            return MilesChunkedFullParameterAdapter.create(
                manifests,
                algorithm="sao",
                components=(component,),
                minimum_fragments=minimum_fragments,
                max_fragment_bytes=self.max_fragment_bytes,
                max_chunk_bytes=self.max_chunk_bytes,
                stream_role=role,
            )

        actor = create_adapter(
            actor_manifests,
            actor_component,
            "actor",
            1,
        )
        critic = create_adapter(
            critic_manifests,
            critic_component,
            "critic",
            1,
        )
        common_fragments = max(
            actor.layout.fragments.num_fragments,
            critic.layout.fragments.num_fragments,
        )
        actor = create_adapter(
            actor_manifests,
            actor_component,
            "actor",
            common_fragments,
        )
        critic = create_adapter(
            critic_manifests,
            critic_component,
            "critic",
            common_fragments,
        )
        if actor.layout.layout_hash == critic.layout.layout_hash or actor.layout.fragments.num_fragments != common_fragments or critic.layout.fragments.num_fragments != common_fragments:
            raise RuntimeError("SAO actor and critic did not derive distinct lockstep layouts")
        self.evidence = {
            "schema": streaming_layout_attestation_schema(),
            "sao_secrlenv_context_sha256": self.secrlenv_context_sha256,
            "settings": {
                "algorithm": "sao",
                "fragment_strategy": "owner_affine",
                "minimum_fragments": common_fragments,
                "max_fragment_bytes": self.max_fragment_bytes,
                "max_chunk_bytes": self.max_chunk_bytes,
                "wire_dtype": "fp32",
                "syncer_profile_sha256": self.syncer_profile.sha256,
            },
            "actor": _role_evidence(actor, actor_component),
            "critic": _role_evidence(critic, critic_component),
        }
        # Skip Miles' rollout loop after both groups and their metadata exist.
        self.args.num_rollout = self.args.start_rollout_id

    async def after_local_train(self, **_kwargs) -> bool:
        raise RuntimeError("metadata-only SAO streaming layout probe entered training")

    async def finalize(self) -> None:
        if self.evidence is None:
            raise RuntimeError("metadata-only SAO streaming layout probe did not complete")
        self.evidence_sha256 = _write_private_json(
            self.evidence_path,
            self.evidence,
        )


def create_sao_streaming_layout_probe(args) -> SAOStreamingLayoutProbe:
    return SAOStreamingLayoutProbe(args)
