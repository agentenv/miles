"""Metadata-only TP1 full-parameter layout attestation for the SAO smoke."""

from __future__ import annotations

import json
import os
import tempfile
from dataclasses import asdict
from pathlib import Path

_EVIDENCE_ENV = "YETO_SAO_TP1_LAYOUT_EVIDENCE"
_MODEL_REVISION_ENV = "YETO_SAO_MODEL_REVISION"
_CONFIG_HASH_ENV = "YETO_SAO_MODEL_CONFIG_SHA256"


def _write_private_json(path: Path, payload: dict[str, object]) -> None:
    if path.exists() or path.is_symlink() or not path.parent.is_dir():
        raise RuntimeError("TP1 layout evidence path is not fresh and safe")
    descriptor, raw = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
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


class SAOTP1LayoutProbe:
    """Construct the real actor and stop before generation or training."""

    supports_critic = False

    def __init__(self, args) -> None:
        self.args = args
        evidence = os.environ.get(_EVIDENCE_ENV)
        revision = os.environ.get(_MODEL_REVISION_ENV)
        config_hash = os.environ.get(_CONFIG_HASH_ENV)
        if not evidence or not revision or not config_hash:
            raise RuntimeError("TP1 layout probe environment is incomplete")
        self.evidence_path = Path(evidence)
        self.model_revision = revision
        self.model_config_sha256 = config_hash
        self.evidence: dict[str, object] | None = None

    async def initialize(self, *, actor_model, rollout_manager) -> None:
        del rollout_manager
        required = {
            "debug_train_only": True,
            "actor_num_nodes": 1,
            "actor_num_gpus_per_node": 1,
            "tensor_model_parallel_size": 1,
            "pipeline_model_parallel_size": 1,
            "context_parallel_size": 1,
            "expert_model_parallel_size": 1,
        }
        for name, expected in required.items():
            if getattr(self.args, name, None) != expected:
                raise RuntimeError(f"TP1 layout probe requires {name}={expected!r}")
        if getattr(self.args, "use_critic", False) or getattr(self.args, "lora_rank", 0) > 0:
            raise RuntimeError("TP1 layout probe requires a full-parameter actor without a critic")

        manifests = tuple(await actor_model.full_parameter_shard_manifests())
        if len(manifests) != 1:
            raise RuntimeError("TP1 layout probe did not observe exactly one owner")
        topology = manifests[0].topology
        if any(
            getattr(topology, f"{axis}_{field}") != expected
            for axis in ("tp", "pp", "ep", "cp", "dp")
            for field, expected in (("rank", 0), ("size", 1))
        ):
            raise RuntimeError("TP1 layout probe observed a non-TP1 topology")

        manifest = manifests[0]
        self.evidence = {
            "schema": "miles-sao-tp1-layout-v1",
            # Centralized Miles publishes directly to SGLang and has no DiLoCo
            # fragment layout. Bind the run to Megatron's exact topology-owned
            # FP32 master manifest rather than inventing a syncer layout.
            "parameter_layout_hash": manifest.layout_hash,
            "parameter_tensor_count": len(manifest.specs),
            "parameter_scalar_count": sum(spec.numel for spec in manifest.specs),
            "model_revision": self.model_revision,
            "model_config_sha256": self.model_config_sha256,
            "topology": asdict(topology),
        }
        # Skip Miles' rollout loop after actor construction and metadata export.
        self.args.num_rollout = self.args.start_rollout_id

    async def after_local_train(self, **_kwargs) -> bool:
        raise RuntimeError("metadata-only TP1 layout probe entered the train loop")

    async def finalize(self) -> None:
        if self.evidence is None:
            raise RuntimeError("metadata-only TP1 layout probe did not complete")
        _write_private_json(self.evidence_path, self.evidence)


def create_sao_tp1_layout_probe(args) -> SAOTP1LayoutProbe:
    return SAOTP1LayoutProbe(args)
