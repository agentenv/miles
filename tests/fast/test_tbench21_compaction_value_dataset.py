from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path

import pytest


REPOSITORY = Path(__file__).resolve().parents[2]
YETO_ROOT = REPOSITORY.parent / "yeto-grpo-diloco-connector"
if YETO_ROOT.is_dir():
    sys.path.insert(0, str(YETO_ROOT))

from yeto.rl import tbench_outcome  # noqa: E402


TOOL = REPOSITORY / "tools" / "probes" / "build_tbench21_compaction_value_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_tbench21_compaction_value_dataset", TOOL)
assert SPEC is not None and SPEC.loader is not None
converter = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(converter)

KEY = b"authenticated-terminal-bench-test-key-000000000000"
TASK_IDS = tuple(f"task-{index:03d}" for index in range(89))
SAMPLE_IDS = tuple(f"baseline:{task_id}:r{replica}" for task_id in TASK_IDS for replica in range(4))
EXPECTED_ISLANDS = {f"baseline:{task_id}:r{replica}": (task_index * 4 + replica) % 8 for task_index, task_id in enumerate(TASK_IDS) for replica in range(4)}


@pytest.fixture(autouse=True)
def _clean_hmac_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv(converter.HMAC_ENV, raising=False)
    monkeypatch.delenv(converter.HMAC_FILE_ENV, raising=False)


@pytest.fixture
def hmac_key_file(tmp_path: Path) -> Path:
    path = tmp_path / "tbench-hmac.key"
    path.write_bytes(KEY)
    path.chmod(0o600)
    return path


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _selection(sample_ids: list[str] | tuple[str, ...]) -> dict[str, object]:
    return {
        "schema": "yeto.sao-value-pretraining-selection.v1",
        "source_phase": "baseline",
        "selection": "all",
        "sample_count": 356,
        "task_count": 89,
        "rollouts_per_task": 4,
        "baseline_sample_ids": list(sample_ids),
        "critic_pretraining_exposes_actor_eval_tasks": True,
        "actor_eval_task_count_exposed_to_critic": 45,
        "actor_eval_trajectory_count_exposed_to_critic": 180,
    }


def _plan_tree(tmp_path: Path) -> tuple[Path, Path]:
    root = tmp_path / "plan"
    root.mkdir()
    selection = root / "value-pretraining-selection.json"
    selection.write_text(json.dumps(_selection(SAMPLE_IDS)), encoding="utf-8")
    selection_sha = _sha256(selection)
    manifest = {
        "schema": "yeto.tbench21-sao-diloco-plan.v1",
        "terminal_bench": {
            "version": "2.1",
            "task_count": 89,
            "task_contracts": {task_id: {} for task_id in TASK_IDS},
        },
        "rollouts": {
            "per_task": 4,
            "baseline": 356,
            "episode_timeout_seconds": 1800,
            "seed_base": 82621,
            "per_island_seeds": list(range(82621, 82629)),
        },
        "value_pretraining": {
            "selection_path": selection.name,
            "selection_sha256": selection_sha,
            "uses_all_baseline_trajectories": True,
            "sample_count": 356,
        },
        "compaction": {
            "enabled": True,
            "trainer_objective": "sao",
            "max_seq_len": 8192,
            "trigger_tokens": 6144,
            "summary_max_tokens": 1024,
            "max_compactions_per_episode": 3,
        },
        "topology": {
            "islands": 8,
            "one_physical_gpu_per_island": True,
            "model": "Qwen/Qwen3.5-0.8B",
            "model_revision": "2fc06364715b967f1860aea9cf38778875588b17",
        },
        "files": {selection.name: selection_sha},
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    return manifest_path, selection


def _segment(
    sample_id: str,
    segment_index: int,
    segment_type: str,
    *,
    reward: float = 1.0,
) -> dict[str, object]:
    return {
        "sample_id": sample_id,
        "task_id": sample_id.split(":")[1],
        "trajectory_id": f"trajectory:{sample_id}",
        "context_window": segment_index // 2,
        "segment_index": segment_index,
        "segment_type": segment_type,
        "context_budget": 8192,
        "island_id": EXPECTED_ISLANDS.get(sample_id, 0),
        "tokens": [1, 2, 3],
        "response_length": 2,
        "loss_mask": [1, 1],
        "reward": reward,
        "outcome_status": "completed",
        "outcome_episode_id": f"episode:{sample_id}",
        "outcome_mac": f"mac:{sample_id}",
        "sample_status": "completed",
        "source": "source.pt",
        "source_ordinal": segment_index,
    }


def _native_segment(
    sample_id: str,
    segment_index: int,
    segment_type: str,
    *,
    reward: float = 1.0,
    signed_task_id: str | None = None,
    signed_sample_id: str | None = None,
    signed_reward: float | None = None,
    signed_status: str = "completed",
) -> dict[str, object]:
    task_id = sample_id.split(":")[1]
    island_id = EXPECTED_ISLANDS[sample_id]
    signed_reward = reward if signed_reward is None else signed_reward
    verifier = tbench_outcome.TIMEOUT_VERIFIER if signed_status == "timeout" else tbench_outcome.TEST_SH_VERIFIER
    signed = tbench_outcome.build_signed_metadata(
        task_id=signed_task_id or task_id,
        sample_id=signed_sample_id or sample_id,
        episode_id=f"episode:{sample_id}",
        status=signed_status,
        reward=signed_reward,
        verifier=verifier,
        testsh_rc=None if signed_status == "timeout" else int(signed_reward == 0.0),
        key=KEY,
    )
    return {
        "metadata": {
            "sample_id": sample_id,
            "task_id": task_id,
            "compaction_trajectory_id": f"trajectory:{sample_id}",
            "compaction_schema_version": 1,
            "compaction_context_window": segment_index // 2,
            "compaction_segment_index": segment_index,
            "compaction_segment_type": segment_type,
            "compaction_context_budget": 8192,
            "island_id": island_id,
            "rollout_seed": 82621 + island_id,
            "split": "baseline",
            "rollout_replica": int(sample_id.rsplit(":r", 1)[1]),
            "episode_timeout_seconds": 1800,
            "max_seq_len": 8192,
            "reward": reward,
            "exit_status": "completed",
            **signed,
        },
        "tokens": [1, 2, 3],
        "response_length": 2,
        "loss_mask": [1, 1],
        "reward": reward,
        "status": "completed",
    }


def _source_tree(tmp_path: Path, *, compact_first: bool = True) -> list[Path]:
    torch = pytest.importorskip("torch")
    roots: list[Path] = []
    for island_id in range(8):
        root = tmp_path / f"island-{island_id}"
        rollout = root / "rollout_data"
        rollout.mkdir(parents=True)
        samples = [_native_segment(sample_id, 0, "execution", reward=0.0) for sample_id in SAMPLE_IDS if EXPECTED_ISLANDS[sample_id] == island_id]
        if island_id == 0 and compact_first:
            first = samples.pop(0)
            sample_id = first["metadata"]["sample_id"]
            samples[:0] = [
                first,
                _native_segment(sample_id, 1, "summary", reward=0.0),
                _native_segment(sample_id, 2, "execution", reward=0.0),
            ]
        torch.save(
            {"rollout_id": 0, "metadata": {}, "samples": samples},
            rollout / "0.pt",
        )
        roots.append(root)
    return roots


def _parse_native(raw: dict[str, object], hmac_key_file: Path) -> dict[str, object]:
    island_id = int(raw["metadata"]["island_id"])
    with converter._authenticated_outcome_source(hmac_key_file) as api:
        return converter._segment(
            raw,
            source=Path("/source/0.pt"),
            ordinal=0,
            max_seq_len=8192,
            outcome_api=api,
            island_id=island_id,
            expected_islands=EXPECTED_ISLANDS,
        )


def test_requires_exact_plan_bound_all_trajectory_selection(tmp_path: Path) -> None:
    manifest, selection = _plan_tree(tmp_path)
    sample_ids = converter._load_selection(selection)
    expected_islands = converter._load_plan_manifest(manifest, selection_path=selection, selection_ids=sample_ids)
    assert sample_ids == SAMPLE_IDS
    assert expected_islands == EXPECTED_ISLANDS

    selection.write_text(json.dumps(_selection(SAMPLE_IDS), indent=2))
    with pytest.raises(converter.ConversionError, match="selection SHA/path"):
        converter._load_plan_manifest(
            manifest,
            selection_path=selection,
            selection_ids=converter._load_selection(selection),
        )


def test_orders_complete_compaction_trajectories_by_plan() -> None:
    expected = ("baseline:task-b:r0", "baseline:task-a:r0")
    segments = [
        _segment(expected[0], 2, "execution"),
        _segment(expected[1], 0, "execution", reward=0.0),
        _segment(expected[0], 0, "execution"),
        _segment(expected[0], 1, "summary"),
    ]
    ordered = converter._validate_and_order(segments, expected)
    assert [row["sample_id"] for row in ordered] == [expected[0]] * 3 + [expected[1]]
    assert [row["segment_index"] for row in ordered] == [0, 1, 2, 0]


@pytest.mark.parametrize(
    "segments, message",
    [
        (
            [
                _segment("baseline:task-a:r0", 0, "execution"),
                _segment("baseline:task-a:r0", 1, "summary"),
            ],
            "segment sequence is invalid",
        ),
        (
            [
                _segment("baseline:task-a:r0", 0, "execution"),
                _segment("baseline:task-a:r0", 2, "execution"),
            ],
            "segment indices are not contiguous",
        ),
        (
            [
                _segment(
                    "baseline:task-a:r0",
                    index,
                    "execution" if index % 2 == 0 else "summary",
                )
                for index in range(9)
            ],
            "exceeds three compactions",
        ),
    ],
)
def test_rejects_invalid_compaction_trajectory(segments: list[dict[str, object]], message: str) -> None:
    with pytest.raises(converter.ConversionError, match=message):
        converter._validate_and_order(segments, ("baseline:task-a:r0",))


def test_build_authenticates_and_consumes_every_segment_exactly_once(tmp_path: Path, hmac_key_file: Path) -> None:
    manifest, selection = _plan_tree(tmp_path)
    sources = _source_tree(tmp_path)
    manifest_path = converter.build(
        source_roots=sources,
        selection_path=selection,
        plan_manifest_path=manifest,
        hmac_key_file=hmac_key_file,
        output_dir=tmp_path / "value-data",
        model="Qwen/Qwen3.5-0.8B",
        revision="2fc06364715b967f1860aea9cf38778875588b17",
        max_seq_len=8192,
        vocab_size=248320,
    )
    value_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads((manifest_path.parent / "report.json").read_text())
    assert value_manifest["train"]["num_samples"] == 358
    assert value_manifest["provenance"]["included_segments"] == 358
    assert value_manifest["provenance"]["all_terminal_bench_outcomes_authenticated"] is True
    assert value_manifest["provenance"]["authenticated_outcomes_bound_to_rows"] is True
    assert report["planned_trajectories"] == 356
    assert report["outcome_statuses"] == {"completed": 358}
    assert report["segment_types"] == {"execution": 357, "summary": 1}


def test_rejects_missing_or_tampered_signed_outcome(
    hmac_key_file: Path,
) -> None:
    raw = _native_segment(SAMPLE_IDS[0], 0, "execution")
    raw["metadata"].pop(tbench_outcome.MAC_KEY)
    with pytest.raises(converter.ConversionError, match="no valid authenticated"):
        _parse_native(raw, hmac_key_file)

    raw = _native_segment(SAMPLE_IDS[0], 0, "execution")
    raw["metadata"][tbench_outcome.OUTCOME_KEY]["reward"] = 0.0
    with pytest.raises(converter.ConversionError, match="no valid authenticated"):
        _parse_native(raw, hmac_key_file)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"signed_task_id": "other-task"}, "signed task_id"),
        ({"signed_sample_id": "baseline:other-task:r0"}, "signed sample_id"),
        ({"signed_reward": 0.0}, "signed reward differs from row metadata"),
        ({"signed_status": "max_turns"}, "signed status"),
    ],
)
def test_rejects_authentic_outcome_bound_to_different_row(hmac_key_file: Path, overrides: dict[str, object], message: str) -> None:
    raw = _native_segment(SAMPLE_IDS[0], 0, "execution", **overrides)
    with pytest.raises(converter.ConversionError, match=message):
        _parse_native(raw, hmac_key_file)


def test_rejects_wrong_context_budget_and_top_level_reward(
    hmac_key_file: Path,
) -> None:
    raw = _native_segment(SAMPLE_IDS[0], 0, "execution")
    raw["metadata"]["compaction_context_budget"] = 4096
    with pytest.raises(converter.ConversionError, match="context budget"):
        _parse_native(raw, hmac_key_file)

    raw = _native_segment(SAMPLE_IDS[0], 0, "execution")
    raw["reward"] = 0.0
    with pytest.raises(converter.ConversionError, match="top-level sample reward"):
        _parse_native(raw, hmac_key_file)


def test_hmac_key_must_be_explicit_private_file_only(tmp_path: Path, hmac_key_file: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(converter.ConversionError, match="unreadable"):
        converter._validate_hmac_key_file(tmp_path / "missing")

    hmac_key_file.chmod(0o644)
    with pytest.raises(converter.ConversionError, match="mode 0400 or 0600"):
        converter._validate_hmac_key_file(hmac_key_file)

    hmac_key_file.chmod(0o600)
    monkeypatch.setenv(converter.HMAC_ENV, "x" * 48)
    with pytest.raises(converter.ConversionError, match="must be absent"):
        with converter._authenticated_outcome_source(hmac_key_file):
            pass


def test_source_contract_requires_eight_exact_single_shard_roots(
    tmp_path: Path,
) -> None:
    roots = []
    for island_id in range(8):
        root = tmp_path / f"island-{island_id}"
        (root / "rollout_data").mkdir(parents=True)
        (root / "rollout_data" / "0.pt").write_bytes(b"fixture")
        roots.append(root)
    assert len(converter._source_files(roots)) == 8
    with pytest.raises(converter.ConversionError, match="exactly eight"):
        converter._source_files(roots[:-1])
    (roots[0] / "rollout_data" / "1.pt").write_bytes(b"extra")
    with pytest.raises(converter.ConversionError, match="exactly rollout_data/0.pt"):
        converter._source_files(roots)
