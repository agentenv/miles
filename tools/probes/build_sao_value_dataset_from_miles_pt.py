"""Convert native Miles rollout shards into a strict SAO value dataset.

This utility is intentionally narrow: it selects short, balanced, real
rollouts from existing ``data_*.pt`` shards and writes the dependency-free
JSONL/manifest contract consumed by ``train_value.py``.  Long trajectories are
windowed contiguously while preserving a non-empty prefix and at least one
active response target.  Source and target tokenizer vocabularies are compared
by token ID before any artifact is published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


VALUE_PRETRAIN_SCHEMA = "miles.value-pretrain.v1"
SHARD_RE = re.compile(r"data_(\d+)\.pt$")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _shard_number(path: Path) -> int:
    match = SHARD_RE.search(path.name)
    if match is None:
        raise ValueError(f"unexpected rollout shard name: {path}")
    return int(match.group(1))


def _token_id_map(path: Path) -> dict[int, str]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    model = raw.get("model")
    vocab = model.get("vocab") if isinstance(model, dict) else None
    if not isinstance(vocab, dict):
        raise ValueError(f"{path}: tokenizer model.vocab must be an object")

    result: dict[int, str] = {}
    for token, token_id in vocab.items():
        if not isinstance(token, str) or isinstance(token_id, bool) or not isinstance(token_id, int):
            raise ValueError(f"{path}: malformed tokenizer vocabulary entry")
        previous = result.setdefault(token_id, token)
        if previous != token:
            raise ValueError(f"{path}: duplicate tokenizer ID {token_id}")
    for entry in raw.get("added_tokens", []):
        if not isinstance(entry, dict):
            raise ValueError(f"{path}: added_tokens entries must be objects")
        token_id = entry.get("id")
        content = entry.get("content")
        if isinstance(token_id, bool) or not isinstance(token_id, int) or not isinstance(content, str):
            raise ValueError(f"{path}: malformed added token")
        previous = result.setdefault(token_id, content)
        if previous != content:
            raise ValueError(f"{path}: tokenizer ID {token_id} has conflicting spellings")
    return result


def _model_vocab_size(path: Path) -> int:
    raw = json.loads(path.read_text(encoding="utf-8"))
    candidate = raw.get("vocab_size")
    if candidate is None and isinstance(raw.get("text_config"), dict):
        candidate = raw["text_config"].get("vocab_size")
    if isinstance(candidate, bool) or not isinstance(candidate, int) or candidate < 1:
        raise ValueError(f"{path}: could not resolve a positive model vocab_size")
    return candidate


@dataclass(frozen=True)
class Candidate:
    split: str
    shard: Path
    shard_index: int
    ordinal: int
    source_index: int
    task_id: str
    attempt_id: int
    reward: int
    tokens: tuple[int, ...]
    response_length: int
    loss_mask: tuple[int, ...]
    original_num_tokens: int
    original_response_length: int
    response_window_start: int
    response_window_end: int
    source_max_token_id: int

    @property
    def sample_id(self) -> str:
        return (
            f"secrlenv:{self.task_id}:attempt-{self.attempt_id}:"
            f"source-{self.source_index}:shard-{self.shard_index}-row-{self.ordinal}"
        )


def _as_binary_reward(value: object, *, location: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{location}: reward must be numeric")
    reward = float(value)
    if not math.isfinite(reward) or reward not in (0.0, 1.0):
        raise ValueError(f"{location}: expected a binary reward, got {value!r}")
    return int(reward)


def _window_sample(
    *,
    tokens: tuple[int, ...],
    response_length: int,
    loss_mask: tuple[int, ...],
    max_seq_len: int,
    location: str,
) -> tuple[tuple[int, ...], int, tuple[int, ...], int, int]:
    if max_seq_len < 2:
        raise ValueError("max_seq_len must leave room for prompt and response tokens")
    boundary = len(tokens) - response_length
    response_capacity = max_seq_len - 1
    if response_length <= response_capacity:
        response_start = 0
        response_end = response_length
    else:
        response_end = response_length
        response_start = response_end - response_capacity
        if not any(loss_mask[response_start:response_end]):
            last_active = max(index for index, value in enumerate(loss_mask) if value)
            response_end = last_active + 1
            response_start = max(0, response_end - response_capacity)

    new_boundary = boundary + response_start
    absolute_end = boundary + response_end
    absolute_start = max(0, absolute_end - max_seq_len)
    if absolute_start >= new_boundary:
        absolute_start = new_boundary - 1
    if absolute_start < 0:
        raise ValueError(f"{location}: could not preserve a prompt token")

    windowed_tokens = tokens[absolute_start:absolute_end]
    windowed_mask = loss_mask[response_start:response_end]
    windowed_response_length = len(windowed_mask)
    if not windowed_tokens or windowed_response_length >= len(windowed_tokens):
        raise ValueError(f"{location}: window must preserve at least one prompt token")
    if not any(windowed_mask):
        raise ValueError(f"{location}: window must preserve at least one active response target")
    return (
        windowed_tokens,
        windowed_response_length,
        windowed_mask,
        response_start,
        response_end,
    )


def _candidate(
    raw: object,
    *,
    split: str,
    shard: Path,
    shard_index: int,
    ordinal: int,
    max_seq_len: int,
) -> Candidate:
    location = f"{shard}:{ordinal}"
    if not isinstance(raw, dict):
        raise ValueError(f"{location}: sample must be an object")

    tokens_raw = raw.get("tokens")
    if not isinstance(tokens_raw, list) or len(tokens_raw) < 2:
        raise ValueError(f"{location}: tokens must contain at least two IDs")
    tokens = tuple(tokens_raw)
    if any(isinstance(token, bool) or not isinstance(token, int) or token < 0 for token in tokens):
        raise ValueError(f"{location}: tokens must be non-negative integers")

    response_length = raw.get("response_length")
    if (
        isinstance(response_length, bool)
        or not isinstance(response_length, int)
        or response_length < 1
        or response_length >= len(tokens)
    ):
        raise ValueError(f"{location}: invalid response_length")

    loss_mask_raw = raw.get("loss_mask")
    if not isinstance(loss_mask_raw, list) or len(loss_mask_raw) != response_length:
        raise ValueError(f"{location}: loss_mask length must equal response_length")
    if any(value not in (0, 1, False, True) for value in loss_mask_raw):
        raise ValueError(f"{location}: loss_mask must be binary")
    loss_mask = tuple(int(value) for value in loss_mask_raw)
    if not any(loss_mask):
        raise ValueError(f"{location}: loss_mask selects no value targets")

    metadata = raw.get("metadata")
    if not isinstance(metadata, dict):
        raise ValueError(f"{location}: metadata must be an object")
    task_id = metadata.get("task_id") or raw.get("label")
    attempt_id = metadata.get("attempt_id", 0)
    source_index = raw.get("index")
    if not isinstance(task_id, str) or not task_id:
        raise ValueError(f"{location}: task_id is missing")
    if isinstance(attempt_id, bool) or not isinstance(attempt_id, int) or attempt_id < 0:
        raise ValueError(f"{location}: attempt_id must be a non-negative integer")
    if isinstance(source_index, bool) or not isinstance(source_index, int) or source_index < 0:
        raise ValueError(f"{location}: source index must be a non-negative integer")

    source_max_token_id = max(tokens)
    original_num_tokens = len(tokens)
    original_response_length = response_length
    tokens, response_length, loss_mask, response_window_start, response_window_end = _window_sample(
        tokens=tokens,
        response_length=response_length,
        loss_mask=loss_mask,
        max_seq_len=max_seq_len,
        location=location,
    )

    return Candidate(
        split=split,
        shard=shard,
        shard_index=shard_index,
        ordinal=ordinal,
        source_index=source_index,
        task_id=task_id,
        attempt_id=attempt_id,
        reward=_as_binary_reward(raw.get("reward"), location=location),
        tokens=tokens,
        response_length=response_length,
        loss_mask=loss_mask,
        original_num_tokens=original_num_tokens,
        original_response_length=original_response_length,
        response_window_start=response_window_start,
        response_window_end=response_window_end,
        source_max_token_id=source_max_token_id,
    )


def _load_candidates(source_root: Path, split: str, *, max_seq_len: int) -> tuple[list[Candidate], dict[str, object]]:
    # Torch is kept out of the module import path so --help and static checks do
    # not require a training environment.
    import torch

    shard_paths = sorted((source_root / split).glob("data_*.pt"), key=_shard_number)
    if not shard_paths:
        raise FileNotFoundError(f"no data_*.pt shards found in {source_root / split}")

    candidates: list[Candidate] = []
    rewards: Counter[int] = Counter()
    total_samples = 0
    corpus_max_token_id = -1
    truncated_samples = 0
    active_shifted_samples = 0
    for shard in shard_paths:
        shard_index = _shard_number(shard)
        payload = torch.load(shard, map_location="cpu", weights_only=False)
        if not isinstance(payload, dict) or not isinstance(payload.get("samples"), list):
            raise ValueError(f"{shard}: expected an object containing a samples list")
        for ordinal, raw in enumerate(payload["samples"]):
            item = _candidate(
                raw,
                split=split,
                shard=shard,
                shard_index=shard_index,
                ordinal=ordinal,
                max_seq_len=max_seq_len,
            )
            total_samples += 1
            rewards[item.reward] += 1
            corpus_max_token_id = max(corpus_max_token_id, item.source_max_token_id)
            truncated_samples += int(item.original_num_tokens != len(item.tokens))
            active_shifted_samples += int(
                item.response_window_end != item.original_response_length
            )
            candidates.append(item)

    return candidates, {
        "shards": len(shard_paths),
        "source_samples": total_samples,
        "source_rewards": dict(sorted(rewards.items())),
        "eligible_samples": len(candidates),
        "eligible_rewards": dict(sorted(rewards.items())),
        "windowed_samples": truncated_samples,
        "windows_shifted_from_tail_to_preserve_active_target": active_shifted_samples,
        "corpus_max_token_id": corpus_max_token_id,
    }


def _select_balanced(candidates: Iterable[Candidate], *, per_reward: int, split: str) -> list[Candidate]:
    selected: list[Candidate] = []
    selected_tasks: set[str] = set()
    by_reward = {
        reward: sorted(
            (item for item in candidates if item.reward == reward),
            key=lambda item: (
                item.original_num_tokens,
                item.task_id,
                item.attempt_id,
                item.shard_index,
                item.ordinal,
            ),
        )
        for reward in (0, 1)
    }
    for reward in (0, 1):
        for item in by_reward[reward]:
            if item.task_id in selected_tasks:
                continue
            selected.append(item)
            selected_tasks.add(item.task_id)
            if sum(candidate.reward == reward for candidate in selected) == per_reward:
                break
        actual = sum(candidate.reward == reward for candidate in selected)
        if actual != per_reward:
            raise ValueError(
                f"{split}: only {actual} unique-task reward={reward} samples fit the sequence limit; "
                f"need {per_reward}"
            )
    return sorted(selected, key=lambda item: (item.reward, len(item.tokens), item.sample_id))


def _row(item: Candidate) -> dict[str, object]:
    return {
        "sample_id": item.sample_id,
        "tokens": list(item.tokens),
        "response_length": item.response_length,
        "returns": [float(item.reward)] * item.response_length,
        "loss_mask": list(item.loss_mask),
    }


def _write_jsonl(path: Path, selected: Iterable[Candidate]) -> int:
    count = 0
    with path.open("w", encoding="utf-8") as stream:
        for item in selected:
            stream.write(json.dumps(_row(item), ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n")
            count += 1
    return count


def _selection_stats(selected: list[Candidate]) -> dict[str, object]:
    lengths = sorted(len(item.tokens) for item in selected)
    original_lengths = sorted(item.original_num_tokens for item in selected)
    active = sorted(sum(item.loss_mask) for item in selected)
    return {
        "samples": len(selected),
        "tasks": len({item.task_id for item in selected}),
        "rewards": dict(sorted(Counter(item.reward for item in selected).items())),
        "tokens": {
            "min": min(lengths),
            "median": lengths[len(lengths) // 2],
            "max": max(lengths),
            "sum": sum(lengths),
        },
        "source_tokens": {
            "min": min(original_lengths),
            "median": original_lengths[len(original_lengths) // 2],
            "max": max(original_lengths),
            "sum": sum(original_lengths),
        },
        "windowed_samples": sum(item.original_num_tokens != len(item.tokens) for item in selected),
        "windows_shifted_from_tail_to_preserve_active_target": sum(
            item.response_window_end != item.original_response_length for item in selected
        ),
        "active_tokens": {
            "min": min(active),
            "median": active[len(active) // 2],
            "max": max(active),
            "sum": sum(active),
        },
        "sample_ids": [item.sample_id for item in selected],
    }


def _validate_tokenizers(
    *,
    source_tokenizer_json: Path,
    target_tokenizer_json: Path,
    target_model_config: Path,
    corpus_max_token_id: int,
) -> dict[str, object]:
    source = _token_id_map(source_tokenizer_json)
    target = _token_id_map(target_tokenizer_json)
    target_model_vocab_size = _model_vocab_size(target_model_config)
    mismatches = [
        token_id
        for token_id in range(corpus_max_token_id + 1)
        if source.get(token_id) != target.get(token_id)
    ]
    if mismatches:
        token_id = mismatches[0]
        raise ValueError(
            "source and target tokenizers disagree within the source corpus token range: "
            f"id={token_id}, source={source.get(token_id)!r}, target={target.get(token_id)!r}"
        )
    if corpus_max_token_id >= target_model_vocab_size:
        raise ValueError(
            f"source corpus token ID {corpus_max_token_id} exceeds target model vocab_size "
            f"{target_model_vocab_size}"
        )
    return {
        "source_tokenizer_json": str(source_tokenizer_json),
        "source_tokenizer_sha256": _sha256(source_tokenizer_json),
        "source_tokenizer_ids": len(source),
        "target_tokenizer_json": str(target_tokenizer_json),
        "target_tokenizer_sha256": _sha256(target_tokenizer_json),
        "target_tokenizer_ids": len(target),
        "target_model_config": str(target_model_config),
        "target_model_config_sha256": _sha256(target_model_config),
        "target_model_vocab_size": target_model_vocab_size,
        "source_corpus_max_token_id": corpus_max_token_id,
        "semantic_id_mapping_equal_through": corpus_max_token_id,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-tokenizer-json", type=Path, required=True)
    parser.add_argument("--target-tokenizer-json", type=Path, required=True)
    parser.add_argument("--target-model-config", type=Path, required=True)
    parser.add_argument("--target-model", default="Qwen/Qwen3.5-0.8B")
    parser.add_argument("--target-revision", required=True)
    parser.add_argument("--max-seq-len", type=int, default=8192)
    parser.add_argument("--train-per-reward", type=int, default=16)
    parser.add_argument("--heldout-per-reward", type=int, default=4)
    args = parser.parse_args()

    for field in ("max_seq_len", "train_per_reward", "heldout_per_reward"):
        value = getattr(args, field)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError(f"--{field.replace('_', '-')} must be a positive integer")

    output_files = [args.output_dir / name for name in ("train.jsonl", "heldout.jsonl", "manifest.json", "report.json")]
    existing = [path for path in output_files if path.exists()]
    if existing:
        raise FileExistsError(f"refusing to overwrite existing output: {existing[0]}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_candidates, train_source_stats = _load_candidates(
        args.source_root, "train", max_seq_len=args.max_seq_len
    )
    heldout_candidates, heldout_source_stats = _load_candidates(
        args.source_root, "validation", max_seq_len=args.max_seq_len
    )
    train = _select_balanced(train_candidates, per_reward=args.train_per_reward, split="train")
    heldout = _select_balanced(
        heldout_candidates, per_reward=args.heldout_per_reward, split="heldout"
    )
    overlapping_tasks = {item.task_id for item in train} & {item.task_id for item in heldout}
    if overlapping_tasks:
        raise ValueError(
            "heldout tasks overlap training tasks: " + ", ".join(sorted(overlapping_tasks))
        )

    corpus_max_token_id = max(
        int(train_source_stats["corpus_max_token_id"]),
        int(heldout_source_stats["corpus_max_token_id"]),
    )
    tokenizer_report = _validate_tokenizers(
        source_tokenizer_json=args.source_tokenizer_json,
        target_tokenizer_json=args.target_tokenizer_json,
        target_model_config=args.target_model_config,
        corpus_max_token_id=corpus_max_token_id,
    )

    train_path = args.output_dir / "train.jsonl"
    heldout_path = args.output_dir / "heldout.jsonl"
    _write_jsonl(train_path, train)
    _write_jsonl(heldout_path, heldout)
    train_sha256 = _sha256(train_path)
    heldout_sha256 = _sha256(heldout_path)

    manifest_path = args.output_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema": VALUE_PRETRAIN_SCHEMA,
            "train": {
                "path": train_path.name,
                "sha256": train_sha256,
                "num_samples": len(train),
            },
            "heldout": {
                "path": heldout_path.name,
                "sha256": heldout_sha256,
                "num_samples": len(heldout),
            },
            "objective": {
                "loss_type": "classification",
                "num_bins": 51,
                "reward_range": [0.0, 1.0],
                "target_type": "hl_gauss",
                "hl_gauss_sigma_ratio": 0.75,
            },
        },
    )

    selected_shards = sorted({item.shard for item in train + heldout})
    source_report = args.source_root / "report.json"
    report_path = args.output_dir / "report.json"
    _write_json(
        report_path,
        {
            "schema": "miles.value-pretrain-conversion-report.v1",
            "source": {
                "kind": "native_miles_rollout_pt",
                "root": str(args.source_root),
                "report_sha256": _sha256(source_report) if source_report.is_file() else None,
                "selected_shard_sha256": {
                    str(path.relative_to(args.source_root)): _sha256(path) for path in selected_shards
                },
            },
            "target": {
                "model": args.target_model,
                "revision": args.target_revision,
                "tokenizer_compatibility": tokenizer_report,
            },
            "selection": {
                "policy": "shortest_binary_reward_balanced_unique_task_contiguous_suffix_window",
                "max_seq_len": args.max_seq_len,
                "train_per_reward": args.train_per_reward,
                "heldout_per_reward": args.heldout_per_reward,
                "train_source": train_source_stats,
                "heldout_source": heldout_source_stats,
                "train": _selection_stats(train),
                "heldout": _selection_stats(heldout),
            },
            "artifacts": {
                "train_jsonl_sha256": train_sha256,
                "heldout_jsonl_sha256": heldout_sha256,
                "manifest_sha256": _sha256(manifest_path),
            },
        },
    )
    print(
        json.dumps(
            {
                "manifest": str(manifest_path),
                "manifest_sha256": _sha256(manifest_path),
                "report": str(report_path),
                "report_sha256": _sha256(report_path),
                "train_samples": len(train),
                "heldout_samples": len(heldout),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
