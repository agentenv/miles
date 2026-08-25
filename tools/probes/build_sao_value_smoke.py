"""Build a tiny tokenizer-compatible dataset for the SAO value-path smoke."""

import argparse
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer

from miles.value_pretraining import VALUE_PRETRAIN_SCHEMA, ValuePretrainDataset, load_value_pretrain_manifest


def _write_json(path: Path, value) -> None:
    path.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--num-samples", type=int, default=16)
    args = parser.parse_args()
    if args.num_samples < 4:
        raise ValueError("--num-samples must be at least 4")

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, trust_remote_code=True)

    rows = []
    for index in range(args.num_samples):
        reward = float(index % 2)
        prompt = f"Classify deterministic training case {index}. Outcome marker: {int(reward)}.\nAnswer:"
        response = " success" if reward else " failure"
        prompt_tokens = tokenizer.encode(prompt, add_special_tokens=False)
        response_tokens = tokenizer.encode(response, add_special_tokens=False)
        if not prompt_tokens or not response_tokens:
            raise RuntimeError("tokenizer produced an empty prompt or response")
        rows.append(
            {
                "sample_id": f"sao-value-smoke-{index:04d}",
                "tokens": prompt_tokens + response_tokens,
                "response_length": len(response_tokens),
                "returns": [reward] * len(response_tokens),
                "loss_mask": [1] * len(response_tokens),
            }
        )

    dataset_path = output_dir / "train.jsonl"
    with dataset_path.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, sort_keys=True, separators=(",", ":")) + "\n")
    dataset_sha256 = hashlib.sha256(dataset_path.read_bytes()).hexdigest()
    manifest_path = output_dir / "manifest.json"
    _write_json(
        manifest_path,
        {
            "schema": VALUE_PRETRAIN_SCHEMA,
            "train": {
                "path": dataset_path.name,
                "sha256": dataset_sha256,
                "num_samples": len(rows),
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
    manifest = load_value_pretrain_manifest(manifest_path)
    ValuePretrainDataset(manifest.train, manifest.objective)
    print(json.dumps({"manifest": str(manifest_path), "manifest_sha256": manifest.sha256}, sort_keys=True))


if __name__ == "__main__":
    main()
