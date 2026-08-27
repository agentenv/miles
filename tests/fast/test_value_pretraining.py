import hashlib
import json
import tempfile
import unittest
from pathlib import Path

from miles.value_pretraining import (
    VALUE_PRETRAIN_SCHEMA,
    ValuePretrainDataset,
    load_value_pretrain_contract,
    load_value_pretrain_manifest,
    value_pretrain_num_steps,
    write_value_pretrain_contract,
)


def _jsonl_bytes(rows):
    return b"".join((json.dumps(row, sort_keys=True) + "\n").encode() for row in rows)


class ValuePretrainingContractTest(unittest.TestCase):
    def _write_fixture(self, root: Path, rows, *, objective=None):
        dataset_path = root / "train.jsonl"
        dataset_bytes = _jsonl_bytes(rows)
        dataset_path.write_bytes(dataset_bytes)
        manifest = {
            "schema": VALUE_PRETRAIN_SCHEMA,
            "train": {
                "path": dataset_path.name,
                "sha256": hashlib.sha256(dataset_bytes).hexdigest(),
                "num_samples": len(rows),
            },
            "objective": objective
            or {
                "loss_type": "classification",
                "num_bins": 51,
                "reward_range": [0.0, 1.0],
                "target_type": "hl_gauss",
                "hl_gauss_sigma_ratio": 0.75,
            },
        }
        manifest_path = root / "manifest.json"
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        return manifest_path

    @staticmethod
    def _rows():
        return [
            {
                "sample_id": f"sample-{index}",
                "tokens": [10, 20, 30 + index, 40 + index],
                "response_length": 2,
                "returns": [float(index % 2), float((index + 1) % 2)],
                "loss_mask": [1, 1],
            }
            for index in range(6)
        ]

    def test_manifest_dataset_and_batches_are_deterministic(self):
        with tempfile.TemporaryDirectory() as temporary:
            manifest_path = self._write_fixture(Path(temporary), self._rows())
            expected_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
            manifest = load_value_pretrain_manifest(manifest_path, expected_sha256=expected_sha)
            dataset = ValuePretrainDataset(manifest.train, manifest.objective)

            first = tuple(dataset.batch_indices(global_batch_size=2, epochs=2, seed=17))
            second = tuple(dataset.batch_indices(global_batch_size=2, epochs=2, seed=17))
            different = tuple(dataset.batch_indices(global_batch_size=2, epochs=2, seed=18))
            self.assertEqual(first, second)
            self.assertNotEqual(first, different)
            self.assertEqual(len(first), value_pretrain_num_steps(num_samples=6, global_batch_size=2, epochs=2))

            batch = dataset.build_rollout_batch(first[0])
            self.assertEqual(set(batch), {"tokens", "response_lengths", "returns", "loss_masks", "sample_indices", "sample_ids"})
            self.assertEqual(len(batch["tokens"]), 2)

    def test_rejects_duplicate_ids_and_out_of_range_targets(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = self._rows()
            rows[1]["sample_id"] = rows[0]["sample_id"]
            manifest = load_value_pretrain_manifest(self._write_fixture(root, rows))
            with self.assertRaisesRegex(ValueError, "duplicate sample_id"):
                ValuePretrainDataset(manifest.train, manifest.objective)

        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = self._rows()
            rows[0]["returns"][0] = 1.1
            manifest = load_value_pretrain_manifest(self._write_fixture(root, rows))
            with self.assertRaisesRegex(ValueError, "classification returns"):
                ValuePretrainDataset(manifest.train, manifest.objective)

    def test_rejects_dataset_hash_mismatch(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = load_value_pretrain_manifest(self._write_fixture(root, self._rows()))
            with manifest.train.path.open("ab") as stream:
                stream.write(b"\n")
            with self.assertRaisesRegex(ValueError, "dataset SHA-256 mismatch"):
                ValuePretrainDataset(manifest.train, manifest.objective)

    def test_checkpoint_contract_round_trip_and_digest_gate(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            manifest = load_value_pretrain_manifest(self._write_fixture(root, self._rows()))
            path, digest = write_value_pretrain_contract(
                root / "checkpoint",
                manifest=manifest,
                completed_steps=3,
                model_identity="Qwen/test-model",
                seed=1234,
                global_batch_size=2,
            )
            loaded = load_value_pretrain_contract(path.parent, expected_sha256=digest)
            self.assertEqual(loaded["completed_steps"], 3)
            self.assertEqual(loaded["source_manifest_sha256"], manifest.sha256)
            self.assertEqual(loaded["batch_plan"]["seed"], 1234)
            with self.assertRaisesRegex(ValueError, "contract SHA-256 mismatch"):
                load_value_pretrain_contract(path.parent, expected_sha256="0" * 64)


if __name__ == "__main__":
    unittest.main()
