# SAO value pretraining

> **Production validation:** the exact Qwen3.5-0.8B value dataset, Qwen3.8-27B
> teacher provenance, checkpoint contract, launch sequence, and measured results
> are documented in
> [Qwen3.5-0.8B Terminal-Bench 2.1 SAO + Streaming DiLoCo Validation](TBENCH21_SAO_QWEN35_08B_VALIDATION_20260826.md).

`train_value.py` is a standalone offline critic-training job. It allocates only
the critic model, never starts an actor or SGLang, and writes a normal Miles
critic checkpoint that the online SAO job can load with `--critic-load`.

This path is intentionally independent of DiLoCo. A future streaming-DiLoCo
trainer can consume the same checkpoint contract without changing the offline
dataset or value objective.

## Dataset contract

The training dataset is JSONL. Each non-empty line has:

```json
{
  "sample_id": "unique-stable-id",
  "tokens": [1, 2, 3, 4],
  "response_length": 2,
  "returns": [0.0, 1.0],
  "loss_mask": [1, 1]
}
```

- `tokens` contains the prompt and response and must leave at least one prompt token.
- `returns` and `loss_mask` are response-token aligned and must have exactly
  `response_length` entries.
- `returns` are explicit offline targets. The trainer does not silently infer a
  credit-assignment rule from episode rewards.
- `loss_mask` defaults to all ones when omitted, but must select at least one token.
- `sample_id` must be unique across the file.

The manifest pins both the data and objective:

```json
{
  "schema": "miles.value-pretrain.v1",
  "train": {
    "path": "train.jsonl",
    "sha256": "<sha256-of-exact-jsonl-bytes>",
    "num_samples": 1000
  },
  "objective": {
    "loss_type": "classification",
    "num_bins": 51,
    "reward_range": [0.0, 1.0],
    "target_type": "hl_gauss",
    "hl_gauss_sigma_ratio": 0.75
  }
}
```

The whole dataset is hash-checked and structurally validated before Miles
reserves any GPU.

## Training

Use the normal Miles model and parallelism flags, plus:

```bash
python train_value.py \
  --value-pretrain-manifest /data/value/manifest.json \
  --value-pretrain-manifest-sha256 <manifest-sha256> \
  --value-pretrain-epochs 1 \
  --critic-save /checkpoints/qwen35-4b-value \
  --critic-num-nodes 1 \
  --critic-num-gpus-per-node 2 \
  --global-batch-size 8 \
  --micro-batch-size 1 \
  <normal Qwen3.5-4B Megatron flags>
```

One complete global batch is one optimizer step. Incomplete final batches are
dropped deterministically. Every saved checkpoint receives
`value_pretrain_contract.json`, which records the exact source manifest,
dataset, objective, model identity, deterministic seed/global-batch plan, and
completed value-pretraining step. A resume may extend the epoch count, but it
cannot silently change the seed or global batch size.

Resume by setting `--critic-load` to the same contracted checkpoint. The job
rejects changed data, objectives, or ambiguous non-zero checkpoints without a
contract.

## SAO handoff

The online job should load the checkpoint and bind the exact contract digest:

```bash
python train.py \
  --critic-load /checkpoints/qwen35-4b-value \
  --critic-value-pretrain-contract-sha256 <contract-sha256> \
  <SAO/PPO flags>
```

Binding the digest automatically restores the classification-head objective,
preventing an MSE/classification head mismatch.

## Smoke gates

The small-model GPU smoke is successful only if:

1. malformed or hash-mismatched data fails before GPU allocation;
2. a classification critic completes optimizer steps with finite loss and EV;
3. save and resume advance without replaying completed batches;
4. `--critic-load` plus the contract digest initializes an online critic;
5. a one-step centralized SAO/PPO probe consumes the pretrained critic.

The later streaming-DiLoCo integration is a separate gate.

The centralized handoff and streaming replacement boundary are specified in
[`SAO_ONLINE.md`](SAO_ONLINE.md).
