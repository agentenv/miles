#!/usr/bin/env bash

set -euo pipefail

MILES_ROOT="${MILES_ROOT:-/root/miles}"
MEGATRON_ROOT="${MEGATRON_ROOT:-/root/Megatron-LM}"
MODEL_ROOT="${MODEL_ROOT:-/root/models/Qwen3.5-0.8B}"
MODEL_ID="Qwen/Qwen3.5-0.8B"
MODEL_REVISION="2fc06364715b967f1860aea9cf38778875588b17"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/root/checkpoints/Qwen3.5-0.8B_torch_dist}"
VALUE_CHECKPOINT="${VALUE_CHECKPOINT:-/root/checkpoints/Qwen3.5-0.8B_value_tbench21_full}"
VALUE_MANIFEST="${VALUE_MANIFEST:-/root/data/tbench21-compaction-value-all/manifest.json}"
VALUE_MANIFEST_SHA256="${VALUE_MANIFEST_SHA256:-}"
VALUE_PLAN="${VALUE_PLAN:-${VALUE_CHECKPOINT}.launch-plan.json}"
RAY_DASHBOARD="${RAY_DASHBOARD:-http://127.0.0.1:8265}"
SEQ_LENGTH="${SEQ_LENGTH:-8192}"
MAX_CRITIC_GPUS="${MAX_CRITIC_GPUS:-8}"
MAX_GLOBAL_BATCH_SIZE="${MAX_GLOBAL_BATCH_SIZE:-64}"
VALUE_PRETRAIN_CANARY_ONE_STEP="${VALUE_PRETRAIN_CANARY_ONE_STEP:-0}"

if [[ -z "${VALUE_MANIFEST_SHA256}" ]]; then
  echo "VALUE_MANIFEST_SHA256 is required; use the digest emitted by the trusted converter" >&2
  exit 1
fi
if [[ ! "${VALUE_MANIFEST_SHA256}" =~ ^[0-9a-f]{64}$ ]]; then
  echo "VALUE_MANIFEST_SHA256 must be a lowercase SHA-256 digest" >&2
  exit 1
fi
if [[ "${VALUE_PRETRAIN_CANARY_ONE_STEP}" != "0" && "${VALUE_PRETRAIN_CANARY_ONE_STEP}" != "1" ]]; then
  echo "VALUE_PRETRAIN_CANARY_ONE_STEP must be 0 or 1" >&2
  exit 1
fi

test -d "${MILES_ROOT}"
test -d "${MEGATRON_ROOT}"
test -f "${MODEL_ROOT}/config.json"
test -f "${BASE_CHECKPOINT}/latest_checkpointed_iteration.txt"
test -f "${VALUE_MANIFEST}"

actual_manifest_sha256="$(
  python3 - "${VALUE_MANIFEST}" <<'PY'
import hashlib
import sys
from pathlib import Path

digest = hashlib.sha256()
with Path(sys.argv[1]).open("rb") as stream:
    for chunk in iter(lambda: stream.read(1024 * 1024), b""):
        digest.update(chunk)
print(digest.hexdigest())
PY
)"
if [[ "${actual_manifest_sha256}" != "${VALUE_MANIFEST_SHA256}" ]]; then
  echo "VALUE_MANIFEST_SHA256 does not match VALUE_MANIFEST" >&2
  exit 1
fi

if [[ -e "${VALUE_PLAN}" || -L "${VALUE_PLAN}" ]]; then
  echo "value-pretraining launch plan path must be fresh: ${VALUE_PLAN}" >&2
  exit 1
fi
if [[ -e "${VALUE_CHECKPOINT}" || -L "${VALUE_CHECKPOINT}" ]]; then
  echo "value-pretraining checkpoint path must be fresh: ${VALUE_CHECKPOINT}" >&2
  exit 1
fi

export PYTHONPATH="${MEGATRON_ROOT}:${MILES_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1

detected_gpu_count="$(nvidia-smi --query-gpu=index --format=csv,noheader | sed '/^[[:space:]]*$/d' | wc -l | tr -d '[:space:]')"
if [[ ! "${detected_gpu_count}" =~ ^[1-9][0-9]*$ ]]; then
  echo "could not detect a positive local GPU count" >&2
  exit 1
fi
if [[ ! "${MAX_CRITIC_GPUS}" =~ ^[1-9][0-9]*$ ]] || [[ "${MAX_CRITIC_GPUS}" -gt 8 ]]; then
  echo "MAX_CRITIC_GPUS must be in [1, 8]" >&2
  exit 1
fi
available_gpu_count="${detected_gpu_count}"
if [[ "${available_gpu_count}" -gt "${MAX_CRITIC_GPUS}" ]]; then
  available_gpu_count="${MAX_CRITIC_GPUS}"
fi

plan_json="$(
  python3 "${MILES_ROOT}/tools/probes/plan_tbench21_value_pretrain.py" \
    --manifest "${VALUE_MANIFEST}" \
    --manifest-sha256 "${VALUE_MANIFEST_SHA256}" \
    --expected-model "${MODEL_ID}" \
    --expected-revision "${MODEL_REVISION}" \
    --available-gpus "${available_gpu_count}" \
    --max-global-batch-size "${MAX_GLOBAL_BATCH_SIZE}" \
    --output "${VALUE_PLAN}"
)"
mapfile -t batch_plan < <(
  python3 - "${plan_json}" <<'PY'
import json
import sys

payload = json.loads(sys.argv[1])
batch = payload["batch"]
for field in ("num_samples", "dp_size", "global_batch_size", "optimizer_steps"):
    print(batch[field])
PY
)
if [[ "${#batch_plan[@]}" -ne 4 ]]; then
  echo "value-pretraining planner returned an invalid batch plan" >&2
  exit 1
fi
NUM_SAMPLES="${batch_plan[0]}"
CRITIC_DP_SIZE="${batch_plan[1]}"
GLOBAL_BATCH_SIZE="${batch_plan[2]}"
EXPECTED_STEPS="${batch_plan[3]}"
RUN_STEPS="${EXPECTED_STEPS}"
CANARY_ARGS=()
if [[ "${VALUE_PRETRAIN_CANARY_ONE_STEP}" == "1" ]]; then
  RUN_STEPS=1
  CANARY_ARGS+=(--value-pretrain-canary-one-step)
fi

python3 - "${VALUE_MANIFEST}" "${VALUE_MANIFEST_SHA256}" "${SEQ_LENGTH}" "${NUM_SAMPLES}" <<'PY'
import sys

from miles.value_pretraining import ValuePretrainDataset, load_value_pretrain_manifest

manifest_path, expected_sha256, seq_length_raw, expected_samples_raw = sys.argv[1:]
seq_length = int(seq_length_raw)
expected_samples = int(expected_samples_raw)
manifest = load_value_pretrain_manifest(manifest_path, expected_sha256=expected_sha256)
datasets = [("train", ValuePretrainDataset(manifest.train, manifest.objective))]
if manifest.heldout is not None:
    datasets.append(("heldout", ValuePretrainDataset(manifest.heldout, manifest.objective)))

report = {}
for name, dataset in datasets:
    max_tokens = 0
    max_token_id = -1
    active_targets = 0
    for index in range(len(dataset)):
        sample = dataset.get(index)
        max_tokens = max(max_tokens, len(sample.tokens))
        max_token_id = max(max_token_id, max(sample.tokens))
        active_targets += sum(sample.loss_mask)
    if max_tokens > seq_length:
        raise ValueError(f"{name} sample length {max_tokens} exceeds seq-length={seq_length}")
    if max_token_id >= 248320:
        raise ValueError(f"{name} token ID {max_token_id} exceeds Qwen3.5 vocabulary")
    report[name] = {
        "samples": len(dataset),
        "max_tokens": max_tokens,
        "max_token_id": max_token_id,
        "active_targets": active_targets,
    }
if len(datasets) != 1 or report["train"]["samples"] != expected_samples:
    raise ValueError("full-baseline value dataset count differs from its launch plan")
print("SAO_VALUE_DATA_PREFLIGHT_OK", report, flush=True)
PY

echo "SAO_VALUE_BATCH_PLAN samples=${NUM_SAMPLES} dp=${CRITIC_DP_SIZE} gbs=${GLOBAL_BATCH_SIZE} full_steps=${EXPECTED_STEPS} run_steps=${RUN_STEPS} canary=${VALUE_PRETRAIN_CANARY_ONE_STEP} dropped=0 repeated=0"

cd "${MILES_ROOT}"
# shellcheck source=/dev/null
source scripts/models/qwen3.5-0.8B.sh

runtime_env_json="$(python3 - <<PY
import json

print(
    json.dumps(
        {
            "env_vars": {
                "PYTHONPATH": "${PYTHONPATH}",
                "PYTHONDONTWRITEBYTECODE": "1",
                "PYTHONUNBUFFERED": "1",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_NVLS_ENABLE": "0",
                "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
                "RAY_DEDUP_LOGS": "0",
            }
        },
        separators=(",", ":"),
    )
)
PY
)"

# Miles omits exactly the fresh critic head while loading this actor checkpoint,
# so strict checkpoint validation still covers every backbone tensor.
ray job submit \
  --address="${RAY_DASHBOARD}" \
  --runtime-env-json="${runtime_env_json}" \
  -- python3 "${MILES_ROOT}/train_value.py" \
  "${MODEL_ARGS[@]}" \
  --train-backend megatron \
  --hf-checkpoint "${MODEL_ROOT}" \
  --model-name qwen3_5 \
  --critic-load "${BASE_CHECKPOINT}" \
  --critic-save "${VALUE_CHECKPOINT}" \
  --save "${VALUE_CHECKPOINT}" \
  --value-pretrain-manifest "${VALUE_MANIFEST}" \
  --value-pretrain-manifest-sha256 "${VALUE_MANIFEST_SHA256}" \
  --value-pretrain-epochs 1 \
  "${CANARY_ARGS[@]}" \
  --critic-num-nodes 1 \
  --critic-num-gpus-per-node "${CRITIC_DP_SIZE}" \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --micro-batch-size 1 \
  --global-batch-size "${GLOBAL_BATCH_SIZE}" \
  --seq-length "${SEQ_LENGTH}" \
  --max-position-embeddings 262144 \
  --save-interval "${RUN_STEPS}" \
  --finetune \
  --bf16 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --attention-backend flash \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --dist-ckpt-strictness raise_unexpected \
  --no-load-optim \
  --no-load-rng \
  --no-save-optim \
  --no-save-rng \
  --seed 82621 \
  --wandb-mode disabled

python3 "${MILES_ROOT}/tools/probes/check_sao_qwen35_raw_checkpoints.py" \
  --actor "${BASE_CHECKPOINT}" \
  --critic "${VALUE_CHECKPOINT}" \
  --hf "${MODEL_ROOT}" \
  --value-num-bins 51

python3 - \
  "${VALUE_MANIFEST}" \
  "${VALUE_MANIFEST_SHA256}" \
  "${VALUE_CHECKPOINT}" \
  "${NUM_SAMPLES}" \
  "${CRITIC_DP_SIZE}" \
  "${GLOBAL_BATCH_SIZE}" \
  "${EXPECTED_STEPS}" \
  "${VALUE_PLAN}" \
  "${VALUE_PRETRAIN_CANARY_ONE_STEP}" \
  "${MODEL_ROOT}" <<'PY'
import hashlib
import json
import os
import sys
from pathlib import Path

from miles.value_pretraining import (
    VALUE_PRETRAIN_CONTRACT_NAME,
    ValuePretrainDataset,
    load_value_pretrain_contract,
    load_value_pretrain_manifest,
)

(
    manifest_path,
    manifest_sha256,
    checkpoint_raw,
    samples_raw,
    dp_size_raw,
    global_batch_raw,
    steps_raw,
    plan_raw,
    canary_raw,
    model_identity,
) = sys.argv[1:]
checkpoint = Path(checkpoint_raw)
samples = int(samples_raw)
dp_size = int(dp_size_raw)
global_batch_size = int(global_batch_raw)
steps = int(steps_raw)
plan = Path(plan_raw)
canary = canary_raw == "1"
manifest = load_value_pretrain_manifest(
    manifest_path,
    expected_sha256=manifest_sha256,
)
if steps * global_batch_size != samples:
    raise RuntimeError("value launch plan does not prove exact one-epoch coverage")
contract_path = checkpoint / VALUE_PRETRAIN_CONTRACT_NAME
marker = (checkpoint / "latest_checkpointed_iteration.txt").read_text().strip()
if canary:
    if os.path.lexists(contract_path):
        raise RuntimeError("one-step canary published a production value contract")
    if marker != "1":
        raise RuntimeError(f"one-step canary checkpoint marker is {marker!r}, expected '1'")
    dataset = ValuePretrainDataset(manifest.train, manifest.objective)
    first_batch = next(
        dataset.batch_indices(
            global_batch_size=global_batch_size,
            epochs=1,
            seed=82621,
        )
    )
    sample_ids = [dataset.get(index).sample_id for index in first_batch]
    sample_ids_sha256 = hashlib.sha256(
        ("\n".join(sample_ids) + "\n").encode("utf-8")
    ).hexdigest()
    receipt = checkpoint / "value_pretrain_canary_receipt.json"
    payload = {
        "schema": "miles.value-pretrain-canary.v1",
        "promotable": False,
        "checkpoint": str(checkpoint),
        "source_manifest_sha256": manifest.sha256,
        "train_dataset_sha256": manifest.train.sha256,
        "launch_plan_sha256": hashlib.sha256(plan.read_bytes()).hexdigest(),
        "model_identity": model_identity,
        "seed": 82621,
        "dp_size": dp_size,
        "global_batch_size": global_batch_size,
        "full_optimizer_steps": steps,
        "executed_optimizer_steps": 1,
        "sample_ids_sha256": sample_ids_sha256,
        "production_contract_present": False,
    }
    encoded = (
        json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True)
        + "\n"
    ).encode("utf-8")
    descriptor = os.open(receipt, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    with os.fdopen(descriptor, "wb", closefd=True) as stream:
        stream.write(encoded)
        stream.flush()
        os.fsync(stream.fileno())
    print(
        "SAO_VALUE_CANARY_READY",
        json.dumps(payload, separators=(",", ":"), sort_keys=True),
        flush=True,
    )
else:
    contract_bytes = contract_path.read_bytes()
    contract_sha256 = hashlib.sha256(contract_bytes).hexdigest()
    contract = load_value_pretrain_contract(
        checkpoint,
        expected_sha256=contract_sha256,
    )
    expected_batch_plan = {
        "seed": 82621,
        "global_batch_size": global_batch_size,
        "drop_incomplete_batch": True,
    }
    if contract["source_manifest_sha256"] != manifest.sha256:
        raise RuntimeError("critic contract references the wrong value manifest")
    if contract["train_dataset_sha256"] != manifest.train.sha256:
        raise RuntimeError("critic contract references the wrong value dataset")
    if contract["objective"] != manifest.objective.to_dict():
        raise RuntimeError("critic contract objective changed")
    if contract["batch_plan"] != expected_batch_plan:
        raise RuntimeError("critic contract batch plan changed")
    if contract["completed_steps"] != steps:
        raise RuntimeError("critic contract does not prove exact one-epoch coverage")
    if contract["model_identity"] != model_identity:
        raise RuntimeError("critic contract model identity changed")
    if marker != str(steps):
        raise RuntimeError(
            f"critic checkpoint marker {marker!r} differs from completed steps {steps}"
        )
    print(
        "SAO_VALUE_CHECKPOINT_READY",
        json.dumps(
            {
                "checkpoint": str(checkpoint),
                "contract_sha256": contract_sha256,
                "samples": samples,
                "dp_size": dp_size,
                "global_batch_size": global_batch_size,
                "optimizer_steps": steps,
                "dropped_rows": 0,
                "repeated_rows": 0,
            },
            separators=(",", ":"),
            sort_keys=True,
        ),
        flush=True,
    )
PY
