#!/usr/bin/env bash

set -euo pipefail

MILES_ROOT="${MILES_ROOT:-/root/miles}"
MODEL_ROOT="${MODEL_ROOT:-/root/models/Qwen3.5-4B}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-/root/checkpoints/Qwen3.5-4B_torch_dist}"
VALUE_CHECKPOINT="${VALUE_CHECKPOINT:-/root/checkpoints/Qwen3.5-4B_value_smoke}"
VALUE_MANIFEST="${VALUE_MANIFEST:-/root/inputs/value-smoke/manifest.json}"
VALUE_MANIFEST_SHA256="${VALUE_MANIFEST_SHA256:-0fccf1f0d52aca1a4a667ea7233818c56c1e1089207df9210a66f4bae85f2ad5}"
RAY_DASHBOARD="${RAY_DASHBOARD:-http://127.0.0.1:8265}"

cd "${MILES_ROOT}"
source scripts/models/qwen3.5-4B.sh

runtime_env_json="$(${PYTHON:-python3} - <<'PY'
import json

print(
    json.dumps(
        {
            "env_vars": {
                "PYTHONPATH": "/root/Megatron-LM:/root/miles:/root/yeto",
                "CUDA_DEVICE_MAX_CONNECTIONS": "1",
                "NCCL_NVLS_ENABLE": "1",
            }
        },
        separators=(",", ":"),
    )
)
PY
)"

exec ray job submit \
  --address="${RAY_DASHBOARD}" \
  --runtime-env-json="${runtime_env_json}" \
  -- python3 "${MILES_ROOT}/train_value.py" \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint "${MODEL_ROOT}" \
  --critic-load "${BASE_CHECKPOINT}" \
  --critic-save "${VALUE_CHECKPOINT}" \
  --save "${VALUE_CHECKPOINT}" \
  --value-pretrain-manifest "${VALUE_MANIFEST}" \
  --value-pretrain-manifest-sha256 "${VALUE_MANIFEST_SHA256}" \
  --value-pretrain-epochs 1 \
  --critic-num-nodes 1 \
  --critic-num-gpus-per-node 1 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --micro-batch-size 1 \
  --global-batch-size 2 \
  --rollout-batch-size 2 \
  --seq-length 128 \
  --save-interval 1 \
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
  --attention-softmax-in-fp32
