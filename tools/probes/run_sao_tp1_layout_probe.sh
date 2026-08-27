#!/usr/bin/env bash

set -euo pipefail

: "${YETO_SAO_TP1_LAYOUT_EVIDENCE:?missing TP1 layout evidence path}"
: "${YETO_SAO_MODEL_REVISION:?missing immutable model revision}"
: "${YETO_SAO_MODEL_CONFIG_SHA256:?missing model config SHA256}"

test -d /root/miles
test -d /root/yeto
test -d /root/models/Qwen3.5-4B
test -f /root/checkpoints/Qwen3.5-4B_torch_dist/latest_checkpointed_iteration.txt
test -f /root/yeto/tests/fixtures/qwen35_full_parameter_probe.jsonl
test ! -e "${YETO_SAO_TP1_LAYOUT_EVIDENCE}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=/root/Megatron-LM:/root/miles:/root/yeto
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MILES_EXPERIMENTAL_FT_TRAINER=0
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=0

runtime_env_json="$({
  python -c '
import json
import os

names = (
    "PYTHONDONTWRITEBYTECODE",
    "PYTHONUNBUFFERED",
    "PYTHONPATH",
    "CUDA_DEVICE_MAX_CONNECTIONS",
    "NCCL_NVLS_ENABLE",
    "PYTORCH_CUDA_ALLOC_CONF",
    "MILES_EXPERIMENTAL_FT_TRAINER",
    "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR",
    "YETO_SAO_TP1_LAYOUT_EVIDENCE",
    "YETO_SAO_MODEL_REVISION",
    "YETO_SAO_MODEL_CONFIG_SHA256",
)
print(json.dumps({"env_vars": {name: os.environ[name] for name in names}}))
'
})"

cd /root/miles
# shellcheck source=/dev/null
source scripts/models/qwen3.5-4B.sh

ray job submit \
  --address=http://127.0.0.1:8265 \
  --runtime-env-json="${runtime_env_json}" \
  -- python3 /root/miles/train.py \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 1 \
  "${MODEL_ARGS[@]}" \
  --hf-checkpoint /root/models/Qwen3.5-4B \
  --ref-load /root/checkpoints/Qwen3.5-4B_torch_dist \
  --prompt-data /root/yeto/tests/fixtures/qwen35_full_parameter_probe.jsonl \
  --input-key prompt \
  --label-key label \
  --rm-type deepscaler \
  --num-rollout 1 \
  --rollout-batch-size 1 \
  --n-samples-per-prompt 1 \
  --rollout-max-response-len 64 \
  --global-batch-size 1 \
  --micro-batch-size 1 \
  --debug-train-only \
  --disable-rollout-global-dataset \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --seq-length 4096 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --external-policy-sync-path \
  tools.probes.sao_tp1_layout_probe.create_sao_tp1_layout_probe

test -s "${YETO_SAO_TP1_LAYOUT_EVIDENCE}"
test "$(stat -c '%a' "${YETO_SAO_TP1_LAYOUT_EVIDENCE}")" = 600
