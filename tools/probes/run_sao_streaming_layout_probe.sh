#!/usr/bin/env bash

set -euo pipefail

: "${YETO_SAO_STREAMING_LAYOUT_EVIDENCE:?missing streaming layout evidence path}"
: "${YETO_SAO_CONTEXT_SHA256:?missing SAO context SHA256}"
: "${YETO_CRITIC_CONTRACT_SHA256:?missing critic contract SHA256}"
: "${YETO_SAO_ACTOR_MODEL_REVISION:?missing actor model revision}"
: "${YETO_SAO_ACTOR_CONFIG_SHA256:?missing actor config SHA256}"
: "${YETO_SAO_CRITIC_MODEL_REVISION:?missing critic model revision}"
: "${YETO_SAO_CRITIC_CONFIG_SHA256:?missing critic config SHA256}"
: "${YETO_SAO_STREAMING_MAX_FRAGMENT_BYTES:?missing max fragment bytes}"
: "${YETO_SAO_STREAMING_MAX_CHUNK_BYTES:?missing max chunk bytes}"

CONTEXT=/root/run/sao-context.json
CRITIC=/root/checkpoints/Qwen3.5-4B_value_smoke
MODEL=/root/models/Qwen3.5-4B
ACTOR=/root/checkpoints/Qwen3.5-4B_torch_dist
DATA=/root/yeto/tests/fixtures/qwen35_full_parameter_probe.jsonl

test -d /root/miles
test -d /root/yeto
test -d "${MODEL}"
test -f "${ACTOR}/latest_checkpointed_iteration.txt"
test -f "${CRITIC}/latest_checkpointed_iteration.txt"
test -f "${CRITIC}/value_pretrain_contract.json"
test -f "${DATA}"
test -f "${CONTEXT}"
test ! -e "${YETO_SAO_STREAMING_LAYOUT_EVIDENCE}"
test "$(sha256sum "${CONTEXT}" | cut -d ' ' -f 1)" = "${YETO_SAO_CONTEXT_SHA256}"
test "$(sha256sum "${CRITIC}/value_pretrain_contract.json" | cut -d ' ' -f 1)" = "${YETO_CRITIC_CONTRACT_SHA256}"

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=/root/Megatron-LM:/root/miles:/root/yeto
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MILES_EXPERIMENTAL_FT_TRAINER=0
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=0
export YETO_SAO_SECRLENV_CONTEXT_SHA256="${YETO_SAO_CONTEXT_SHA256}"

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
    "YETO_SAO_STREAMING_LAYOUT_EVIDENCE",
    "YETO_SAO_SECRLENV_CONTEXT_SHA256",
    "YETO_SAO_ACTOR_MODEL_REVISION",
    "YETO_SAO_ACTOR_CONFIG_SHA256",
    "YETO_SAO_CRITIC_MODEL_REVISION",
    "YETO_SAO_CRITIC_CONFIG_SHA256",
    "YETO_SAO_STREAMING_MAX_FRAGMENT_BYTES",
    "YETO_SAO_STREAMING_MAX_CHUNK_BYTES",
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
  --train-backend megatron \
  --hf-checkpoint "${MODEL}" \
  --ref-load "${ACTOR}" \
  --critic-load "${CRITIC}" \
  --critic-value-pretrain-contract-sha256 "${YETO_CRITIC_CONTRACT_SHA256}" \
  --megatron-to-hf-mode raw \
  --model-name qwen3_5 \
  "${MODEL_ARGS[@]}" \
  --max-position-embeddings 262144 \
  --seq-length 8192 \
  --num-gpus-per-node 2 \
  --actor-num-nodes 1 \
  --actor-num-gpus-per-node 1 \
  --critic-num-nodes 1 \
  --critic-num-gpus-per-node 1 \
  --tensor-model-parallel-size 1 \
  --pipeline-model-parallel-size 1 \
  --context-parallel-size 1 \
  --expert-model-parallel-size 1 \
  --expert-tensor-parallel-size 1 \
  --recompute-granularity full \
  --recompute-method uniform \
  --recompute-num-layers 1 \
  --micro-batch-size 1 \
  --global-batch-size 1 \
  --optimizer adam \
  --lr 1e-6 \
  --lr-decay-style constant \
  --weight-decay 0.0 \
  --adam-beta1 0.9 \
  --adam-beta2 0.98 \
  --prompt-data "${DATA}" \
  --input-key prompt \
  --label-key label \
  --rm-type deepscaler \
  --num-rollout 1 \
  --rollout-batch-size 1 \
  --n-samples-per-prompt 1 \
  --num-steps-per-rollout 1 \
  --rollout-max-response-len 64 \
  --sao-online-recipe coding \
  --debug-train-only \
  --disable-rollout-global-dataset \
  --attention-dropout 0.0 \
  --hidden-dropout 0.0 \
  --accumulate-allreduce-grads-in-fp32 \
  --attention-softmax-in-fp32 \
  --attention-backend flash \
  --bf16 \
  --no-load-optim \
  --no-load-rng \
  --dist-ckpt-strictness raise_unexpected \
  --external-policy-sync-path \
  tools.probes.sao_streaming_layout_probe.create_sao_streaming_layout_probe

test -s "${YETO_SAO_STREAMING_LAYOUT_EVIDENCE}"
test "$(stat -c '%a' "${YETO_SAO_STREAMING_LAYOUT_EVIDENCE}")" = 600
echo "SAO_STREAMING_LAYOUT_SHA256=$(sha256sum "${YETO_SAO_STREAMING_LAYOUT_EVIDENCE}" | cut -d ' ' -f 1)"
