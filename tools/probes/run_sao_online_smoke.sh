#!/usr/bin/env bash

set -euo pipefail

: "${YETO_SAO_CONTEXT_SHA256:?missing SAO context SHA256}"
: "${YETO_CRITIC_CONTRACT_SHA256:?missing critic contract SHA256}"
: "${YETO_CODEX_HARNESS_CONTRACT_SHA256:?missing Codex harness contract SHA256}"
: "${SECRLENV_TASK_PACK_SHA256:?missing SecRLEnv task-pack SHA256}"
: "${SECRLENV_DAEMON_URL:?missing SecRLEnv daemon URL}"
: "${SECRLENV_BEARER_TOKEN_FILE:?missing SecRLEnv token file}"

CONTEXT=/root/run/sao-context.json
CODEX_CONTRACT=/root/run/codex-harness.json
LAYOUT_EVIDENCE=/root/run/tp1-layout.json
CRITIC=/root/checkpoints/Qwen3.5-4B_value_smoke
MODEL=/root/models/Qwen3.5-4B
ACTOR=/root/checkpoints/Qwen3.5-4B_torch_dist
DATA=/root/inputs/qwen35-m1-dense-secrlenv-v1/train.jsonl

test -d /root/miles
test -d /root/yeto
test -d "${MODEL}"
test -f "${ACTOR}/latest_checkpointed_iteration.txt"
test -f "${CRITIC}/latest_checkpointed_iteration.txt"
test -f "${CRITIC}/value_pretrain_contract.json"
test -f "${DATA}"
test -f "${CONTEXT}"
test -f "${CODEX_CONTRACT}"
test -f "${LAYOUT_EVIDENCE}"
test -f "${SECRLENV_BEARER_TOKEN_FILE}"
test "$(stat -c '%a' "${SECRLENV_BEARER_TOKEN_FILE}")" = 600
test "$(sha256sum "${CONTEXT}" | cut -d ' ' -f 1)" = "${YETO_SAO_CONTEXT_SHA256}"
test "$(sha256sum "${CODEX_CONTRACT}" | cut -d ' ' -f 1)" = "${YETO_CODEX_HARNESS_CONTRACT_SHA256}"
test "$(sha256sum "${CRITIC}/value_pretrain_contract.json" | cut -d ' ' -f 1)" = "${YETO_CRITIC_CONTRACT_SHA256}"
test "$(sha256sum "${LAYOUT_EVIDENCE}" | cut -d ' ' -f 1)" = 64f1f0c29f80fe6102d77b1e59b64f9f5ec28ea8402803b5178ce3ae28044d18
test "$(sha256sum "${DATA}" | cut -d ' ' -f 1)" = de1c7b371ed5bc71fcfcf5563284ba7a88320b465130978ec051a0cb662f338f
test "$(sha256sum /root/yeto/yeto_miles_secrlenv/codex_harness_agent.py | cut -d ' ' -f 1)" = 509b3623df0a972031082f697f96223a23e13ffddc8169a0eb1060d56eb91868
test "$(sha256sum /root/yeto/yeto_miles_secrlenv/generate.py | cut -d ' ' -f 1)" = 9e034d6b2e9fec642501ea4a638a8fe196819dacde614ce2903359fc54ea1713
test "$(sha256sum /root/yeto/yeto_miles_secrlenv/reward.py | cut -d ' ' -f 1)" = e54a69ee754ac1babda79663521bb402b61661fdddb1864e9d0b28f6a1949b57
test "$(sha256sum /opt/yeto/codex/codex-x86_64-unknown-linux-musl | cut -d ' ' -f 1)" = a2a05dafaa1acb002a45eaec0a462de5b13694fcfcd7bc43305f14781ce7be14

export PYTHONDONTWRITEBYTECODE=1
export PYTHONUNBUFFERED=1
export PYTHONPATH=/root/Megatron-LM:/root/miles:/root/yeto
export CUDA_DEVICE_MAX_CONNECTIONS=1
export NCCL_NVLS_ENABLE=1
export WITH_NVIDIA_PEERMEM=0
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export MILES_EXPERIMENTAL_FT_TRAINER=0
export MILES_EXPERIMENTAL_ROLLOUT_REFACTOR=1
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HUB_DISABLE_TELEMETRY=1
export HF_HOME=/root/run/hf-cache
export FLASHINFER_USE_CUDA_NORM=1
export RAY_DEDUP_LOGS=0

mkdir -p "${HF_HOME}"

python - <<'PY'
from yeto_miles_secrlenv.client import require_daemon_ready
require_daemon_ready(timeout_seconds=10)
PY

python /root/miles/tools/probes/check_sao_qwen35_raw_checkpoints.py \
  --actor "${ACTOR}" \
  --critic "${CRITIC}" \
  --hf "${MODEL}"

cd /root/miles
# shellcheck source=/dev/null
source scripts/models/qwen3.5-4B.sh

MILES_ARGS=(
  --train-backend megatron
  --hf-checkpoint "${MODEL}"
  --ref-load "${ACTOR}"
  --critic-load "${CRITIC}"
  --critic-value-pretrain-contract-sha256 "${YETO_CRITIC_CONTRACT_SHA256}"
  # The staged actor and value checkpoints use the native Miles/Megatron
  # Qwen3.5 layout.  Building an AutoBridge VLM wrapper here prefixes every
  # checkpoint key with language_model./vision_model. and silently leaves a
  # random model behind, so keep training on the matching native layout and
  # use the existing Qwen3.5 exporter for rollout publication.
  --megatron-to-hf-mode raw
  --model-name qwen3_5
  "${MODEL_ARGS[@]}"
  --max-position-embeddings 262144
  --seq-length 8192
  --num-gpus-per-node 4
  --actor-num-nodes 1
  --actor-num-gpus-per-node 1
  --critic-num-nodes 1
  --critic-num-gpus-per-node 1
  --rollout-num-gpus 1
  --rollout-num-gpus-per-engine 1
  --tensor-model-parallel-size 1
  --pipeline-model-parallel-size 1
  --context-parallel-size 1
  --expert-model-parallel-size 1
  --expert-tensor-parallel-size 1
  --recompute-granularity full
  --recompute-method uniform
  --recompute-num-layers 1
  --micro-batch-size 1
  --optimizer adam
  --lr 1e-6
  --lr-decay-style constant
  --weight-decay 0.0
  --adam-beta1 0.9
  --adam-beta2 0.98
  --prompt-data "${DATA}"
  --input-key messages
  --metadata-key metadata
  --num-rollout 1
  --rollout-batch-size 1
  --n-samples-per-prompt 1
  --over-sampling-batch-size 2
  --num-steps-per-rollout 1
  --global-batch-size 1
  --rollout-max-context-len 8192
  --rollout-max-response-len 4096
  --rollout-function-path yeto.rl.miles.generate_rollout
  --custom-generate-function-path yeto_miles_secrlenv.generate.generate
  --custom-agent-function-path yeto_miles_secrlenv.codex_harness_agent.run
  --custom-rm-path yeto_miles_secrlenv.reward.reward_func
  --dynamic-sampling-filter-path yeto_miles_secrlenv.reward.check_group
  --rollout-all-samples-process-path yeto.rl.miles.queue_completed_groups
  --sao-online-recipe coding
  --use-session-server
  --session-server-ip 127.0.0.1
  --session-server-port 31801
  --tito-model qwen35
  --tito-allowed-append-roles tool user
  --max-seq-len 8192
  --apply-chat-template-kwargs '{"clear_thinking":false}'
  --sglang-reasoning-parser qwen3
  --sglang-tool-call-parser qwen3_coder
  --sglang-context-length 8192
  --sglang-mem-fraction-static 0.60
  --sglang-max-running-requests 1
  --sglang-disable-cuda-graph
  --sglang-cuda-graph-backend-prefill disabled
  --sglang-chunked-prefill-size 4096
  --sglang-enable-deterministic-inference
  --rollout-engine-base-port 21000
  --sglang-router-port 23000
  --sglang-router-prometheus-port 23001
  --train-master-base-port 25000
  --update-weight-buffer-size 1073741824
  --update-weight-transfer-mode broadcast
  --rollout-weight-version-format yeto-policy
  --attention-dropout 0.0
  --hidden-dropout 0.0
  --attention-backend flash
  --accumulate-allreduce-grads-in-fp32
  --attention-softmax-in-fp32
  --bf16
  --no-load-optim
  --no-load-rng
  # Never permit another partial/random checkpoint load.  Optimizer-only
  # checkpoint entries may be omitted, but every tensor requested by the
  # current model must exist in the checkpoint.
  --dist-ckpt-strictness raise_unexpected
  --no-save-optim
  --no-save-rng
  --finetune
  --seed 731
  --rollout-seed 731
  --distributed-timeout-minutes 30
  --pin-rollout-manager-to-head
  --wandb-mode disabled
  --dump-details /root/run/details
)

# Parse and bind the exact production argument vector without reserving a GPU.
python - "${CONTEXT}" "${YETO_SAO_CONTEXT_SHA256}" "${DATA}" "${MILES_ARGS[@]}" <<'PY'
import json
import sys
from tools.probes.train_sao_secrlenv import bind_context, load_context

context_path, context_sha, data_path, *miles_args = sys.argv[1:]
context = load_context(context_path, context_sha)
with open(data_path, encoding="utf-8") as source:
    rows = [json.loads(line) for line in source if line.strip()]
assert rows
assert all(
    isinstance(row.get("messages"), list)
    and isinstance(row.get("metadata"), dict)
    and isinstance(row["metadata"].get("task_id"), str)
    for row in rows
)
previous = sys.argv
try:
    sys.argv = ["sao-online-preflight", *miles_args]
    from miles.utils.arguments import parse_args
    args = parse_args()
finally:
    sys.argv = previous
bind_context(args, context)
assert args.sao_online_recipe == "coding"
assert args.policy_objective == "sao_dis"
assert args.use_critic is True
assert args.value_loss_type == "classification"
assert args.value_num_bins == 51
assert args.n_samples_per_prompt == 1
assert args.actor_num_gpus_per_node == 1
assert args.critic_num_gpus_per_node == 1
assert args.rollout_num_gpus == 1
assert args.megatron_to_hf_mode == "raw"
assert args.bridge_distributed_weight_sync is False
assert getattr(args.dist_ckpt_strictness, "value", args.dist_ckpt_strictness) == "raise_unexpected"
assert args.external_policy_sync_path is None
print("SAO_ARGUMENT_PREFLIGHT_OK", flush=True)
PY

if [[ "${YETO_SAO_PREFLIGHT_ONLY:-0}" == 1 ]]; then
  echo "SAO_ONLINE_PREFLIGHT_ONLY_OK"
  exit 0
fi

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
    "WITH_NVIDIA_PEERMEM",
    "PYTORCH_CUDA_ALLOC_CONF",
    "MILES_EXPERIMENTAL_FT_TRAINER",
    "MILES_EXPERIMENTAL_ROLLOUT_REFACTOR",
    "HF_HUB_OFFLINE",
    "TRANSFORMERS_OFFLINE",
    "HF_HUB_DISABLE_TELEMETRY",
    "HF_HOME",
    "FLASHINFER_USE_CUDA_NORM",
    "RAY_DEDUP_LOGS",
    "SECRLENV_DAEMON_URL",
    "SECRLENV_TASK_PACK_SHA256",
    "SECRLENV_BEARER_TOKEN_FILE",
    "YETO_CODEX_BINARY_PATH",
    "YETO_CODEX_BINARY_SHA256",
    "YETO_CODEX_BINARY_SIZE_BYTES",
    "YETO_CODEX_VERSION",
    "YETO_CODEX_APP_SERVER_PROTOCOL_REVISION",
    "YETO_CODEX_APP_SERVER_SCHEMA_SHA256",
    "YETO_CODEX_BASE_INSTRUCTIONS_SHA256",
    "YETO_CODEX_TERMINAL_EXEC_TOOL_SCHEMA_SHA256",
    "YETO_CODEX_SUBMIT_TOOL_SCHEMA_SHA256",
    "YETO_CODEX_DYNAMIC_TOOLS_SCHEMA_SHA256",
    "YETO_CODEX_REASONING_EFFORT",
    "YETO_CODEX_BACKEND_MAX_TOKENS",
    "YETO_CODEX_BACKEND_REASONING_EFFORT",
    "YETO_CODEX_BACKEND_THINKING",
    "YETO_CODEX_CHAT_TEMPLATE",
    "YETO_CODEX_CHAT_TEMPLATE_KWARGS",
    "YETO_CODEX_TITO_ALLOWED_APPEND_ROLES",
    "YETO_CODEX_HARNESS_CONTRACT_SHA256",
)
print(json.dumps({"env_vars": {name: os.environ[name] for name in names}}))
'
})"

ray job submit \
  --address=http://127.0.0.1:8265 \
  --runtime-env-json="${runtime_env_json}" \
  -- python3 /root/miles/tools/probes/train_sao_secrlenv.py \
  --sao-secrlenv-context "${CONTEXT}" \
  --sao-secrlenv-context-sha256 "${YETO_SAO_CONTEXT_SHA256}" \
  "${MILES_ARGS[@]}"
