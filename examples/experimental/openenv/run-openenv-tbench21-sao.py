"""Two-rollout SAO end-to-end gate on Terminal-Bench 2.1.

This launcher intentionally keeps the actor, critic, and SGLang worker on
separate GPUs.  It consumes an offline-pretrained value checkpoint, executes
the real Terminal-Bench 2.1 canonical verifier through the OpenEnv adapter,
and saves actor and critic checkpoints independently after each rollout.

The default model is Qwen/Qwen3.5-0.8B: close to the requested 0.7B scale and
tokenizer-compatible with Qwen3.5-family value data once that compatibility is
verified by the data-preparation step.
"""

from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import openenv_launch_common as C
import typer

import miles.utils.external_utils.command_utils as U


SCRIPT_DIR = Path(__file__).resolve().parent


def _q(value: str | Path) -> str:
    return shlex.quote(str(value))


@dataclass
class ScriptArgs(U.ExecuteTrainConfig):
    run_id: str = U.create_run_id()
    megatron_model_type: str = "qwen3.5-0.8B"
    megatron_path: str = "/root/Megatron-LM"
    num_gpus_per_node: int = 8

    # Model/checkpoint paths.
    skip_prepare: bool = False
    checkpoint_conversion_gpus: int = 1
    model_name: str = "Qwen3.5-0.8B"
    hf_checkpoint: str = "/root/models/Qwen3.5-0.8B"
    ref_load: str = "/root/checkpoints/Qwen3.5-0.8B_torch_dist"
    critic_load: str = "/root/checkpoints/Qwen3.5-0.8B_value"
    critic_contract_sha256: str = ""
    actor_save: str = "/root/runs/tbench21-sao/actor"
    critic_save: str = "/root/runs/tbench21-sao/critic"
    dump_details: str = "/root/runs/tbench21-sao/details"

    # A two-rollout gate is the minimum that proves the published actor is
    # consumed by a subsequent environment episode.
    num_rollout: int = 2
    max_seq_len: int = 8192
    rollout_max_response_len: int = 4096
    seed: int = 82621

    # Terminal-Bench/OpenEnv settings.
    prompt_data: str = "/root/data/tbench21_fix_git.jsonl"
    openenv_env_url: str = os.environ.get("OPENENV_ENV_URL", "http://127.0.0.1:8003")
    agent_model_name: str = os.environ.get("AGENT_MODEL_NAME", "model")
    openenv_max_turns: int = int(os.environ.get("OPENENV_MAX_TURNS", "8"))
    openenv_max_rollout_time_seconds: int = int(
        os.environ.get("OPENENV_MAX_ROLLOUT_TIME_SECONDS", "900")
    )
    openenv_tb2_tasks_dir: str = ""
    daytona_api_key_file: str = ""
    router_external_host: str = ""
    miles_host_ip: str = ""

    # Shared helper protocol fields. This gate deliberately avoids external
    # telemetry so its result is entirely recoverable from local evidence.
    wandb_key: str = ""
    wandb_project: str = "sao-tbench21"
    wandb_team: str = ""
    wandb_run_name: str = "sao-tbench21"
    use_prometheus: bool = False
    prometheus_port: int = 9090
    prometheus_run_name: str = "sao-tbench21"


def _contract_sha256(args: ScriptArgs) -> str:
    contract_path = Path(args.critic_load) / "value_pretrain_contract.json"
    if not contract_path.is_file():
        raise FileNotFoundError(f"missing critic value-pretraining contract: {contract_path}")
    actual = hashlib.sha256(contract_path.read_bytes()).hexdigest()
    if args.critic_contract_sha256 and args.critic_contract_sha256 != actual:
        raise ValueError(
            "critic value-pretraining contract SHA-256 mismatch: "
            f"expected {args.critic_contract_sha256}, got {actual}"
        )
    return actual


def _preflight(args: ScriptArgs) -> str:
    if args.num_rollout < 2:
        raise ValueError("the Terminal-Bench SAO E2E gate requires at least two rollouts")
    if args.checkpoint_conversion_gpus != 1:
        raise ValueError("Qwen3.5-0.8B uses a TP1 checkpoint layout for this gate")

    prompt_path = Path(args.prompt_data)
    if not prompt_path.is_file():
        raise FileNotFoundError(f"missing Terminal-Bench prompt data: {prompt_path}")
    rows = [json.loads(line) for line in prompt_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not rows:
        raise ValueError("Terminal-Bench prompt data is empty")
    for index, row in enumerate(rows):
        if not isinstance(row.get("prompt"), list):
            raise ValueError(f"Terminal-Bench prompt row {index} has no chat-message prompt")
        metadata = row.get("metadata")
        if not isinstance(metadata, dict) or not isinstance(metadata.get("task_id"), str):
            raise ValueError(f"Terminal-Bench prompt row {index} has no metadata.task_id")

    for output_path in (Path(args.actor_save), Path(args.critic_save), Path(args.dump_details)):
        if output_path.exists() and any(output_path.iterdir()):
            raise FileExistsError(
                f"refusing to mix this fresh E2E gate with existing evidence in {output_path}"
            )

    contract_sha256 = _contract_sha256(args)
    checker = U.repo_base_dir / "tools" / "probes" / "check_sao_qwen35_raw_checkpoints.py"
    checker_env = os.environ.copy()
    checker_env["PYTHONPATH"] = os.pathsep.join(
        dict.fromkeys(
            entry
            for entry in (
                args.megatron_path,
                str(U.repo_base_dir),
                *checker_env.get("PYTHONPATH", "").split(os.pathsep),
            )
            if entry
        )
    )
    subprocess.run(
        [
            sys.executable,
            str(checker),
            "--actor",
            args.ref_load,
            "--critic",
            args.critic_load,
            "--hf",
            args.hf_checkpoint,
        ],
        check=True,
        env=checker_env,
    )
    return contract_sha256


def prepare(args: ScriptArgs) -> None:
    U.convert_checkpoint(
        model_name=args.model_name,
        megatron_model_type=args.megatron_model_type,
        num_gpus_per_node=args.checkpoint_conversion_gpus,
        dir_dst=str(Path(args.ref_load).parent),
        hf_checkpoint=args.hf_checkpoint,
        megatron_path=args.megatron_path,
    )


def execute(args: ScriptArgs) -> None:
    contract_sha256 = _preflight(args)
    print(f"SAO critic contract SHA-256: {contract_sha256}", flush=True)

    checkpoint_args = (
        f"--hf-checkpoint {_q(args.hf_checkpoint)} "
        f"--ref-load {_q(args.ref_load)} "
        f"--critic-load {_q(args.critic_load)} "
        f"--critic-value-pretrain-contract-sha256 {contract_sha256} "
        f"--save {_q(args.actor_save)} "
        f"--critic-save {_q(args.critic_save)} "
        "--save-interval 1 "
        "--megatron-to-hf-mode raw "
        "--model-name qwen3_5 "
        "--dist-ckpt-strictness raise_unexpected "
        "--no-load-optim --no-load-rng --no-save-optim --no-save-rng --finetune "
    )

    rollout_args = (
        f"--prompt-data {_q(args.prompt_data)} "
        "--input-key prompt --metadata-key metadata "
        f"--num-rollout {args.num_rollout} "
        "--rollout-batch-size 1 --n-samples-per-prompt 1 "
        "--over-sampling-batch-size 2 --num-steps-per-rollout 1 "
        "--global-batch-size 1 --micro-batch-size 1 "
        "--rollout-temperature 0.7 "
        f"--rollout-max-context-len {args.max_seq_len} "
        f"--rollout-max-response-len {args.rollout_max_response_len} "
        f"--max-seq-len {args.max_seq_len} "
        f"--dump-details {_q(args.dump_details)} "
    )

    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant "
        "--weight-decay 0.0 --adam-beta1 0.9 --adam-beta2 0.98 "
    )

    topology_args = (
        f"--num-gpus-per-node {args.num_gpus_per_node} "
        "--actor-num-nodes 1 --actor-num-gpus-per-node 1 "
        "--critic-num-nodes 1 --critic-num-gpus-per-node 1 "
        "--rollout-num-gpus 1 --rollout-num-gpus-per-engine 1 "
        "--tensor-model-parallel-size 1 --pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 --expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
    )

    sglang_args = (
        "--sglang-mem-fraction-static 0.50 "
        "--sglang-reasoning-parser qwen3 --sglang-tool-call-parser qwen3_coder "
        f"--sglang-context-length {args.max_seq_len} "
        "--sglang-max-running-requests 1 --sglang-disable-cuda-graph "
        "--sglang-cuda-graph-backend-prefill disabled "
        "--sglang-chunked-prefill-size 4096 "
        "--sglang-mamba-scheduler-strategy extra_buffer "
        "--sglang-enable-deterministic-inference "
        "--rollout-engine-base-port 21000 --sglang-router-port 23000 "
        "--sglang-router-prometheus-port 23001 --train-master-base-port 25000 "
        "--update-weight-buffer-size 1073741824 "
        "--update-weight-transfer-mode broadcast "
    )

    agent_args = C.agent_args(
        "qwen35",
        agent_function="openenv_subprocess_agent_function.run",
        dynamic_sampling_filter_path="openenv_generate.check_terminal_bench_episode",
    )
    chat_template_args = (
        f"--apply-chat-template-kwargs {_q('{\"clear_thinking\":false}')} "
    )
    runtime_args = (
        "--sao-online-recipe coding "
        "--max-position-embeddings 262144 "
        f"--seq-length {args.max_seq_len} "
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--attention-backend flash --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --bf16 "
        f"--seed {args.seed} --rollout-seed {args.seed} "
        "--distributed-timeout-minutes 30 --pin-rollout-manager-to-head "
        "--wandb-mode disabled "
    )

    train_args = (
        checkpoint_args
        + rollout_args
        + optimizer_args
        + topology_args
        + sglang_args
        + agent_args
        + chat_template_args
        + runtime_args
    )

    extra_env_vars = C.base_env_vars(args, str(SCRIPT_DIR), args.megatron_path, U.repo_base_dir)
    C.apply_optional_env_vars(extra_env_vars, args)
    extra_env_vars.update(
        {
            # The actor-to-SGLang publication only spans two ranks here.  Do
            # not force NVLink SHARP merely because the host has NVLink: some
            # provisioned NVSwitch nodes expose the links but cannot allocate
            # NVLS multicast memory.  Ordinary NCCL/NVLink transport remains
            # available with NVLS disabled.
            "NCCL_NVLS_ENABLE": "0",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PYTHONUNBUFFERED": "1",
            "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True",
            "RAY_DEDUP_LOGS": "0",
            "OPENENV_NATIVE_EVALUATE": "1",
            "OPENENV_MESSAGE_TIMEOUT_S": "900",
            "OPENENV_AGENT_PYTHON": "/root/openenv-venv/bin/python",
            "HF_HUB_DISABLE_TELEMETRY": "1",
        }
    )

    U.execute_train(
        train_args=train_args,
        config=args,
        num_gpus_per_node=args.num_gpus_per_node,
        megatron_model_type=args.megatron_model_type,
        megatron_path=args.megatron_path,
        extra_env_vars=extra_env_vars,
    )


@U.dataclass_cli
def main(args: ScriptArgs) -> None:
    C.cleanup()
    if not args.skip_prepare:
        prepare(args)
    execute(args)


if __name__ == "__main__":
    typer.run(main)
