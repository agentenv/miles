# SAO online training

Algorithm settings follow [Single-Rollout Asynchronous Optimization for
Agentic Reinforcement Learning](https://arxiv.org/abs/2607.07508).

The first online validation uses Miles' centralized publication path:

```text
SecRLEnv + SGLang
  -> one authenticated rollout per prompt, including behavior logprobs
  -> fixed critic targets + two critic updates
  -> refreshed critic values
  -> observation-skipping, length-adaptive GAE + actor DIS update
  -> direct actor-to-SGLang weight publication
```

The critic is trainer-only. Its parameters, optimizer state, and value head are
never published to SGLang. This is the temporary replacement for streaming
DiLoCo: do not set `--external-policy-sync-path` for the centralized probe.
When the streaming synchronizer is ready, it replaces only the policy
synchronization boundary; the SecRLEnv, SAO actor, and critic paths stay the
same. A compatible synchronizer must declare `supports_critic = True`.

Launch the temporary probe through
`tools/probes/train_sao_secrlenv.py`. It accepts the normal Miles flags plus a
strict, hash-bindable `--sao-secrlenv-context` JSON file. The file contains no
daemon bearer token; keep that secret in the existing private token file/env
path used by the SecRLEnv adapter.

The context schema is:

```json
{
  "schema": "miles.sao-secrlenv.v1",
  "model": "Qwen/Qwen3.5-4B",
  "data": "/absolute/path/dataset.jsonl",
  "data_sha256": "<sha256>",
  "base_model_revision": "<immutable-hf-revision>",
  "rollout_model_revision": "<immutable-hf-revision>",
  "data_revision": null,
  "layout_hash": "<sha256>",
  "lora_config_hash": "full-parameter-centralized",
  "reward_sha256": "<sha256>",
  "dynamic_sampling_max_replacements": 0,
  "secrlenv_max_infrastructure_replacements": 1,
  "completed_groups_path": "/absolute/run/completed-groups.pt",
  "event_tape": "/absolute/run/events.jsonl",
  "learner_id": 0
}
```

The entrypoint verifies the context digest when
`--sao-secrlenv-context-sha256` is supplied, re-hashes the dataset, binds the
strict Yeto rollout identity, and rejects an external synchronizer so this gate
cannot accidentally test two new systems at once.

## Required online flags

Use the normal model, SecRLEnv rollout, reward, and topology flags, then add:

```bash
--sao-online-recipe coding \
--n-samples-per-prompt 1 \
--critic-load /checkpoints/qwen35-4b-value \
--critic-value-pretrain-contract-sha256 <contract-sha256> \
--critic-save /checkpoints/qwen35-4b-sao-critic
```

The `coding` recipe applies the paper-backed structure:

- DIS with strict ratio bounds `(0.2, 4.0)` (`eps_low=0.8`, `eps_high=3.0`);
- one rollout per prompt;
- observation-skipping GAE;
- policy lambda `1 - 1 / (1.5 * active_action_tokens)`;
- critic targets with lambda 1;
- two critic optimizer updates per rollout batch;
- actor learning rate `1e-6` with no auxiliary KL or entropy loss;
- critic learning rate `5e-6`, ten-iteration warmup, and frozen attention.

Use `--sao-online-recipe reasoning` for the paper's reasoning DIS bounds
`(0.7, 6.0)`. Topology, batch size, checkpoint paths, and SecRLEnv hooks remain
deployment settings rather than algorithm defaults.

Do not combine DIS with TIS, OPSM, or the old mismatch-correction path. DIS
uses the rollout engine's per-token behavior logprobs directly. Before learner
training, Miles now rejects a batch if any sample has missing/non-finite
behavior logprobs, a malformed response mask, no trainable action token, or a
non-finite reward. Environment-observation tokens may remain in the response;
their loss mask is zero and GAE bridges from one action token to the next.

## Small centralized probe

For Qwen3.5-4B on one 4xH200 host, use three GPUs initially:

| GPU allocation | Role |
| --- | --- |
| 1 GPU | full-parameter actor, TP1/PP1 |
| 1 GPU | full-parameter critic, TP1/PP1 |
| 1 GPU | SGLang inference, TP1 |
| 1 GPU | spare for OOM recovery or a second inference engine |

The key topology flags are:

```bash
--actor-num-nodes 1 \
--actor-num-gpus-per-node 1 \
--critic-num-nodes 1 \
--critic-num-gpus-per-node 1 \
--rollout-num-gpus 1 \
--rollout-num-gpus-per-engine 1 \
--tensor-model-parallel-size 1 \
--pipeline-model-parallel-size 1
```

This is non-colocated. The placement group therefore reserves three distinct
GPUs. If the full-parameter actor or critic does not fit a single GPU, move to
2 actor + 2 critic + 1 inference GPUs on an 8-GPU host; do not change the SAO
algorithm to work around a memory problem.

For the first smoke, use one rollout batch and one learner batch, disable W&B,
and exit after one rollout. Keep the real SecRLEnv generate/reward hooks and
DinD debugging enabled so the smoke exercises the same online boundary as the
eventual run.

## Success gates

The centralized online smoke passes only when all of these hold:

1. every SecRLEnv reward is authenticated and tied to the selected task pack;
2. rollout behavior logprobs align exactly with response tokens and masks;
3. the pretrained critic contract digest loads and the value objective matches;
4. actor DIS loss, critic loss, advantages, returns, and gradients are finite;
5. two critic updates complete, refreshed values reach the actor, and one actor
   update completes;
6. the updated actor is published to SGLang and the next rollout identifies the
   new policy;
7. the critic checkpoint advances independently and no critic tensors are sent
   to inference.

After this passes, the next integration test runs the same recipe with the
manager's streaming DiLoCo factory supplied through
`--external-policy-sync-path`. Only synchronization-specific assertions should
change. The streaming layer must either pin one inference policy for an entire
multi-turn episode or retain the exact behavior logprob for every token across
mid-episode policy changes. It must not replace those values with scores from a
newer policy; DIS deliberately consumes the recorded behavior probabilities.
