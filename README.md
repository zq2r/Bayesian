# BayesianPRM Training

This directory contains the training code for BayesianPRM. Training has
two stages:

1. `train_ensemble_prm.sh` trains the ensemble reward hypotheses.
2. `train_bayesian_prm.sh` loads the ensemble checkpoint, freezes it, and
   trains the Bayesian belief head.

The metadata JSON and image root are supplied externally through `META_PATH`.

## Requirements

Use the same CUDA, PyTorch, Transformers, DeepSpeed, PEFT, FlashAttention,
TorchVision, Pillow, OpenCV, Decord, `timm`, `einops`, and `wandb` versions as
the main InternVL environment. Install this directory as a source package:

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
```

The base model checkpoint is also external. Set `MODEL_PATH` to it and set
`META_PATH` to the VisualPRM metadata JSON before launching either stage. If
the metadata uses relative `root` or `annotation` paths, set `DATA_ROOT` to
the directory containing the dataset (or an ancestor containing that path).

## Training

Run the ensemble stage first:

```bash
META_PATH=/path/to/meta_visualprm400k.json \
DATA_ROOT=/path/to/dataset-project \
MODEL_PATH=/path/to/InternVL3-8B \
bash scripts/train_ensemble_prm.sh
```

Then run BayesianPRM using the resulting ensemble checkpoint directory:

```bash
META_PATH=/path/to/meta_visualprm400k.json \
DATA_ROOT=/path/to/dataset-project \
ENSEMBLE_OUTPUT_DIR=/path/to/ensemble-output \
bash scripts/train_bayesian_prm.sh
```

For a short smoke test, set `MAX_STEPS=1` and reduce `BATCH_SIZE` and
`PER_DEVICE_BATCH_SIZE` as needed. On GPUs with limited memory, also set
`FREEZE_LLM=True FREEZE_MLP=True` for the ensemble stage; the default values
remain suitable for the full training run.

The Bayesian stage uses the complementary `belief` split when
`PRM_DATA_SPLIT_ENABLE=True`; keep the split ratio and seed identical between
the two stages.

## Evaluate Process Rewards

`eval/process_reward.py` scores every `<prm>` step in each candidate solution.
The input is a JSON list; each item contains an image path, a question, and
`solutions_splits` (a list of candidate solutions, each a list of step strings):

```json
[
  {
    "id": "example-1",
    "image": "example.png",
    "question": "What is the answer?",
    "solutions_splits": [["First step.", "Second step."], ["Alternative step."]]
  }
]
```

```bash
export PYTHONPATH="$PWD/src:$PYTHONPATH"
python eval/process_reward.py \
  --checkpoint /path/to/bayesian-checkpoint \
  --annotation /path/to/rollout-annotations.json \
  --image-root /path/to/images \
  --output /path/to/process-rewards.json \
  --dynamic
```

The output is a JSON list with `prm_scores` aligned to
`solutions_splits`: one final reward in `[0, 1]` per step. Each item also contains
`prm_mu_rel` (reward before conservatism), `prm_mu_heads` (per-head rewards),
`prm_rel_weights`, and `prm_post_weights`. For each step and head `m`, the
calculation matches the training checkpoint's diagnostics:

```text
mu_m = sigmoid(ensemble_logit_m)
alpha_rel = softmax(belief_logits)
alpha_post = softmax(log(alpha_rel) - mu / beta_2)
process_reward = sum_m alpha_post_m * mu_m
```

When conservatism is disabled, `alpha_post = alpha_rel`. The evaluator uses
`belief_use_conservatism` and `belief_conservatism_beta` from the checkpoint by
default. Use `--belief-use-conservatism true|false` or
`--belief-conservatism-beta VALUE` to compare settings without retraining.
Each output item records the effective setting and beta.

## Test-Time Scaling

`eval/test_time_scaling.py` evaluates best-of-N selection over an existing
candidate pool. It accepts the `prm_scores` written by `process_reward.py`, or
the Bayesian head-level fields `prm_mu_heads` and `prm_rel_weights`. The
dataset and generation model are deliberately left as injection points; only
candidate generation and PRM scoring need to be connected for a new task.

The default protocol is the same 20-repeat random-subset evaluation used by
the original TTS code. Every repeat stores selected candidate indices, and
labeled input additionally reports selection accuracy and oracle pass rate.

```bash
python eval/test_time_scaling.py \
  --input-json /path/to/scored-rollouts.json \
  --output-json /path/to/tts-results.json \
  --n-grid 1,2,4,8,16 \
  --repeats 20
```

Input records must contain `solutions_splits` and aligned PRM scores. Add a
`labels` list to compute accuracy; omit it when only selected candidates are
needed. Set `MODEL`, `DATASET`, and implement `generate_candidates` when
wiring task-specific generation.

## Calibration Against Monte-Carlo Labels

`eval/calibration.py` compares terminal BayesianPRM probabilities with the
rollout success probability `K/N`. It reports Brier Score, Positive Brier
Score, ECE, and AverageCE, plus the existing AdaptiveCE diagnostic. The
prediction file can use the `prm_mu_rel` and `prm_scores` fields produced by
inference; the latter is the conservatism-aware final prediction. The model
and dataset are intentionally not loaded by this evaluator.

```bash
python eval/calibration.py \
  --predictions /path/to/bayesian-inference.json \
  --mc-labels /path/to/mc-labels.jsonl \
  --output-json /path/to/calibration.json \
  --output-csv /path/to/calibration.csv \
  --num-bins 10
```

MC records require `prefix_id` and either `success_prob`,
`target_success_prob`, `mc_correct`/`mc_total`, or `rollout_labels`. Prediction
and MC prefix IDs are checked before any metric is computed.
