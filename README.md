# BayesianPRM Training

This directory contains the training code for BayesianPRM only. Training has
two stages:

1. `train_ensemble_prm.sh` trains the ensemble reward hypotheses.
2. `train_bayesian_prm.sh` loads the ensemble checkpoint, freezes it, and
   trains the Bayesian belief head.

Evaluation scripts, generated outputs, and datasets are intentionally omitted.
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
