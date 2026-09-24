#!/usr/bin/env bash
set -euo pipefail
set -x

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# ============================================================
# BayesianPRM Stage-2 Training
#
# This script ONLY trains the Bayesian belief network.
#
# The target EnsemblePRM checkpoint is selected by:
#   - ENSEMBLE_PRM_USE_PRIOR_NETWORK
#   - ENSEMBLE_PRM_NUM_HEADS
#   - ENSEMBLE_PRM_PRIOR_SCALE
#
# These selector variables are used ONLY to locate and verify the
# EnsemblePRM checkpoint. The actual frozen ensemble architecture
# is restored from checkpoint/config.json.
# ============================================================

# ============================================================
# Offline
# ============================================================
export HF_HUB_OFFLINE=${HF_HUB_OFFLINE:-1}
export TRANSFORMERS_OFFLINE=${TRANSFORMERS_OFFLINE:-1}
export HF_DATASETS_OFFLINE=${HF_DATASETS_OFFLINE:-1}

# ============================================================
# Distributed / device
# ============================================================
export CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-"0,1,2,3"}

GPUS=${GPUS:-4}
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
NPROC_PER_NODE=${NPROC_PER_NODE:-${GPUS}}
export MASTER_PORT=${MASTER_PORT:-4320}

model_name=${model_name:-"InternVL3-8B"}

# True = much smaller checkpoints, but optimizer/scheduler state
# is not available for a strict training resume.
SAVE_ONLY_MODEL=${SAVE_ONLY_MODEL:-True}
RESUME_BELIEF_TRAINING=${RESUME_BELIEF_TRAINING:-0}
# ============================================================
# Ensemble checkpoint selector
#
# IMPORTANT:
# These values do NOT redefine the loaded EnsemblePRM.
# They only select the experiment directory whose latest
# checkpoint will be loaded.
# ============================================================
ENSEMBLE_PRM_USE_PRIOR_NETWORK=${ENSEMBLE_PRM_USE_PRIOR_NETWORK:-True}
ENSEMBLE_PRM_NUM_HEADS=${ENSEMBLE_PRM_NUM_HEADS:-10}
ENSEMBLE_PRM_PRIOR_SCALE=${ENSEMBLE_PRM_PRIOR_SCALE:-100}
BELIEF_BETA_KL=${BELIEF_BETA_KL:-0.05}

# ============================================================
# Bayesian belief network
# ============================================================

# Belief-network architecture.
BELIEF_HIDDEN_DIM=${BELIEF_HIDDEN_DIM:-256}
BELIEF_DROPOUT=${BELIEF_DROPOUT:-0.0}
BELIEF_USE_REWARD_PROBS=${BELIEF_USE_REWARD_PROBS:-True}


# False:
#   use the full Binomial log likelihood
#       K log(mu) + (N-K) log(1-mu)
#
# True:
#   divide the likelihood by N.
BELIEF_LOGLIK_NORMALIZE_BY_N=${BELIEF_LOGLIK_NORMALIZE_BY_N:-False}

# ============================================================
# Conservatism-aware belief calibration
#
# alpha_final_m
#   ∝ alpha_rel_m * exp(-reward_m / beta_2)
#
# NOTE:
# beta_2 does NOT affect belief-network training gradients.
# It is stored in the checkpoint config for downstream use.
# ============================================================
BELIEF_USE_CONSERVATISM=${BELIEF_USE_CONSERVATISM:-True}
BELIEF_CONSERVATISM_BETA=${BELIEF_CONSERVATISM_BETA:-0.1}

# Use exactly the same directory naming convention as
# visualprm400k_train_ensemble_prm.sh.
if [ "${ENSEMBLE_PRM_USE_PRIOR_NETWORK}" = "True" ] || \
   [ "${ENSEMBLE_PRM_USE_PRIOR_NETWORK}" = "true" ]; then
    DEFAULT_ENSEMBLE_OUTPUT_DIR="${REPO_ROOT}/log/ensemble-prior-head${ENSEMBLE_PRM_NUM_HEADS}-scale${ENSEMBLE_PRM_PRIOR_SCALE}-${model_name}-visualprm400k"
    DEFAULT_BAYESIAN_OUTPUT_DIR="${REPO_ROOT}/log/bayesian-prior-head${ENSEMBLE_PRM_NUM_HEADS}-scale${ENSEMBLE_PRM_PRIOR_SCALE}-beta${BELIEF_BETA_KL}-${model_name}-visualprm400k"
else
    DEFAULT_ENSEMBLE_OUTPUT_DIR="${REPO_ROOT}/log/ensemble-head${ENSEMBLE_PRM_NUM_HEADS}-scale${ENSEMBLE_PRM_PRIOR_SCALE}-${model_name}-visualprm400k"
    DEFAULT_BAYESIAN_OUTPUT_DIR="${REPO_ROOT}/log/bayesian-head${ENSEMBLE_PRM_NUM_HEADS}-scale${ENSEMBLE_PRM_PRIOR_SCALE}-beta${BELIEF_BETA_KL}-${model_name}-visualprm400k"
fi

ENSEMBLE_OUTPUT_DIR=${ENSEMBLE_OUTPUT_DIR:-"${DEFAULT_ENSEMBLE_OUTPUT_DIR}"}
BAYESIAN_OUTPUT_DIR=${BAYESIAN_OUTPUT_DIR:-"${DEFAULT_BAYESIAN_OUTPUT_DIR}"}

# Optional escape hatch:
# if ENSEMBLE_CHECKPOINT is explicitly supplied, use it;
# otherwise automatically select the latest checkpoint under
# ENSEMBLE_OUTPUT_DIR.
ENSEMBLE_CHECKPOINT=${ENSEMBLE_CHECKPOINT:-""}

META_PATH=${META_PATH:?Set META_PATH to the external VisualPRM metadata JSON.}
DEEPSPEED_CONFIG=${DEEPSPEED_CONFIG:-"${REPO_ROOT}/configs/zero_stage3_config.json"}

mkdir -p "${BAYESIAN_OUTPUT_DIR}"

# ============================================================
# Resolve latest EnsemblePRM checkpoint
# ============================================================
if [ -z "${ENSEMBLE_CHECKPOINT}" ]; then
    echo "Searching EnsemblePRM checkpoint..."
    echo "  use_prior_network: ${ENSEMBLE_PRM_USE_PRIOR_NETWORK}"
    echo "  num_heads:         ${ENSEMBLE_PRM_NUM_HEADS}"
    echo "  prior_scale:       ${ENSEMBLE_PRM_PRIOR_SCALE}"
    echo "  directory:         ${ENSEMBLE_OUTPUT_DIR}"

    if [ ! -d "${ENSEMBLE_OUTPUT_DIR}" ]; then
        echo "ERROR: EnsemblePRM experiment directory does not exist:"
        echo "  ${ENSEMBLE_OUTPUT_DIR}"
        exit 1
    fi

    ENSEMBLE_CHECKPOINT="$(
        find "${ENSEMBLE_OUTPUT_DIR}" \
            -maxdepth 1 \
            -type d \
            -name "checkpoint-*" \
            2>/dev/null \
        | sort -V \
        | tail -n 1
    )"

    if [ -z "${ENSEMBLE_CHECKPOINT}" ]; then
        echo "ERROR: No EnsemblePRM checkpoint found under:"
        echo "  ${ENSEMBLE_OUTPUT_DIR}"
        exit 1
    fi
else
    echo "Using explicitly specified ENSEMBLE_CHECKPOINT:"
    echo "  ${ENSEMBLE_CHECKPOINT}"
fi

if [ ! -d "${ENSEMBLE_CHECKPOINT}" ]; then
    echo "ERROR: EnsemblePRM checkpoint does not exist:"
    echo "  ${ENSEMBLE_CHECKPOINT}"
    exit 1
fi

if [ ! -f "${ENSEMBLE_CHECKPOINT}/config.json" ]; then
    echo "ERROR: config.json not found in EnsemblePRM checkpoint:"
    echo "  ${ENSEMBLE_CHECKPOINT}"
    exit 1
fi

echo "Resolved EnsemblePRM checkpoint:"
echo "  ${ENSEMBLE_CHECKPOINT}"

# ============================================================
# Inspect + verify EnsemblePRM checkpoint
#
# The selector is checked against config.json here.
# The training code itself still uses checkpoint config as the
# single source of truth for the frozen ensemble architecture.
# ============================================================
python - \
    "${ENSEMBLE_CHECKPOINT}" \
    "${ENSEMBLE_PRM_USE_PRIOR_NETWORK}" \
    "${ENSEMBLE_PRM_NUM_HEADS}" \
    "${ENSEMBLE_PRM_PRIOR_SCALE}" <<'PY'
import json
import math
import os
import sys

ckpt = sys.argv[1]
selector_prior_raw = sys.argv[2]
selector_num_heads = int(sys.argv[3])
selector_prior_scale = float(sys.argv[4])

selector_use_prior = selector_prior_raw.lower() in {
    "true",
    "1",
    "yes",
    "y",
}

config_path = os.path.join(ckpt, "config.json")
with open(config_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

required = [
    "ensemble_prm_num_heads",
    "ensemble_prm_hidden_dim",
    "ensemble_prm_dropout",
    "ensemble_prm_use_prior_network",
    "ensemble_prm_prior_scale",
]

missing = [k for k in required if k not in cfg]
if missing:
    raise RuntimeError(
        "The resolved checkpoint is missing required EnsemblePRM "
        f"configuration fields: {missing}"
    )

actual_num_heads = int(cfg["ensemble_prm_num_heads"])
actual_use_prior = bool(cfg["ensemble_prm_use_prior_network"])
actual_prior_scale = float(cfg["ensemble_prm_prior_scale"])

errors = []

if actual_num_heads != selector_num_heads:
    errors.append(
        f"num_heads mismatch: selector={selector_num_heads}, "
        f"checkpoint={actual_num_heads}"
    )

if actual_use_prior != selector_use_prior:
    errors.append(
        f"use_prior_network mismatch: selector={selector_use_prior}, "
        f"checkpoint={actual_use_prior}"
    )

if not math.isclose(
    actual_prior_scale,
    selector_prior_scale,
    rel_tol=1e-9,
    abs_tol=1e-12,
):
    errors.append(
        f"prior_scale mismatch: selector={selector_prior_scale}, "
        f"checkpoint={actual_prior_scale}"
    )

if errors:
    raise RuntimeError(
        "Resolved EnsemblePRM checkpoint does not match the requested "
        "selector:\n  - " + "\n  - ".join(errors)
    )

print("========== Loaded EnsemblePRM config ==========")
print("checkpoint:", ckpt)
for key in required:
    print(f"{key}: {cfg[key]}")
print(
    "ensemble_prm_bootstrap_prob:",
    cfg.get("ensemble_prm_bootstrap_prob", "<not stored>"),
)
print("checkpoint prm_loss_type:", cfg.get("prm_loss_type", "<missing>"))
print("Selector/config verification: PASSED")
print("================================================")
PY

# ============================================================
# PRM data split
#
# The EnsemblePRM checkpoint must have been trained on the
# complementary "ensemble" subset using exactly the same:
#
#   META_PATH
#   split ratio
#   split seed
#
# BayesianPRM uses the "belief" subset.
# ============================================================
PRM_DATA_SPLIT_ENABLE=${PRM_DATA_SPLIT_ENABLE:-True}
PRM_DATA_SPLIT_RATIO=${PRM_DATA_SPLIT_RATIO:-0.8}
PRM_DATA_SPLIT_SEED=${PRM_DATA_SPLIT_SEED:-42}

# ============================================================
# Optimization
# ============================================================
BATCH_SIZE=${BATCH_SIZE:-512}
PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE:-2}

DENOM=$((PER_DEVICE_BATCH_SIZE * GPUS))
if (( BATCH_SIZE % DENOM != 0 )); then
    echo "ERROR: BATCH_SIZE must be divisible by"
    echo "       PER_DEVICE_BATCH_SIZE * GPUS."
    echo
    echo "BATCH_SIZE=${BATCH_SIZE}"
    echo "PER_DEVICE_BATCH_SIZE=${PER_DEVICE_BATCH_SIZE}"
    echo "GPUS=${GPUS}"
    exit 1
fi

GRADIENT_ACC=$((BATCH_SIZE / DENOM))

LEARNING_RATE=${LEARNING_RATE:-1e-5}
WEIGHT_DECAY=${WEIGHT_DECAY:-0.05}
NUM_TRAIN_EPOCHS=${NUM_TRAIN_EPOCHS:-1}
WARMUP_RATIO=${WARMUP_RATIO:-0.05}

SAVE_STEPS=${SAVE_STEPS:-100}
SAVE_TOTAL_LIMIT=${SAVE_TOTAL_LIMIT:-1}



# ============================================================
# Optional short smoke test
# ============================================================
MAX_STEPS=${MAX_STEPS:-""}
MAX_STEPS_ARGS=()

if [ -n "${MAX_STEPS}" ]; then
    MAX_STEPS_ARGS+=(--max_steps "${MAX_STEPS}")
fi

# ============================================================
# Resume Bayesian belief training
# ============================================================
is_complete_checkpoint() {
    local ckpt="$1"
    local ds_step_dir=""

    if [ ! -s "${ckpt}/trainer_state.json" ]; then
        return 1
    fi

    ds_step_dir=$(
        find "${ckpt}" \
            -maxdepth 1 \
            -type d \
            -name "global_step*" \
            -print -quit \
            2>/dev/null
    )

    if [ -z "${ds_step_dir}" ]; then
        return 1
    fi

    if ! find "${ds_step_dir}" \
        -maxdepth 1 \
        -type f \
        -name "*model_states.pt" \
        -size +0c \
        -print -quit \
        2>/dev/null | grep -q .; then
        return 1
    fi

    if ! find "${ds_step_dir}" \
        -maxdepth 1 \
        -type f \
        -name "*optim_states.pt" \
        -size +0c \
        -print -quit \
        2>/dev/null | grep -q .; then
        return 1
    fi

    return 0
}

# Fresh BayesianPRM always starts from the resolved EnsemblePRM.
BAYESIAN_MODEL_PATH="${ENSEMBLE_CHECKPOINT}"
BAYESIAN_RESUME_ARGS=()

if [ "${RESUME_BELIEF_TRAINING}" = "1" ]; then

    LATEST_BAYESIAN_CHECKPOINT=$(
        find "${BAYESIAN_OUTPUT_DIR}" \
            -maxdepth 1 \
            -type d \
            -name "checkpoint-*" \
            2>/dev/null \
        | sort -V \
        | tail -n 1
    )

    if [ -z "${LATEST_BAYESIAN_CHECKPOINT}" ]; then
        echo "============================================================"
        echo "[INFO] RESUME_BELIEF_TRAINING=1"
        echo "[INFO] No BayesianPRM checkpoint found."
        echo "[INFO] Start fresh belief training from EnsemblePRM:"
        echo "       ${ENSEMBLE_CHECKPOINT}"
        echo "============================================================"

    elif is_complete_checkpoint "${LATEST_BAYESIAN_CHECKPOINT}"; then
        echo "============================================================"
        echo "[INFO] Found complete BayesianPRM checkpoint:"
        echo "       ${LATEST_BAYESIAN_CHECKPOINT}"
        echo "[INFO] Resume belief training from this checkpoint."
        echo "============================================================"

        BAYESIAN_MODEL_PATH="${LATEST_BAYESIAN_CHECKPOINT}"

        BAYESIAN_RESUME_ARGS+=(
            --resume_from_checkpoint "${LATEST_BAYESIAN_CHECKPOINT}"
        )

    else
        echo "============================================================"
        echo "[WARNING] Latest BayesianPRM checkpoint is incomplete:"
        echo "          ${LATEST_BAYESIAN_CHECKPOINT}"
        echo "[WARNING] Cannot perform true resume."
        echo "[INFO] Removing old Bayesian checkpoints."
        echo "[INFO] Restart belief training from EnsemblePRM:"
        echo "       ${ENSEMBLE_CHECKPOINT}"
        echo "============================================================"

        rm -rf "${BAYESIAN_OUTPUT_DIR}"/checkpoint-*

        BAYESIAN_MODEL_PATH="${ENSEMBLE_CHECKPOINT}"
    fi

else
    echo "============================================================"
    echo "[INFO] RESUME_BELIEF_TRAINING=0"
    echo "[INFO] Removing old Bayesian checkpoints."
    echo "[INFO] Start fresh belief training from EnsemblePRM:"
    echo "       ${ENSEMBLE_CHECKPOINT}"
    echo "============================================================"

    rm -rf "${BAYESIAN_OUTPUT_DIR}"/checkpoint-*

    BAYESIAN_MODEL_PATH="${ENSEMBLE_CHECKPOINT}"
fi

# ============================================================
# Runtime
# ============================================================
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}:${PYTHONPATH:-}"
export TF_CPP_MIN_LOG_LEVEL=3
export LAUNCHER=pytorch

if [ -z "${CUDA_HOME:-}" ] && command -v nvcc >/dev/null 2>&1; then
    export CUDA_HOME="$(dirname "$(dirname "$(command -v nvcc)")")"
fi

if [ -n "${CUDA_HOME:-}" ]; then
    export PATH="${CUDA_HOME}/bin:${PATH}"
fi

CUDA_TARGETS_INCLUDE="${CUDA_HOME:-}/targets/x86_64-linux/include"
CUDA_CCCL_INCLUDE="${CUDA_HOME:-}/targets/x86_64-linux/include/cccl"

if [ -d "${CUDA_TARGETS_INCLUDE}" ]; then
    export CPATH="${CUDA_TARGETS_INCLUDE}:${CPATH:-}"
    export CPLUS_INCLUDE_PATH="${CUDA_TARGETS_INCLUDE}:${CPLUS_INCLUDE_PATH:-}"
fi

if [ -d "${CUDA_CCCL_INCLUDE}" ]; then
    export CPATH="${CUDA_CCCL_INCLUDE}:${CPATH:-}"
    export CPLUS_INCLUDE_PATH="${CUDA_CCCL_INCLUDE}:${CPLUS_INCLUDE_PATH:-}"
fi

export TORCH_EXTENSIONS_DIR=${TORCH_EXTENSIONS_DIR:-/inspire/hdd/global_user/zhouzhixiang-240107010008/qzj/cache/torch_extensions}
mkdir -p "${TORCH_EXTENSIONS_DIR}"

# ============================================================
# WandB
# ============================================================
export WANDB_MODE=${WANDB_MODE:-offline}
export WANDB_PROJECT=${WANDB_PROJECT:-"BayesianPRM"}

if [ "${ENSEMBLE_PRM_USE_PRIOR_NETWORK}" = "True" ] || \
   [ "${ENSEMBLE_PRM_USE_PRIOR_NETWORK}" = "true" ]; then
    DEFAULT_WANDB_NAME="bayesian-prior-head${ENSEMBLE_PRM_NUM_HEADS}-scale${ENSEMBLE_PRM_PRIOR_SCALE}-${model_name}-visualprm400k"
    DEFAULT_WANDB_RUN_GROUP="bayesian-prior-head${ENSEMBLE_PRM_NUM_HEADS}-scale${ENSEMBLE_PRM_PRIOR_SCALE}-${model_name}"
    DEFAULT_WANDB_TAGS="visualprm400k,bayesian,ensemble_prior,head${ENSEMBLE_PRM_NUM_HEADS},scale${ENSEMBLE_PRM_PRIOR_SCALE}"
else
    DEFAULT_WANDB_NAME="bayesian-head${ENSEMBLE_PRM_NUM_HEADS}-scale${ENSEMBLE_PRM_PRIOR_SCALE}-${model_name}-visualprm400k"
    DEFAULT_WANDB_RUN_GROUP="bayesian-head${ENSEMBLE_PRM_NUM_HEADS}-scale${ENSEMBLE_PRM_PRIOR_SCALE}-${model_name}"
    DEFAULT_WANDB_TAGS="visualprm400k,bayesian,head${ENSEMBLE_PRM_NUM_HEADS},scale${ENSEMBLE_PRM_PRIOR_SCALE}"
fi

export WANDB_NAME=${WANDB_NAME:-"${DEFAULT_WANDB_NAME}"}
export WANDB_RUN_GROUP=${WANDB_RUN_GROUP:-"${DEFAULT_WANDB_RUN_GROUP}"}
export WANDB_TAGS=${WANDB_TAGS:-"${DEFAULT_WANDB_TAGS}"}
export WANDB_DIR=${WANDB_DIR:-"${BAYESIAN_OUTPUT_DIR}/wandb"}

mkdir -p "${WANDB_DIR}"

# ============================================================
# Summary
# ============================================================
echo "================ BayesianPRM training ================"
echo "Ensemble selector:"
echo "  ENSEMBLE_PRM_USE_PRIOR_NETWORK: ${ENSEMBLE_PRM_USE_PRIOR_NETWORK}"
echo "  ENSEMBLE_PRM_NUM_HEADS: ${ENSEMBLE_PRM_NUM_HEADS}"
echo "  ENSEMBLE_PRM_PRIOR_SCALE: ${ENSEMBLE_PRM_PRIOR_SCALE}"
echo "ENSEMBLE_OUTPUT_DIR: ${ENSEMBLE_OUTPUT_DIR}"
echo "ENSEMBLE_CHECKPOINT: ${ENSEMBLE_CHECKPOINT}"
echo "BAYESIAN_MODEL_PATH: ${BAYESIAN_MODEL_PATH}"
echo "BAYESIAN_OUTPUT_DIR: ${BAYESIAN_OUTPUT_DIR}"
echo
echo "META_PATH: ${META_PATH}"
echo "PRM_DATA_SPLIT_ENABLE: ${PRM_DATA_SPLIT_ENABLE}"
echo "PRM_DATA_SPLIT_RATIO: ${PRM_DATA_SPLIT_RATIO}"
echo "PRM_DATA_SPLIT_SEED: ${PRM_DATA_SPLIT_SEED}"
echo
echo "BELIEF_HIDDEN_DIM: ${BELIEF_HIDDEN_DIM}"
echo "BELIEF_DROPOUT: ${BELIEF_DROPOUT}"
echo "BELIEF_USE_REWARD_PROBS: ${BELIEF_USE_REWARD_PROBS}"
echo "BELIEF_BETA_KL (beta_1): ${BELIEF_BETA_KL}"
echo "BELIEF_LOGLIK_NORMALIZE_BY_N: ${BELIEF_LOGLIK_NORMALIZE_BY_N}"
echo
echo "BELIEF_USE_CONSERVATISM: ${BELIEF_USE_CONSERVATISM}"
echo "BELIEF_CONSERVATISM_BETA (beta_2): ${BELIEF_CONSERVATISM_BETA}"
echo
echo "GPUS: ${GPUS}"
echo "BATCH_SIZE: ${BATCH_SIZE}"
echo "PER_DEVICE_BATCH_SIZE: ${PER_DEVICE_BATCH_SIZE}"
echo "GRADIENT_ACC: ${GRADIENT_ACC}"
echo "LEARNING_RATE: ${LEARNING_RATE}"
echo "NUM_TRAIN_EPOCHS: ${NUM_TRAIN_EPOCHS}"
echo "WANDB_NAME: ${WANDB_NAME}"
echo "========================================================"

# ============================================================
# Train Bayesian belief network
# ============================================================
python -m torch.distributed.run \
    --nnodes="${NNODES}" \
    --node_rank="${NODE_RANK}" \
    --master_addr="${MASTER_ADDR}" \
    --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${MASTER_PORT}" \
    "${REPO_ROOT}/src/internvl/train/train_bayesian_prm.py" \
    \
    --model_name_or_path "${BAYESIAN_MODEL_PATH}" \
    --prm_loss_type bayesian_prm \
    \
    --conv_style "internvl2_5" \
    --use_fast_tokenizer False \
    --force_image_size 448 \
    --max_dynamic_patch 6 \
    --down_sample_ratio 0.5 \
    --drop_path_rate 0.4 \
    --vision_select_layer -1 \
    --max_seq_length 8192 \
    --dynamic_image_size True \
    --use_thumbnail True \
    --ps_version v2 \
    \
    --meta_path "${META_PATH}" \
    --prm_data_split_enable "${PRM_DATA_SPLIT_ENABLE}" \
    --prm_data_split_ratio "${PRM_DATA_SPLIT_RATIO}" \
    --prm_data_split_seed "${PRM_DATA_SPLIT_SEED}" \
    --prm_data_split_part belief \
    \
    --belief_hidden_dim "${BELIEF_HIDDEN_DIM}" \
    --belief_dropout "${BELIEF_DROPOUT}" \
    --belief_beta_kl "${BELIEF_BETA_KL}" \
    --belief_use_reward_probs "${BELIEF_USE_REWARD_PROBS}" \
    --belief_loglik_normalize_by_n "${BELIEF_LOGLIK_NORMALIZE_BY_N}" \
    --belief_use_conservatism "${BELIEF_USE_CONSERVATISM}" \
    --belief_conservatism_beta "${BELIEF_CONSERVATISM_BETA}" \
    \
    --freeze_llm True \
    --freeze_mlp True \
    --freeze_backbone True \
    --grad_checkpoint True \
    \
    --output_dir "${BAYESIAN_OUTPUT_DIR}" \
    --overwrite_output_dir True \
    "${BAYESIAN_RESUME_ARGS[@]}" \
    \
    --do_train True \
    --bf16 True \
    --num_train_epochs "${NUM_TRAIN_EPOCHS}" \
    "${MAX_STEPS_ARGS[@]}" \
    --per_device_train_batch_size "${PER_DEVICE_BATCH_SIZE}" \
    --gradient_accumulation_steps "${GRADIENT_ACC}" \
    --learning_rate "${LEARNING_RATE}" \
    --weight_decay "${WEIGHT_DECAY}" \
    --warmup_ratio "${WARMUP_RATIO}" \
    --lr_scheduler_type cosine \
    \
    --dataloader_num_workers 4 \
    --group_by_length True \
    \
    --save_strategy steps \
    --save_steps "${SAVE_STEPS}" \
    --save_total_limit "${SAVE_TOTAL_LIMIT}" \
    --save_only_model "${SAVE_ONLY_MODEL}" \
    \
    --logging_steps 1 \
    --deepspeed "${DEEPSPEED_CONFIG}" \
    --report_to wandb \
    --run_name "${WANDB_NAME}" \
    \
    2>&1 | python -u -c '
import sys
from collections import deque
from pathlib import Path

log_file = Path(sys.argv[1])
max_lines = int(sys.argv[2])
flush_interval = int(sys.argv[3])

buf = deque(maxlen=max_lines)
log_file.parent.mkdir(parents=True, exist_ok=True)
log_file.write_text("", encoding="utf-8")

def dump():
    tmp = log_file.with_suffix(log_file.suffix + ".tmp")
    tmp.write_text("".join(buf), encoding="utf-8")
    tmp.replace(log_file)

for i, line in enumerate(sys.stdin, 1):
    sys.stdout.write(line)
    sys.stdout.flush()
    buf.append(line)

    if i % flush_interval == 0:
        dump()

dump()
' "${BAYESIAN_OUTPUT_DIR}/training_log.txt" 1000 100
