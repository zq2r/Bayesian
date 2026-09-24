# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------
#
# BayesianPRM and EnsemblePRM training
# ====================================
# Stage 1 trains an ensemble of frozen reward hypotheses. Stage 2 trains the
# Bayesian belief head on a disjoint subset while keeping the ensemble frozen.
#
# Training data contract:
# - PRM positions are '<prm>' placeholder tokens.
# - Each position carries a ratio label and count supervision.
# - Stage 1 uses ratio labels for ensemble reward hypotheses.
# - Stage 2 uses count likelihoods for the Bayesian reliability posterior.
#
# Data interface notes:
# - Datasets may provide a top-level prm_counts dict: {'k': [...], 'n': [...]} aligned with '<prm>' steps.
# - The preprocess builds three aligned channels at '<prm>' positions: labels (ratio), prm_counts_k, prm_counts_n.

import logging
import math
import os
import random
import sys
import traceback
import warnings
from copy import deepcopy
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, Literal, Optional

import numpy as np

try:
    import orjson as json
except:
    import json

import torch
import torch.distributed as dist
import torch.nn.functional as F
import transformers
from PIL import Image, ImageFile, PngImagePlugin, UnidentifiedImageError
from torch.utils.data import Dataset, Subset
from transformers import (AutoConfig, AutoModelForCausalLM, AutoTokenizer,
                          HfArgumentParser, Trainer, TrainerCallback,
                          TrainingArguments,
                          set_seed)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils.logging import (enable_default_handler,
                                        enable_explicit_format, set_verbosity)

from internvl.conversation import get_conv_template
from internvl.dist_utils import init_dist
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.internvl_chat import (InternVisionConfig,
                                          InternVisionModel,
                                          InternVLChatConfig)
from internvl.model.internvl_chat.modeling_bayesian_prm import \
    InternVLChatModel
from internvl.patch import (concat_pad_data_collator,
                            replace_internlm2_attention_class,
                            replace_llama_attention_class,
                            replace_llama_rmsnorm_with_fused_rmsnorm,
                            replace_phi3_attention_class,
                            replace_qwen2_attention_class,
                            replace_train_dataloader, replace_train_sampler)
from internvl.train.constants import (BOX_END_TOKEN, BOX_START_TOKEN,
                                      IMG_CONTEXT_TOKEN, IMG_END_TOKEN,
                                      IMG_START_TOKEN, PRM_TOKEN,
                                      QUAD_END_TOKEN, QUAD_START_TOKEN,
                                      REF_END_TOKEN, REF_START_TOKEN,
                                      REWARD_TOKENS)
from internvl.train.dataset import (ConcatDataset, TCSLoader,
                                    WeightedConcatDataset, build_transform,
                                    check_conversations_repetition,
                                    dynamic_preprocess, preprocess,
                                    preprocess_internlm,
                                    preprocess_mpt, preprocess_phi3)
from internvl.train.dataset_packed import PackedDataset, packed_collate_fn

# Try to import petrel_client for image loading, fallback to PIL if unavailable
try:
    from petrel_client.client import Client
    from petrel_client.common.config import Config

    has_tcs_loader = True
except ImportError as E:
    print('petrel_client is not installed. Using PIL to load images.')
    has_tcs_loader = False

# Set constants for image processing and logging
IGNORE_INDEX = -100
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True
MaximumDecompressedSize = 1024
MegaByte = 2**20
PngImagePlugin.MAX_TEXT_CHUNK = MaximumDecompressedSize * MegaByte

warnings.filterwarnings('ignore')
logger = logging.getLogger(__name__)

os.environ['TOKENIZERS_PARALLELISM'] = 'true'


def _resolve_external_data_path(path, meta_path):
    """Resolve relative dataset paths without requiring data in this repo."""
    if not path or os.path.isabs(path):
        return path

    candidates = []
    data_root = os.environ.get('DATA_ROOT')
    if data_root:
        candidates.append(os.path.join(data_root, path))

    meta_path = os.path.abspath(meta_path)
    meta_dir = os.path.dirname(meta_path)
    candidates.append(os.path.join(meta_dir, path))

    parent = meta_dir
    while parent and parent != os.path.dirname(parent):
        candidates.append(os.path.join(parent, path))
        parent = os.path.dirname(parent)

    candidates.append(os.path.abspath(path))
    for candidate in dict.fromkeys(candidates):
        if os.path.exists(candidate):
            return candidate
    return path


class PRMStatsCallback(TrainerCallback):
    """Attach stage-specific PRM statistics to Trainer logs."""

    def __init__(self):
        super().__init__()
        self._model_ref = None

    def on_train_begin(self, args, state, control, model=None, **kwargs):
        if model is not None:
            self._model_ref = model

    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is None:
            return

        if model is None:
            model = self._model_ref

        if model is None:
            return

        target_model = model.module if hasattr(model, 'module') else model
        stats = getattr(target_model, '_prm_last_stats', None)

        if not stats:
            return

        for k, v in stats.items():
            logs[f'train/{k}'] = float(v)


@dataclass
class ModelArguments:
    """
    Arguments for specifying model, tokenizer, and configurations.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Path to a pretrained model (local or from huggingface.co/models).'
        },
    )
    vision_path: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Path to a pretrained model (local or from huggingface.co/models).'
        },
    )
    llm_path: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Path to a pretrained model (local or from huggingface.co/models).'
        },
    )
    mlp_path: Optional[str] = field(
        default=None,
        metadata={
            'help': 'Path to a pretrained model (local or from huggingface.co/models).'
        },
    )
    freeze_llm: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the LLM. Default is False.'},
    )
    freeze_backbone: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the ViT. Default is False.'},
    )
    freeze_mlp: bool = field(
        default=False,
        metadata={'help': 'Set to True to freeze the MLP. Default is False.'},
    )
    unfreeze_vit_layers: int = field(
        default=0,
        metadata={
            'help': 'Specify the number of ViT layers to unfreeze. Default is 0.'
        },
    )
    vision_select_layer: int = field(
        default=-1,
        metadata={
            'help': 'Specify the layer of ViT feature map to use. Default is -1 for the last layer.'
        },
    )
    use_backbone_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the ViT. Default is 0.'},
    )
    use_llm_lora: int = field(
        default=0,
        metadata={'help': 'Set the LoRA adapter rank for the LLM. Default is 0.'},
    )
    unfreeze_lm_head: bool = field(
        default=False,
        metadata={'help': 'Set to True to unfreeze the head of LLM. Default is False.'},
    )
    grad_checkpoint: bool = field(
        default=True,
        metadata={
            'help': 'Set to True to use gradient checkpointing. Default is True.'
        },
    )
    drop_path_rate: float = field(
        default=0.0,
        metadata={'help': 'Set the drop path rate for the ViT. Default is 0.'},
    )
    ps_version: Literal['v1', 'v2'] = field(
        default='v2',
        metadata={
            'help': 'Specify the version of pixel shuffle implementation. Default is v2.'
        },
    )
    use_fast_tokenizer: bool = field(
        default=False,
        metadata={'help': 'Set to True to use the fast mode of the tokenizer.'},
    )
    use_liger: bool = field(
        default=False, metadata={'help': 'Set to True to use the liger kernel.'}
    )
    prm_loss_type: str = field(
        default="ensemble_prm",
        metadata={"help": "Training stage: ensemble_prm or bayesian_prm."},
    )
    ensemble_prm_num_heads: int = field(
        default=8,
        metadata={"help": "Number of ensemble reward heads for ensemble_prm."},
    )
    ensemble_prm_hidden_dim: int = field(
        default=256,
        metadata={"help": "Hidden dimension of each ensemble reward head."},
    )
    ensemble_prm_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate used inside ensemble reward heads."},
    )
    belief_hidden_dim: int = field(
        default=256,
        metadata={"help": "Hidden dimension of the BayesianPRM belief network."},
    )
    belief_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate used inside the BayesianPRM belief network."},
    )
    belief_beta_kl: float = field(
        default=0.1,
        metadata={"help": "KL coefficient beta_KL for BayesianPRM belief ELBO."},
    )
    belief_use_reward_probs: bool = field(
        default=True,
        metadata={
            "help": "Concatenate frozen ensemble reward probabilities to belief head input."
        },
    )
    belief_loglik_normalize_by_n: bool = field(
        default=True,
        metadata={
            "help": "Normalize BayesianPRM count log-likelihood by N for stable training."
        },
    )
    belief_use_conservatism: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to use a conservative posterior correction in "
                "BayesianPRM. When False, BayesianPRM reduces to the original "
                "reliability-posterior version."
            )
        },
    )
    belief_conservatism_beta: float = field(
        default=0.1,
        metadata={
            "help": (
            "Temperature beta_2 for conservatism-aware belief "
            "calibration. The final belief is proportional to "
            "alpha_rel * exp(-reward / beta_2). Must be positive."
            )
        },
    )
    ensemble_prm_bootstrap_prob: float = field(
        default=1.0,
        metadata={
            "help": (
                "Head-wise bootstrap keep probability for ensemble PRM training. "
                "1.0 disables bootstrap."
            )
        },
    )
    ensemble_prm_use_prior_network: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to add a frozen randomized prior network to "
                "the ensemble PRM head."
            )
        },
    )
    ensemble_prm_prior_scale: float = field(
        default=1.0,
        metadata={
            "help": (
                "Scale of the frozen randomized prior logits in ensemble PRM. "
                "Final logit = learned_logit + prior_scale * prior_logit."
            )
        },
    )

@dataclass
class DataTrainingArguments:
    """Arguments for the VisualPRM training data and image pipeline."""

    max_seq_length: int = field(
        default=8192,
        metadata={
            'help': (
                'The maximum total input sequence length after tokenization. Sequences longer '
                'than this will be truncated, sequences shorter will be padded.'
            )
        },
    )
    force_image_size: int = field(
        default=448,
        metadata={'help': 'Set the desired size for the image. Default is 448.'},
    )
    down_sample_ratio: float = field(
        default=0.5,
        metadata={
            'help': 'Set the desired down-sampling ratio for the image. Default is 0.5.'
        },
    )
    pad2square: bool = field(
        default=False,
        metadata={
            'help': 'Pad the image to a square shape if set to True. Default is False.'
        },
    )
    conv_style: str = field(
        default='internlm2-chat', metadata={'help': 'Prompt style for a conversation.'}
    )
    meta_path: str = field(
        default=None,
        metadata={'help': 'The path of the meta file of datasets.'},
    )
    prm_data_split_enable: bool = field(
        default=False,
        metadata={
            "help": (
                "Whether to split the PRM training data into an ensemble "
                "training subset and a belief/reliability training subset."
            )
        },
    )
    prm_data_split_ratio: float = field(
        default=0.8,
        metadata={
            "help": (
                "Fraction of the training data used for ensemble PRM. "
                "The remaining data is used for BayesianPRM belief training."
            )
        },
    )
    prm_data_split_seed: int = field(
        default=42,
        metadata={
            "help": (
                "Random seed for deterministic PRM data split."
            )
        },
    )
    prm_data_split_part: str = field(
        default="auto",
        metadata={
            "help": (
                "Which subset to use when PRM data split is enabled: "
                "auto, ensemble, belief, or all. "
                "auto uses ensemble split for ensemble_prm, belief split "
                "for bayesian_prm, and all data for other modes."
            )
        },
    )
    use_data_resampling: bool = field(
        default=False,
        metadata={'help': 'Set to True to use data resampling. Default is False.'},
    )
    dynamic_image_size: bool = field(
        default=False,
        metadata={
            'help': 'Set to True to use dynamic high resolution strategy. Default is False.'
        },
    )
    use_thumbnail: bool = field(
        default=False,
        metadata={'help': 'Set to True to add a thumbnail image. Default is False.'},
    )
    min_dynamic_patch: int = field(
        default=1,
        metadata={'help': 'The minimum number of dynamic patches. Default is 1.'},
    )
    max_dynamic_patch: int = field(
        default=12,
        metadata={'help': 'The maximum number of dynamic patches. Default is 12.'},
    )
    min_num_frame: int = field(
        default=8,
        metadata={'help': 'The minimum number of frames for video data. Default is 8.'},
    )
    max_num_frame: int = field(
        default=32,
        metadata={
            'help': 'The maximum number of frames for video data. Default is 32.'
        },
    )
    normalize_type: Literal['imagenet', 'clip', 'siglip'] = field(
        default='imagenet',
        metadata={'help': 'The normalization type for the image. Default is imagenet.'},
    )
    use_packed_ds: bool = field(
        default=False,
        metadata={
            'help': 'Whether to use packed dataset for efficient training. Default is False.'
        },
    )
    num_images_expected: int = field(
        default=40,
        metadata={
            'help': 'The maximum number of images per packed sample. Default is 40.'
        },
    )
    max_packed_tokens: int = field(
        default=8192,
        metadata={
            'help': 'The required token length of per packed sample. Default is 8192.'
        },
    )
    max_buffer_size: int = field(
        default=20,
        metadata={'help': 'The buffer size of the packed dataset. Default is 20.'},
    )
    log_freq: int = field(
        default=1000,
        metadata={'help': 'The log frequency of the packed dataset. Default is 1000.'},
    )
    strict_mode: bool = field(
        default=True,
        metadata={
            'help': 'Whether to pad the number of images to satisfy num_images_expected. Default is True.'
        },
    )
    replacement: bool = field(
        default=False,
        metadata={
            'help': 'Whether to restart the dataset after it is exhausted. Default is False.'
        },
    )
    allow_overflow: bool = field(
        default=False,
        metadata={
            'help': 'Whether to drop the sample over the specified max_packed_tokens. Default is False.'
        },
    )
    loss_reduction: str = field(
        default='token',
        metadata={'help': 'Loss reduction method. Default is token.'},
    )
    loss_reduction_all_gather: bool = field(
        default=False,
        metadata={
            'help': 'Whether to gather all during loss reduction. Default is False.'
        },
    )


def _safe_float(v):
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None
    
def split_prm_train_dataset(
    train_dataset,
    split_enable: bool,
    split_ratio: float,
    split_seed: int,
    split_part: str,
    prm_loss_type: str,
):
    """
    Deterministically split the PRM training dataset into two disjoint parts.

    - ensemble split:
        Used to train the ensemble PRM reward hypotheses.

    - belief split:
        Used to train the BayesianPRM reliability posterior / belief head
        with the ensemble frozen.

    This helper does not modify the underlying dataset. It only wraps it
    with torch.utils.data.Subset.
    """
    if not split_enable:
        return train_dataset, {
            "enabled": False,
            "resolved_part": "all",
            "raw_len": len(train_dataset),
            "used_len": len(train_dataset),
            "ensemble_len": len(train_dataset),
            "belief_len": 0,
        }

    if not hasattr(train_dataset, "__len__"):
        raise ValueError(
            "PRM data split requires a map-style dataset with __len__."
        )

    n_total = len(train_dataset)

    if n_total <= 1:
        raise ValueError(
            f"PRM data split requires at least 2 samples, got {n_total}."
        )

    if not 0.0 < float(split_ratio) < 1.0:
        raise ValueError(
            f"prm_data_split_ratio must be in (0, 1), got {split_ratio}."
        )

    split_part = str(split_part).lower()

    if split_part == "auto":
        if prm_loss_type == "ensemble_prm":
            split_part = "ensemble"
        elif prm_loss_type == "bayesian_prm":
            split_part = "belief"
        else:
            split_part = "all"

    if split_part == "all":
        return train_dataset, {
            "enabled": True,
            "resolved_part": "all",
            "raw_len": n_total,
            "used_len": n_total,
            "ensemble_len": n_total,
            "belief_len": 0,
        }

    if split_part not in ("ensemble", "belief"):
        raise ValueError(
            "prm_data_split_part must be one of: "
            f"auto, ensemble, belief, all. Got {split_part}."
        )

    n_ensemble = int(round(n_total * float(split_ratio)))
    n_ensemble = max(1, min(n_total - 1, n_ensemble))
    n_belief = n_total - n_ensemble

    generator = torch.Generator()
    generator.manual_seed(int(split_seed))

    perm = torch.randperm(
        n_total,
        generator=generator,
    ).tolist()

    ensemble_indices = perm[:n_ensemble]
    belief_indices = perm[n_ensemble:]

    if split_part == "ensemble":
        selected_indices = ensemble_indices
    else:
        selected_indices = belief_indices

    split_dataset = Subset(
        train_dataset,
        selected_indices,
    )

    split_info = {
        "enabled": True,
        "resolved_part": split_part,
        "raw_len": n_total,
        "used_len": len(selected_indices),
        "ensemble_len": n_ensemble,
        "belief_len": n_belief,
        "split_ratio": float(split_ratio),
        "split_seed": int(split_seed),
    }

    return split_dataset, split_info


def preprocess_internvl2_5_prm(
    template_name,
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    num_image_token_list: list,
    text_only: bool = False,
    group_by_length: bool = False,
    use_packed_ds: bool = False,
    ds_name: str = None,
    num_image: int = 1,
):
    """InternVL2.5 preprocess with extra prm_counts_k/n channels."""
    PRM_TOKEN_ID = tokenizer.convert_tokens_to_ids(PRM_TOKEN)
    assert len(sources) == 1, 'process only the first conversations'
    conversations = sources[0]

    if conversations[0]['from'] == 'system':
        system_prompt = conversations[0]['value']
        conversations = conversations[1:]
    else:
        conv = get_conv_template(template_name)
        system_prompt = conv.system_message

    if not text_only:
        new_conversations = []
        current_image_idx = 0
        for conversation in conversations:
            if conversation['from'] == 'human':
                image_cnt = conversation['value'].count('<image>')
                for _ in range(image_cnt):
                    if current_image_idx == num_image:
                        break
                    image_tokens = (
                        f'{IMG_START_TOKEN}'
                        f'{IMG_CONTEXT_TOKEN * num_image_token_list[current_image_idx]}'
                        f'{IMG_END_TOKEN}'
                    )
                    conversation['value'] = conversation['value'].replace(
                        '<image>', image_tokens, 1
                    )
                    current_image_idx += 1
            new_conversations.append(conversation)
        conversations = new_conversations
        assert current_image_idx == num_image, f'{current_image_idx} != {num_image}'

    batches, roles = [], []
    if system_prompt is not None:
        batches.append(f'<|im_start|>system\n{system_prompt}<|im_end|>\n')
        roles.append('system')
    batches.append(f'<|im_start|>user\n{""}<|im_end|>\n')
    roles.append('human')
    batches.append(f'<|im_start|>assistant\n{conversations[0]["value"]}<|im_end|>\n')
    roles.append('gpt')

    # Fallback to ratios if prm_counts is absent.
    ratios = conversations[1]['value']
    # Dataset injects prm_counts into assistant conversation before calling preprocess.
    prm_counts = conversations[1].get('prm_counts', None)
    k_list = None
    n_list = None
    if isinstance(prm_counts, dict):
        k_list = prm_counts.get('k', None)
        n_list = prm_counts.get('n', None)
    if k_list is None or n_list is None:
        # recover counts from ratio labels
        n_list = [16.0] * len(ratios)
        k_list = []
        for r in ratios:
            rv = _safe_float(r)
            if rv is None:
                rv = 0.0
            rv = min(max(rv, 0.0), 1.0)
            k_list.append(round(rv * 16.0))

    normalized_ratios = []
    normalized_k = []
    normalized_n = []
    for idx in range(len(ratios)):
        ratio_v = _safe_float(ratios[idx])
        if ratio_v is None:
            ratio_v = 0.0
        ratio_v = min(max(ratio_v, 0.0), 1.0)

        k_v = _safe_float(k_list[idx]) if idx < len(k_list) else None
        n_v = _safe_float(n_list[idx]) if idx < len(n_list) else None
        if n_v is None or n_v <= 0:
            n_v = 16.0
        if k_v is None:
            k_v = round(ratio_v * n_v)
        k_v = min(max(k_v, 0.0), n_v)

        normalized_ratios.append(ratio_v)
        normalized_k.append(k_v)
        normalized_n.append(n_v)

    labels = torch.tensor(normalized_ratios, dtype=torch.float)
    k_tensor = torch.tensor(normalized_k, dtype=torch.float)
    n_tensor = torch.tensor(normalized_n, dtype=torch.float)

    add_bos_token = getattr(tokenizer, 'add_bos_token', False)
    if add_bos_token:
        batches[0] = tokenizer.bos_token + batches[0]

    input_ids = tokenizer(
        batches,
        return_tensors='np',
        padding=False,
        max_length=tokenizer.model_max_length,
        truncation=False,
    ).input_ids

    if add_bos_token:
        input_ids = [item[1:] for item in input_ids]

    final_input_ids = []
    for _, input_id in zip(roles, input_ids):
        final_input_ids.append(input_id)
    input_ids = torch.tensor(np.concatenate(final_input_ids))

    prm_mask = input_ids == PRM_TOKEN_ID
    if int(prm_mask.sum().item()) != len(labels):
        raise ValueError(
            f'PRM token count mismatch in {ds_name}: '
            f'num_prm={int(prm_mask.sum().item())}, labels={len(labels)}'
        )

    targets = input_ids.clone().to(torch.float)
    targets.fill_(IGNORE_INDEX)
    targets[prm_mask] = labels

    prm_counts_k = input_ids.clone().to(torch.float)
    prm_counts_k.fill_(IGNORE_INDEX)
    prm_counts_k[prm_mask] = k_tensor

    prm_counts_n = input_ids.clone().to(torch.float)
    prm_counts_n.fill_(IGNORE_INDEX)
    prm_counts_n[prm_mask] = n_tensor

    input_ids = input_ids[: tokenizer.model_max_length]
    targets = targets[: tokenizer.model_max_length]
    prm_counts_k = prm_counts_k[: tokenizer.model_max_length]
    prm_counts_n = prm_counts_n[: tokenizer.model_max_length]

    padding = False if group_by_length or use_packed_ds else True
    if padding:
        current_length = input_ids.size(0)
        padding_length = tokenizer.model_max_length - current_length
        input_ids = F.pad(input_ids, (0, padding_length), value=tokenizer.pad_token_id)
        targets = F.pad(targets, (0, padding_length), value=IGNORE_INDEX)
        prm_counts_k = F.pad(prm_counts_k, (0, padding_length), value=IGNORE_INDEX)
        prm_counts_n = F.pad(prm_counts_n, (0, padding_length), value=IGNORE_INDEX)

    input_ids = input_ids.unsqueeze(0)
    targets = targets.unsqueeze(0)
    prm_counts_k = prm_counts_k.unsqueeze(0)
    prm_counts_n = prm_counts_n.unsqueeze(0)

    return dict(
        input_ids=input_ids,
        labels=targets,
        attention_mask=input_ids.ne(tokenizer.pad_token_id),
        prm_counts_k=prm_counts_k,
        prm_counts_n=prm_counts_n,
    )


def concat_pad_data_collator_prm(features, max_item_length=None, pad_id=0):
    """Pad count supervision together with input_ids and labels."""
    first = features[0]
    batch = {}

    batch_lens = [feat['input_ids'].shape for feat in features]
    max_item_length = max_item_length or max(batch_lens)[0]
    for idx in range(len(features)):
        feat = features[idx]
        temp_input_ids = torch.LongTensor([pad_id] * max_item_length)
        temp_input_ids[: feat['input_ids'].shape[0]] = feat['input_ids']
        feat['input_ids'] = temp_input_ids

        temp_labels = torch.FloatTensor([IGNORE_INDEX] * max_item_length)
        temp_labels[: feat['labels'].shape[0]] = feat['labels']
        feat['labels'] = temp_labels
        feat['attention_mask'] = feat['input_ids'].ne(pad_id)

        if 'position_ids' in feat:
            temp_position_ids = torch.LongTensor([pad_id] * max_item_length)
            temp_position_ids[: feat['position_ids'].shape[0]] = feat['position_ids']
            feat['position_ids'] = temp_position_ids

        if 'loss_weight' in feat:
            temp_loss_weight = torch.FloatTensor([pad_id] * max_item_length)
            temp_loss_weight[: feat['loss_weight'].shape[0]] = feat['loss_weight']
            feat['loss_weight'] = temp_loss_weight

        if 'prm_counts_k' in feat:
            temp_k = torch.FloatTensor([IGNORE_INDEX] * max_item_length)
            temp_k[: feat['prm_counts_k'].shape[0]] = feat['prm_counts_k']
            feat['prm_counts_k'] = temp_k

        if 'prm_counts_n' in feat:
            temp_n = torch.FloatTensor([IGNORE_INDEX] * max_item_length)
            temp_n[: feat['prm_counts_n'].shape[0]] = feat['prm_counts_n']
            feat['prm_counts_n'] = temp_n

    if 'label' in first and first['label'] is not None:
        batch['labels'] = torch.tensor([f['label'] for f in features], dtype=torch.float)
    elif 'label_ids' in first and first['label_ids'] is not None:
        if isinstance(first['label_ids'], torch.Tensor):
            batch['labels'] = torch.stack([f['label_ids'] for f in features])
        else:
            batch['labels'] = torch.tensor(
                [f['label_ids'] for f in features], dtype=torch.float
            )

    for k, v in first.items():
        if (
            k not in ('label', 'label_ids', 'pixel_values', 'image_flags')
            and v is not None
            and not isinstance(v, str)
        ):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.tensor(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.tensor([f[k] for f in features])
        if k in ('pixel_values', 'image_flags'):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.concat([f[k] for f in features])
            elif isinstance(v, np.ndarray):
                batch[k] = torch.concat(np.stack([f[k] for f in features]))
            else:
                batch[k] = torch.concat([f[k] for f in features])
    return batch


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        template_name,
        meta,
        tokenizer,
        tcs_loader,
        ds_name,
        num_image_token,
        image_size=448,
        is_train=True,
        pad2square=False,
        group_by_length=False,
        dynamic_image_size=False,
        use_thumbnail=False,
        min_dynamic_patch=1,
        max_dynamic_patch=12,
        min_num_frame=8,  # for video data
        max_num_frame=32,  # for video data
        sampling_method='rand',  # for video data
        repeat_time=1,
        normalize_type='imagenet',
        # hyperparameters for packed training
        use_packed_ds=False,
        data_rank=0,
        data_world_size=1,
        distributed_mode=False,
        force_shuffle=False,
        random_seed=0,
    ):
        super(LazySupervisedDataset, self).__init__()
        self.ds_name = ds_name
        self.tokenizer = tokenizer
        self.template_name = template_name
        self.num_image_token = num_image_token
        logger.info(f'[Dataset] num_image_token: {num_image_token}')
        logger.info(f'[Dataset] dynamic_image_size: {dynamic_image_size}')
        logger.info(f'[Dataset] use_thumbnail: {use_thumbnail}')
        logger.info(
            f'[Dataset] min_dynamic_patch: {min_dynamic_patch}, max_dynamic_patch: {max_dynamic_patch}'
        )

        self.image_size = image_size
        self.is_train = is_train
        self.pad2square = pad2square
        self.max_num_frame = max_num_frame
        self.min_num_frame = min_num_frame
        self.sampling_method = sampling_method

        # hyperparameters for distributed training
        self.use_packed_ds = use_packed_ds
        self.data_rank = data_rank
        self.data_world_size = data_world_size
        self.worker_id = None
        self.worker_state_key = None
        self.worker_distributed = False
        self.distributed_mode = distributed_mode
        # hyperparameters for packed dataset
        self.dataset_type = 'pair'
        self.max_num_images = 1
        self.max_tokens = tokenizer.model_max_length
        self.force_shuffle = force_shuffle
        # TODO: quick resume
        self._state_dict = {}

        logger.info('Formatting inputs...Skip in lazy mode')
        assert meta['annotation'].endswith(
            'jsonl'
        ), f'annotation must be jsonl, but got {meta["annotation"]}'

        with open(meta['annotation'], 'r') as f:
            self.raw_data = f.readlines()
            if repeat_time < 1:
                # If repeat_time is less than 1, select a portion of the data
                self.raw_data = self.raw_data[: int(len(self.raw_data) * repeat_time)]
            if repeat_time > 1:
                assert isinstance(repeat_time, int)
                # Repeat the list if repeat_time is greater than 1
                self.raw_data = self.raw_data * repeat_time

        self.rng = np.random.default_rng(seed=random_seed)
        if self.force_shuffle:
            self.rng.shuffle(self.raw_data)

        self.root = meta['root']
        self.cached_data_dict = {}
        self.tcs_loader = tcs_loader
        self.group_by_length = group_by_length
        self.dynamic_image_size = dynamic_image_size
        self.use_thumbnail = use_thumbnail
        self.min_dynamic_patch = min_dynamic_patch
        self.max_dynamic_patch = max_dynamic_patch
        self.normalize_type = normalize_type

        # If the precomputed length does not exist, roughly estimate the length of
        # each sample to improve the efficiency of group_by_length.
        if self.group_by_length:
            self.conv2length = (
                {}
            )  # Using a dictionary to speed up token length calculation
            self.length = []
            for data_item in self.raw_data:
                data_item = json.loads(data_item)
                if 'length' in data_item:
                    token_length = data_item[
                        'length'
                    ]  # Use precomputed length if available
                else:
                    # Compute token length using the tokenizer
                    conversations = data_item['conversations'][0]['value']
                    str_length = len(conversations)
                    if str_length not in self.conv2length:
                        token_length = tokenizer(
                            conversations,
                            return_tensors='pt',
                            padding=False,
                            truncation=False,
                        ).input_ids.size(1)
                        self.conv2length[str_length] = (
                            token_length
                            + num_image_token * (max_dynamic_patch + use_thumbnail)
                        )
                    else:
                        token_length = self.conv2length[str_length]
                self.length.append(token_length)

    def __len__(self):
        return len(self.raw_data)

    def get_preprocess_function(self):
        # Select the appropriate preprocessing function based on the template name
        if self.template_name == 'Hermes-2':
            preprocess_function = preprocess_mpt
        elif self.template_name == 'internlm2-chat':
            preprocess_function = preprocess_internlm
        elif self.template_name == 'phi3-chat':
            preprocess_function = preprocess_phi3
        elif self.template_name == 'internvl2_5':
            preprocess_function = preprocess_internvl2_5_prm
        else:
            preprocess_function = preprocess
        return preprocess_function

    def load_image(self, image_path):
        # Load the image using tcs_loader if available, otherwise use PIL
        if self.tcs_loader is not None and 's3://' in image_path:
            return self.tcs_loader(image_path)
        return Image.open(image_path).convert('RGB')

    def get_image_path(self, image_path):
        if image_path.startswith('s3://'):  # for ceph
            image_path = self.root + image_path
        elif image_path.startswith('images/train-'):
            image_path = os.path.join(
                self.root,
                'VisualPRM400K-v1.1-Raw',
                'nlvr2',
                image_path,
            )
        else:  # for local image
            image_path = os.path.join(self.root, image_path)
        return image_path

    def get_transform(self):
        # Build transformation function
        transform = build_transform(
            is_train=self.is_train,
            input_size=self.image_size,
            pad2square=self.pad2square,
            normalize_type=self.normalize_type,
        )
        return transform

    def _pack_conversations_with_prm_counts(self, data_item):
        packed_conversations = deepcopy(data_item['conversations'])
        if 'prm_counts' not in data_item:
            return packed_conversations

        assistant_idx = None
        for idx, msg in enumerate(packed_conversations):
            if isinstance(msg, dict) and msg.get('from') == 'gpt':
                assistant_idx = idx
        if assistant_idx is None:
            assistant_idx = 1 if len(packed_conversations) > 1 else 0

        packed_conversations[assistant_idx]['prm_counts'] = data_item['prm_counts']
        return packed_conversations

    def multi_modal_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains an image placeholder
        if '<image>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = (
                '<image>\n' + data_item['conversations'][0]['value']
            )

        # Merge the image path
        image_path = self.get_image_path(data_item['image'])

        # Load the image using tcs_loader if available, otherwise use PIL
        image = self.load_image(image_path)

        if (
            self.dynamic_image_size
        ):  # If dynamic image size is enabled, preprocess the image dynamically
            images = dynamic_preprocess(
                image,
                min_num=self.min_dynamic_patch,
                max_num=self.max_dynamic_patch,
                image_size=self.image_size,
                use_thumbnail=self.use_thumbnail,
            )
        else:  # Otherwise, use the original image as a single patch
            images = [image]

        # Apply the transformation to each image and stack the results into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)

        # Ensure that there is only one patch if dynamic image size is not enabled
        num_patches = pixel_values.size(0)
        if not self.dynamic_image_size:
            assert (
                num_patches == 1
            ), f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        packed_conversations = self._pack_conversations_with_prm_counts(data_item)

        ret = preprocess_function(
            self.template_name,
            [packed_conversations],
            self.tokenizer,
            [self.num_image_token * num_patches],
            group_by_length=self.group_by_length,
            use_packed_ds=self.use_packed_ds,
            ds_name=self.ds_name,
        )

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (
            ret['input_ids'][0] == image_end_token_id
        ).sum() == 1, f'image tokens are truncated, this dataset is {self.ds_name}'
        prm_counts_k = ret.get('prm_counts_k', None)
        prm_counts_n = ret.get('prm_counts_n', None)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        if prm_counts_k is not None and prm_counts_n is not None:
            ret['prm_counts_k'] = prm_counts_k[0]
            ret['prm_counts_n'] = prm_counts_n[0]
        return ret

    def multi_modal_multi_image_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        images, num_tiles = [], []
        num_image = len(data_item['image'])
        for image_path in data_item['image']:
            # Merge the image path
            image_path = self.get_image_path(image_path)
            # Load the image using tcs_loader if available, otherwise use PIL
            image = self.load_image(image_path)
            if (
                self.dynamic_image_size
            ):  # If dynamic image size is enabled, preprocess the image dynamically
                image = dynamic_preprocess(
                    image,
                    min_num=self.min_dynamic_patch,
                    max_num=max(1, self.max_dynamic_patch // num_image),
                    image_size=self.image_size,
                    use_thumbnail=self.use_thumbnail,
                )
                images += image
                num_tiles.append(len(image))
            else:  # Otherwise, use the original image as a single patch
                images.append(image)
                num_tiles.append(1)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token * num_tile for num_tile in num_tiles]
        packed_conversations = self._pack_conversations_with_prm_counts(data_item)

        ret = preprocess_function(
            self.template_name,
            [packed_conversations],
            self.tokenizer,
            num_image_tokens,
            group_by_length=self.group_by_length,
            use_packed_ds=self.use_packed_ds,
            ds_name=self.ds_name,
            num_image=num_image,
        )

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        image_end_token_id = self.tokenizer.convert_tokens_to_ids(IMG_END_TOKEN)
        assert (
            ret['input_ids'][0] == image_end_token_id
        ).sum() == num_image, (
            f'image tokens are truncated, this dataset is {self.ds_name}'
        )
        prm_counts_k = ret.get('prm_counts_k', None)
        prm_counts_n = ret.get('prm_counts_n', None)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        if prm_counts_k is not None and prm_counts_n is not None:
            ret['prm_counts_k'] = prm_counts_k[0]
            ret['prm_counts_n'] = prm_counts_n[0]
        return ret

    def video_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Ensure the first conversation contains a video placeholder
        if '<video>' not in data_item['conversations'][0]['value']:
            data_item['conversations'][0]['value'] = (
                '<video>\n' + data_item['conversations'][0]['value']
            )

        # Get the video file path
        video_file = data_item['video']
        video_path = os.path.join(self.root, video_file)

        # Load the video frames using tcs_loader
        # TODO: Load videos without using tcsloader.
        image_list = self.tcs_loader(
            video_path,
            image_type='video',
            max_num_frames=self.max_num_frame,
            min_num_frames=self.min_num_frame,
            sample=self.sampling_method,
            clip=data_item.get('clip', None),
        )

        # Generate special tokens for each video frame
        special_tokens = '\n'.join(
            ['Frame-{}: <image>'.format(i + 1) for i in range(len(image_list))]
        )
        data_item['conversations'][0]['value'] = data_item['conversations'][0][
            'value'
        ].replace('<video>\n', special_tokens + '\n')

        # Transform each frame image and stack them into a tensor
        pixel_values = [transform(image) for image in image_list]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        num_image_tokens = [self.num_image_token] * num_patches
        packed_conversations = self._pack_conversations_with_prm_counts(data_item)

        ret = preprocess_function(
            self.template_name,
            [packed_conversations],
            self.tokenizer,
            num_image_tokens,
            group_by_length=self.group_by_length,
            use_packed_ds=self.use_packed_ds,
            ds_name=self.ds_name,
            num_image=num_patches,
        )

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        prm_counts_k = ret.get('prm_counts_k', None)
        prm_counts_n = ret.get('prm_counts_n', None)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([1] * num_patches, dtype=torch.long),
        )
        if prm_counts_k is not None and prm_counts_n is not None:
            ret['prm_counts_k'] = prm_counts_k[0]
            ret['prm_counts_n'] = prm_counts_n[0]
        return ret

    def pure_text_get_item(self, data_item):
        # Build transformation function
        transform = self.get_transform()

        # Create a blank white image
        image = Image.new('RGB', (224, 224), (255, 255, 255))

        # Dynamically preprocess the image to generate patches
        images = dynamic_preprocess(
            image,
            min_num=self.min_dynamic_patch,
            max_num=1,
            image_size=self.image_size,
            use_thumbnail=self.use_thumbnail,
        )

        # Apply the transformation to each image patch and stack them into a tensor
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        num_patches = pixel_values.size(0)

        # Ensure there is only one patch
        assert (
            num_patches == 1
        ), f'The number of patches should be 1, but got {num_patches}.'

        # Select the appropriate preprocessing function based on the template name
        preprocess_function = self.get_preprocess_function()

        # Preprocess the conversations and generate the return dictionary
        packed_conversations = self._pack_conversations_with_prm_counts(data_item)

        ret = preprocess_function(
            self.template_name,
            [packed_conversations],
            self.tokenizer,
            [self.num_image_token * num_patches],
            text_only=True,
            group_by_length=self.group_by_length,
            use_packed_ds=self.use_packed_ds,
            ds_name=self.ds_name,
        )

        # Calculate position_ids for packed dataset
        position_ids = ret['attention_mask'].long().cumsum(-1) - 1
        position_ids.masked_fill_(ret['attention_mask'] == 0, 1)
        prm_counts_k = ret.get('prm_counts_k', None)
        prm_counts_n = ret.get('prm_counts_n', None)

        # Create the final return dictionary
        ret = dict(
            input_ids=ret['input_ids'][0],
            labels=ret['labels'][0],
            attention_mask=ret['attention_mask'][0],
            position_ids=position_ids[0],
            pixel_values=pixel_values,
            image_flags=torch.tensor([0] * num_patches, dtype=torch.long),
        )
        if prm_counts_k is not None and prm_counts_n is not None:
            ret['prm_counts_k'] = prm_counts_k[0]
            ret['prm_counts_n'] = prm_counts_n[0]
        return ret

    def _enable_worker_distributed(self):
        if (
            self.distributed_mode
            and not self.worker_distributed
            and self.worker_id is not None
        ):
            self.worker_distributed = True
            self.raw_data = self.raw_data[self.worker_id :: self.num_workers]
            logger.info(
                f'worker_distributed is enabled, {self.num_workers=}, {len(self.raw_data)=}'
            )

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        if i >= len(self.raw_data):
            if self.use_packed_ds:
                raise NotImplementedError
            else:
                i = i % len(self.raw_data)

        try_cnt, max_try = 0, 10
        while True:
            if try_cnt > max_try:
                raise StopIteration
            try:
                data_item = json.loads(self.raw_data[i])
                # conversations = data_item['conversations']
                # check_conversations_repetition(conversations, repeat_threshold=0.4, ngram=10)
                if 'image' in data_item and len(data_item['image']) != 0:
                    if type(data_item['image']) == list:
                        ret = self.multi_modal_multi_image_get_item(data_item)
                    else:
                        ret = self.multi_modal_get_item(data_item)
                elif (
                    'video' in data_item
                    and data_item['video'] is not None
                    and data_item['video'] != ''
                ):
                    ret = self.video_get_item(data_item)
                else:
                    ret = self.pure_text_get_item(data_item)
                break
            except Exception as e:
                try_cnt += 1
                print(e, self.ds_name, flush=True)
                if not isinstance(e, (UnidentifiedImageError, FileNotFoundError)):
                    traceback.print_exc()
                data_item = json.loads(self.raw_data[i])
                if 'image' in data_item:
                    if type(data_item['image']) == list:
                        images = [self.root + item for item in data_item['image']]
                        print(
                            f'Failed to load image: {images}, the dataset is: {self.ds_name}'
                        )
                    else:
                        if data_item['image'].startswith('s3://'):
                            data_path = self.root + data_item['image']
                        else:
                            data_path = os.path.join(self.root, data_item['image'])
                        print(
                            f'Failed to load image: {data_path}, the dataset is: {self.ds_name}'
                        )
                elif 'video' in data_item:
                    data_path = os.path.join(self.root, data_item['video'])
                    print(
                        f'Failed to load video: {data_path}, the dataset is: {self.ds_name}'
                    )
                i = random.randint(0, len(self.raw_data) - 1)
        return ret

    def __iter__(self):
        self._enable_worker_distributed()
        start_idx = 0

        assert self.worker_state_key is not None
        if (
            self.worker_state_key in self._state_dict
            and len(self._state_dict[self.worker_state_key]) > 0
        ):
            start_idx = self._state_dict[self.worker_state_key]['current_idx']

            self._state_dict.pop(self.worker_state_key)

        if self.worker_id == 0:
            logger.info(
                f'[{self.ds_name}] [Worker id {self.worker_id}] '
                f'begin to iter with {start_idx=}'
            )

        for i in range(start_idx, len(self)):
            yield self[i]


def build_datasets(
    data_args,
    tokenizer,
    tcs_loader,
    model,
    group_by_length=False,
    dynamic_image_size=False,
    use_thumbnail=False,
    min_dynamic_patch=1,
    max_dynamic_patch=12,
    min_num_frame=8,
    max_num_frame=32,
    normalize_type='imagenet',
):
    datasets = []
    lengths = []
    data_rank = dist.get_rank()
    data_world_size = dist.get_world_size()
    ds_collections = json.loads(open(data_args.meta_path).read())
    for ds_idx, ds_name in enumerate(ds_collections.keys()):
        ds_collections[ds_name]['annotation'] = _resolve_external_data_path(
            ds_collections[ds_name]['annotation'], data_args.meta_path
        )
        ds_collections[ds_name]['root'] = _resolve_external_data_path(
            ds_collections[ds_name].get('root', ''), data_args.meta_path
        )
        repeat_time = ds_collections[ds_name]['repeat_time']
        if 'max_dynamic_patch' in ds_collections[ds_name]:
            max_num = ds_collections[ds_name]['max_dynamic_patch']
            logger.info(
                f'max_dynamic_patch is set to {max_num} according to the meta file'
            )
        else:
            max_num = max_dynamic_patch
        dataset = LazySupervisedDataset(
            data_args.conv_style,
            ds_collections[ds_name],
            tokenizer,
            tcs_loader,
            ds_name=ds_name,
            num_image_token=model.num_image_token,
            image_size=data_args.force_image_size,
            is_train=ds_collections[ds_name]['data_augment'],
            pad2square=data_args.pad2square,
            group_by_length=group_by_length and not data_args.use_packed_ds,
            dynamic_image_size=dynamic_image_size,
            use_thumbnail=use_thumbnail,
            min_dynamic_patch=min_dynamic_patch,
            max_dynamic_patch=max_num,
            min_num_frame=min_num_frame,
            max_num_frame=max_num_frame,
            repeat_time=repeat_time,
            normalize_type=normalize_type,
            # hyperparameters for packed training
            use_packed_ds=data_args.use_packed_ds,
            data_rank=data_rank,
            data_world_size=data_world_size,
            distributed_mode=data_args.use_packed_ds,
            force_shuffle=data_args.use_packed_ds,
            random_seed=ds_idx,
        )
        logger.info(f'Add dataset: {ds_name} with length: {len(dataset)}')
        datasets.append(dataset)
        if data_args.use_data_resampling:
            lengths.append(math.sqrt(len(dataset)))
        else:
            lengths.append(len(dataset))

    if data_args.use_packed_ds:
        total_length = sum(lengths)
        train_dataset = PackedDataset(
            tokenizer=tokenizer,
            data_rank=data_rank,
            data_world_size=data_world_size,
            datasets=datasets,
            dataset_weight=[l / total_length for l in lengths],
            num_images_expected=data_args.num_images_expected,
            max_packed_tokens=data_args.max_packed_tokens,
            max_buffer_size=data_args.max_buffer_size,
            log_freq=data_args.log_freq,
            strict_mode=data_args.strict_mode,
            replacement=data_args.replacement,
            allow_overflow=data_args.allow_overflow,
            allow_deduplicated_ds_name=False,
        )
    elif data_args.use_data_resampling:
        total_length = sum(lengths)
        weights = [l / total_length for l in lengths]
        train_dataset = WeightedConcatDataset(datasets, weights)
    else:
        train_dataset = ConcatDataset(datasets)
    return train_dataset


def len2weight(x, loss_reduction):
    if x == 0:
        return x
    if loss_reduction == 'token':
        return 1
    if loss_reduction == 'sample':
        return 1 / x
    if loss_reduction == 'square':
        return 1 / (x**0.5)
    raise NotImplementedError(loss_reduction)


def main():
    # Apply necessary patches for the transformers library
    replace_llama_rmsnorm_with_fused_rmsnorm()
    replace_train_sampler()
    replace_train_dataloader()

    # Parse input arguments
    # See all possible arguments in src/transformers/training_args.py
    # If use DeepSpeed zero3, init_dist must before HfArgumentParser
    launcher = os.environ.get('LAUNCHER', 'slurm')
    init_dist(launcher=launcher, backend='nccl')
    parser = HfArgumentParser(
        (ModelArguments, DataTrainingArguments, TrainingArguments)
    )
    if len(sys.argv) == 2 and sys.argv[1].endswith('.json'):
        # If we pass only one argument to the script, and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    if model_args.prm_loss_type not in {"ensemble_prm", "bayesian_prm"}:
        raise ValueError(
            "prm_loss_type must be 'ensemble_prm' or 'bayesian_prm'."
        )
    if not data_args.meta_path:
        raise ValueError("meta_path is required for PRM training.")

    training_args.use_packed_ds = data_args.use_packed_ds

    # Sending telemetry. Tracking the example usage helps us better allocate resources to maintain them. The
    # information sent is the one passed as arguments along with your Python/PyTorch versions.
    # send_example_telemetry('InternV-Chat', model_args, data_args)

    # Setup logging
    logging.basicConfig(
        format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
        datefmt='%m/%d/%Y %H:%M:%S',
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    if training_args.should_log:
        # The default of training_args.log_level is passive, so we set log level at info here to have that default.
        transformers.utils.logging.set_verbosity_info()

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    set_verbosity(log_level)
    enable_default_handler()
    enable_explicit_format()

    # Log on each process the small summary:
    logger.warning(
        f'Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}'
        + f'distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}'
    )
    logger.info(f'Training parameters {training_args}')

    # Detecting last checkpoint and eventually continue from last checkpoint.
    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f'Output directory ({training_args.output_dir}) already exists and is not empty. '
                'Use --overwrite_output_dir to overcome.'
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            logger.info(
                f'Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change '
                'the `--output_dir` or add `--overwrite_output_dir` to train from scratch.'
            )
    # Resolve the effective resume checkpoint once and use it consistently.
    resume_checkpoint = (
        training_args.resume_from_checkpoint
        if training_args.resume_from_checkpoint is not None
        else last_checkpoint
    )

    is_resume_training = resume_checkpoint is not None

    if is_resume_training:
        resume_checkpoint = os.path.abspath(str(resume_checkpoint))

        if not os.path.isdir(resume_checkpoint):
            raise ValueError(
                f"Resume checkpoint does not exist: {resume_checkpoint}"
            )

        if dist.get_rank() == 0:
            logger.info(
                f"Strict resume training from: {resume_checkpoint}"
            )
    # Set seed before initializing model.
    set_seed(training_args.seed)

    # Load pretrained model, tokenizer, and image processor
    tokenizer_path = model_args.model_name_or_path or model_args.llm_path
    logger.info(f'Loading Tokenizer: {tokenizer_path}')
    tokenizer = AutoTokenizer.from_pretrained(
        tokenizer_path,
        add_eos_token=False,
        trust_remote_code=True,
        use_fast=model_args.use_fast_tokenizer,
    )
    tokenizer.tokenizer_path = tokenizer_path
    tokenizer.model_max_length = data_args.max_seq_length
    token_list = [
        IMG_START_TOKEN,
        IMG_END_TOKEN,
        IMG_CONTEXT_TOKEN,
        QUAD_START_TOKEN,
        QUAD_END_TOKEN,
        REF_START_TOKEN,
        REF_END_TOKEN,
        BOX_START_TOKEN,
        BOX_END_TOKEN,
        PRM_TOKEN,
    ]
    num_new_tokens = tokenizer.add_tokens(token_list, special_tokens=True)
    img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
    tcs_loader = TCSLoader('~/petreloss.conf') if has_tcs_loader else None

    if data_args.use_packed_ds:
        replace_internlm2_attention_class()
        replace_qwen2_attention_class()
        replace_phi3_attention_class()
        replace_llama_attention_class()

    if model_args.use_liger:
        from liger_kernel.transformers import (apply_liger_kernel_to_llama,
                                               apply_liger_kernel_to_qwen2)

        from internvl.patch import apply_liger_kernel_to_internvit

        apply_liger_kernel_to_llama()
        apply_liger_kernel_to_qwen2()
        # apply_liger_kernel_to_internvit()

    if model_args.model_name_or_path is not None:
        logger.info('Loading InternVLChatModel...')
        config_load_path = (
            resume_checkpoint
            if is_resume_training
            else model_args.model_name_or_path
        )
        config = InternVLChatConfig.from_pretrained(config_load_path)
        config.vision_config.drop_path_rate = model_args.drop_path_rate
        if config.llm_config.model_type == 'internlm2':
            config.llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
        else:
            config.llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
        config.template = data_args.conv_style
        config.select_layer = model_args.vision_select_layer
        config.dynamic_image_size = data_args.dynamic_image_size
        config.use_thumbnail = data_args.use_thumbnail
        config.ps_version = model_args.ps_version
        config.min_dynamic_patch = data_args.min_dynamic_patch
        config.max_dynamic_patch = data_args.max_dynamic_patch
        model = InternVLChatModel.from_pretrained(
            model_args.model_name_or_path, torch_dtype=torch.bfloat16, config=config
        )
    else:
        logger.info('Loading ViT-6B...')
        vision_config = InternVisionConfig.from_pretrained(model_args.vision_path)
        vision_config.drop_path_rate = model_args.drop_path_rate
        vision_model = InternVisionModel.from_pretrained(
            model_args.vision_path, torch_dtype=torch.bfloat16, config=vision_config
        )
        logger.info('Loading LLaMA...')
        llm_config = AutoConfig.from_pretrained(
            model_args.llm_path, trust_remote_code=True
        )
        if llm_config.model_type == 'internlm2':
            model_type = InternLM2ForCausalLM
            llm_config.attn_implementation = 'flash_attention_2'  # for InternLM
            logger.info('Using flash_attention_2 for InternLM')
        else:
            model_type = AutoModelForCausalLM
            llm_config._attn_implementation = 'flash_attention_2'  # for LLaMA
            logger.info('Using flash_attention_2 for LLaMA')
        llm = model_type.from_pretrained(
            model_args.llm_path,
            torch_dtype=torch.bfloat16,
            config=llm_config,
            trust_remote_code=True,
        )
        logger.info('Building InternVLChatConfig...')
        internvl_chat_config = InternVLChatConfig(
            vision_config.to_dict(),
            llm_config.to_dict(),
            downsample_ratio=data_args.down_sample_ratio,
            pad2square=data_args.pad2square,
            template=data_args.conv_style,
            select_layer=model_args.vision_select_layer,
            dynamic_image_size=data_args.dynamic_image_size,
            use_thumbnail=data_args.use_thumbnail,
            ps_version=model_args.ps_version,
            min_dynamic_patch=data_args.min_dynamic_patch,
            max_dynamic_patch=data_args.max_dynamic_patch,
        )
        internvl_chat_config.force_image_size = data_args.force_image_size
        logger.info('Building InternVLChatModel...')
        model = InternVLChatModel(internvl_chat_config, vision_model, llm)
    model.prm_token_id = tokenizer.convert_tokens_to_ids(PRM_TOKEN)
    model.reward_token_ids = tokenizer.convert_tokens_to_ids(REWARD_TOKENS)
    model.img_context_token_id = img_context_token_id
    
    # Training stage:
    #   ensemble_prm -> fit reward hypotheses
    #   bayesian_prm -> freeze hypotheses and fit the belief network
    model.config.prm_loss_type = model_args.prm_loss_type
    model.prm_loss_type = model_args.prm_loss_type

    # -------------------------------------------------------------
    # Ensemble PRM configuration.
    #
    # ensemble_prm:
    #   The ensemble architecture is being created/trained now, so
    #   CLI arguments define the architecture.
    #
    # bayesian_prm:
    #   model_name_or_path must point to an existing EnsemblePRM
    #   (or BayesianPRM) checkpoint. The loaded checkpoint config is
    #   the single source of truth for the frozen ensemble.
    # -------------------------------------------------------------
    if model_args.prm_loss_type == 'ensemble_prm' and not is_resume_training:
        model.config.ensemble_prm_num_heads = (
            model_args.ensemble_prm_num_heads
        )
        model.config.ensemble_prm_hidden_dim = (
            model_args.ensemble_prm_hidden_dim
        )
        model.config.ensemble_prm_dropout = (
            model_args.ensemble_prm_dropout
        )
        model.config.ensemble_prm_use_prior_network = (
            model_args.ensemble_prm_use_prior_network
        )
        model.config.ensemble_prm_prior_scale = (
            model_args.ensemble_prm_prior_scale
        )
        model.config.ensemble_prm_bootstrap_prob = (
            model_args.ensemble_prm_bootstrap_prob
        )

        model.ensemble_prm_num_heads = int(
            model_args.ensemble_prm_num_heads
        )
        model.ensemble_prm_hidden_dim = int(
            model_args.ensemble_prm_hidden_dim
        )
        model.ensemble_prm_dropout = float(
            model_args.ensemble_prm_dropout
        )
        model.ensemble_prm_use_prior_network = bool(
            model_args.ensemble_prm_use_prior_network
        )
        model.ensemble_prm_prior_scale = float(
            model_args.ensemble_prm_prior_scale
        )
        model.ensemble_prm_bootstrap_prob = float(
            model_args.ensemble_prm_bootstrap_prob
        )
        model.init_ensemble_prm_head(force_reinit=True)

    elif (
        model_args.prm_loss_type == 'ensemble_prm'
        and is_resume_training
    ):
        required_ensemble_config = (
            'ensemble_prm_num_heads',
            'ensemble_prm_hidden_dim',
            'ensemble_prm_dropout',
            'ensemble_prm_use_prior_network',
            'ensemble_prm_prior_scale',
            'ensemble_prm_bootstrap_prob',
        )

        missing = [
            key
            for key in required_ensemble_config
            if not hasattr(model.config, key)
        ]

        if missing:
            raise RuntimeError(
                "Cannot resume EnsemblePRM because checkpoint config "
                f"is missing fields: {missing}. "
                f"checkpoint={resume_checkpoint}"
            )

        if dist.get_rank() == 0:
            logger.info(
                "Resume EnsemblePRM: using architecture from checkpoint config: "
                f"num_heads={model.config.ensemble_prm_num_heads}, "
                f"hidden_dim={model.config.ensemble_prm_hidden_dim}, "
                f"dropout={model.config.ensemble_prm_dropout}, "
                f"use_prior_network={model.config.ensemble_prm_use_prior_network}, "
                f"prior_scale={model.config.ensemble_prm_prior_scale}, "
                f"bootstrap_prob={model.config.ensemble_prm_bootstrap_prob}"
            )

    elif model_args.prm_loss_type == 'bayesian_prm':
        required_ensemble_config = (
            'ensemble_prm_num_heads',
            'ensemble_prm_hidden_dim',
            'ensemble_prm_dropout',
            'ensemble_prm_use_prior_network',
            'ensemble_prm_prior_scale',
        )

        missing = [
            key
            for key in required_ensemble_config
            if not hasattr(model.config, key)
        ]

        if missing:
            raise RuntimeError(
                "BayesianPRM requires an existing EnsemblePRM checkpoint "
                "whose config contains the ensemble architecture. "
                f"Missing fields: {missing}. "
                f"model_name_or_path={model_args.model_name_or_path}"
            )

        if dist.get_rank() == 0:
            logger.info(
                "BayesianPRM: using ensemble architecture directly from "
                "the loaded checkpoint config: "
                f"num_heads={model.config.ensemble_prm_num_heads}, "
                f"hidden_dim={model.config.ensemble_prm_hidden_dim}, "
                f"dropout={model.config.ensemble_prm_dropout}, "
                f"use_prior_network="
                f"{model.config.ensemble_prm_use_prior_network}, "
                f"prior_scale={model.config.ensemble_prm_prior_scale}"
            )
    # BayesianPRM belief-network hyperparameters.
    if model_args.prm_loss_type == 'bayesian_prm':
        if not is_resume_training:
            # Fresh BayesianPRM training:
            # the EnsemblePRM checkpoint has no trained belief head yet,
            # so CLI defines the new belief architecture/hyperparameters.
            model.config.belief_hidden_dim = model_args.belief_hidden_dim
            model.config.belief_dropout = model_args.belief_dropout
            model.config.belief_beta_kl = model_args.belief_beta_kl
            model.config.belief_use_reward_probs = (
                model_args.belief_use_reward_probs
            )
            model.config.belief_loglik_normalize_by_n = (
                model_args.belief_loglik_normalize_by_n
            )
            model.config.belief_use_conservatism = (
                model_args.belief_use_conservatism
            )
            model.config.belief_conservatism_beta = (
                model_args.belief_conservatism_beta
            )

            model.belief_hidden_dim = int(
                model_args.belief_hidden_dim
            )
            model.belief_dropout = float(
                model_args.belief_dropout
            )
            model.belief_beta_kl = float(
                model_args.belief_beta_kl
            )
            model.belief_use_reward_probs = bool(
                model_args.belief_use_reward_probs
            )
            model.belief_loglik_normalize_by_n = bool(
                model_args.belief_loglik_normalize_by_n
            )
            model.belief_use_conservatism = bool(
                model_args.belief_use_conservatism
            )
            model.belief_conservatism_beta = float(
                model_args.belief_conservatism_beta
            )

        else:
            # Resume BayesianPRM:
            # checkpoint config is the source of truth.
            required_belief_config = (
                'belief_hidden_dim',
                'belief_dropout',
                'belief_beta_kl',
                'belief_use_reward_probs',
                'belief_loglik_normalize_by_n',
                'belief_use_conservatism',
                'belief_conservatism_beta',
            )

            missing = [
                key
                for key in required_belief_config
                if not hasattr(model.config, key)
            ]

            if missing:
                raise RuntimeError(
                    "Cannot resume BayesianPRM because checkpoint config "
                    f"is missing fields: {missing}. "
                    f"checkpoint={resume_checkpoint}"
                )

            if dist.get_rank() == 0:
                logger.info(
                    "Resume BayesianPRM: using belief configuration "
                    "from checkpoint config."
                )
    

    if model_args.prm_loss_type in ('ensemble_prm', 'bayesian_prm'):
        if not hasattr(model, 'init_ensemble_prm_head'):
            raise RuntimeError(
                f"prm_loss_type='{model_args.prm_loss_type}' requires "
                "InternVLChatModel.init_ensemble_prm_head()."
            )

        model.init_ensemble_prm_head(force_reinit=False)

        # Sanity check: the constructed ensemble head must match the CLI/config
        # prior-network setting. This is especially important when loading an
        # ensemble checkpoint into BayesianPRM.
        actual_use_prior = getattr(
            model.ensemble_prm_head,
            "use_prior_network",
            False,
        )
        if (
            model_args.prm_loss_type == 'ensemble_prm'
            and not is_resume_training
        ):
            # Fresh EnsemblePRM: CLI defines the architecture.
            expected_use_prior = bool(
                model_args.ensemble_prm_use_prior_network
            )
        else:
            # EnsemblePRM resume or BayesianPRM:
            # checkpoint config is the source of truth.
            expected_use_prior = bool(
                model.config.ensemble_prm_use_prior_network
            )

        if model_args.prm_loss_type == 'bayesian_prm':
            head = model.ensemble_prm_head

            expected_num_heads = int(
                model.config.ensemble_prm_num_heads
            )
            expected_hidden_dim = int(
                model.config.ensemble_prm_hidden_dim
            )

            if head.num_heads != expected_num_heads:
                raise RuntimeError(
                    "Ensemble checkpoint inconsistency: "
                    f"config num_heads={expected_num_heads}, "
                    f"head num_heads={head.num_heads}."
                )

            if head.hidden_dim != expected_hidden_dim:
                raise RuntimeError(
                    "Ensemble checkpoint inconsistency: "
                    f"config hidden_dim={expected_hidden_dim}, "
                    f"head hidden_dim={head.hidden_dim}."
                )

    if model_args.prm_loss_type == 'ensemble_prm':
        # Make sure only the learned ensemble branch is trainable.
        if hasattr(model.ensemble_prm_head, "learned_parameters"):
            for p in model.ensemble_prm_head.learned_parameters():
                p.requires_grad = True
        else:
            # Backward-compatible fallback for old EnsembleScalarRewardHead.
            for p in model.ensemble_prm_head.parameters():
                p.requires_grad = True

        # Keep the randomized prior branch frozen.
        if hasattr(model.ensemble_prm_head, "freeze_prior_network"):
            model.ensemble_prm_head.freeze_prior_network()

        if (
            getattr(model.ensemble_prm_head, "use_prior_network", False)
            and hasattr(model.ensemble_prm_head, "prior_is_frozen")
        ):
            assert model.ensemble_prm_head.prior_is_frozen(), (
                "Ensemble prior network should be frozen, but some prior "
                "parameters are trainable."
            )

    if model_args.prm_loss_type == 'bayesian_prm':
        # BayesianPRM trains only the belief network.
        # The ensemble PRM is treated as a frozen set of reward hypotheses.

        if not hasattr(model, 'ensemble_prm_head') or model.ensemble_prm_head is None:
            raise RuntimeError(
                "prm_loss_type='bayesian_prm' requires an initialized "
                "ensemble_prm_head. Please load or initialize an ensemble PRM "
                "checkpoint before training the belief head."
            )

        # 1. Freeze the whole ensemble hypothesis set.
        for p in model.ensemble_prm_head.parameters():
            p.requires_grad = False

        # 2. Extra safety: if the ensemble head contains a randomized prior
        #    branch, keep it frozen even if previous code accidentally
        #    performed blanket unfreezing.
        if hasattr(model.ensemble_prm_head, "freeze_prior_network"):
            model.ensemble_prm_head.freeze_prior_network()

        if (
            getattr(model.ensemble_prm_head, "use_prior_network", False)
            and hasattr(model.ensemble_prm_head, "prior_is_frozen")
        ):
            assert model.ensemble_prm_head.prior_is_frozen(), (
                "BayesianPRM expects the ensemble prior network to be frozen, "
                "but some prior parameters are trainable."
            )

        # 3. Initialize the belief head.
        if not hasattr(model, 'init_belief_head'):
            raise RuntimeError(
                "prm_loss_type='bayesian_prm' requires "
                "InternVLChatModel.init_belief_head()."
            )

        model.init_belief_head(force_reinit=True)

        # 4. Train only the belief head.
        for p in model.belief_head.parameters():
            p.requires_grad = True

    if dist.get_rank() == 0:
        logger.info(f'Using PRM loss type: {model_args.prm_loss_type}')
        if model_args.prm_loss_type == 'ensemble_prm':
            logger.info(
                f'Using ensemble PRM head: '
                f'num_heads={model_args.ensemble_prm_num_heads}, '
                f'hidden_dim={model_args.ensemble_prm_hidden_dim}, '
                f'dropout={model_args.ensemble_prm_dropout}, '
                f'use_prior_network={model_args.ensemble_prm_use_prior_network}, '
                f'prior_scale={model_args.ensemble_prm_prior_scale}'
            )

    assert model.config.downsample_ratio == data_args.down_sample_ratio

    if model_args.mlp_path is not None:
        logger.info('Loading pretrained MLP projector...')
        state_dict = torch.load(model_args.mlp_path, map_location='cpu')
        message = model.mlp1.load_state_dict(state_dict)
        logger.info(message)
    logger.info('Finished')

    patch_size = model.config.vision_config.patch_size
    logger.info(f'model.config.force_image_size: {model.config.force_image_size}')
    logger.info(f'data_args.force_image_size: {data_args.force_image_size}')
    logger.info(
        f'model.config.vision_config.image_size: {model.config.vision_config.image_size}'
    )
    if model.config.vision_config.image_size != data_args.force_image_size:
        logger.info(
            f'Resizing position embedding from '
            f'{model.config.vision_config.image_size} '
            f'to {data_args.force_image_size}...'
        )
        model.vision_model.resize_pos_embeddings(
            old_size=model.config.vision_config.image_size,
            new_size=data_args.force_image_size,
            patch_size=patch_size,
        )
        model.config.vision_config.image_size = data_args.force_image_size
    model.config.force_image_size = data_args.force_image_size
    model.num_image_token = int(
        (data_args.force_image_size // patch_size) ** 2
        * (data_args.down_sample_ratio**2)
    )

    if num_new_tokens > 0:
        model.language_model.resize_token_embeddings(len(tokenizer))
        output_embeddings = model.language_model.get_output_embeddings().weight.data
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True
        )
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

        model.config.llm_config.vocab_size = len(tokenizer)
        model.language_model.config.vocab_size = len(tokenizer)

    model.language_model.config.use_cache = False
    model.vision_model.gradient_checkpointing = True
    model.vision_model.encoder.gradient_checkpointing = True
    if model_args.grad_checkpoint:
        model.language_model._set_gradient_checkpointing()

    train_dataset = build_datasets(
        data_args,
        tokenizer,
        tcs_loader,
        model,
        group_by_length=training_args.group_by_length,
        dynamic_image_size=data_args.dynamic_image_size,
        use_thumbnail=data_args.use_thumbnail,
        min_dynamic_patch=data_args.min_dynamic_patch,
        max_dynamic_patch=data_args.max_dynamic_patch,
        normalize_type=data_args.normalize_type,
        min_num_frame=data_args.min_num_frame,
        max_num_frame=data_args.max_num_frame,
    )

    raw_train_len = len(train_dataset)

    train_dataset, prm_split_info = split_prm_train_dataset(
        train_dataset=train_dataset,
        split_enable=bool(data_args.prm_data_split_enable),
        split_ratio=float(data_args.prm_data_split_ratio),
        split_seed=int(data_args.prm_data_split_seed),
        split_part=str(data_args.prm_data_split_part),
        prm_loss_type=str(model_args.prm_loss_type),
    )

    if dist.get_rank() == 0:
        logger.info(
            "PRM data split: "
            f"enable={data_args.prm_data_split_enable}, "
            f"ratio={data_args.prm_data_split_ratio}, "
            f"seed={data_args.prm_data_split_seed}, "
            f"requested_part={data_args.prm_data_split_part}, "
            f"resolved_part={prm_split_info['resolved_part']}, "
            f"prm_loss_type={model_args.prm_loss_type}, "
            f"raw_train_len={raw_train_len}, "
            f"used_train_len={len(train_dataset)}, "
            f"ensemble_len={prm_split_info['ensemble_len']}, "
            f"belief_len={prm_split_info['belief_len']}"
        )

    def _freeze_params(module):
        for param in module.parameters():
            param.requires_grad = False

    if model_args.freeze_backbone:
        # model.vision_model = model.vision_model.eval()
        _freeze_params(model.vision_model)

    if model_args.freeze_llm:
        model.language_model = model.language_model.eval()
        _freeze_params(model.language_model)

    if model_args.unfreeze_lm_head:
        model.language_model.lm_head.requires_grad = True

    if model_args.use_backbone_lora:
        model.wrap_backbone_lora(
            r=model_args.use_backbone_lora, lora_alpha=2 * model_args.use_backbone_lora
        )
        model.config.use_backbone_lora = model_args.use_backbone_lora

    if model_args.use_llm_lora:
        model.wrap_llm_lora(
            r=model_args.use_llm_lora, lora_alpha=2 * model_args.use_llm_lora
        )
        model.config.use_llm_lora = model_args.use_llm_lora

    if model_args.freeze_mlp:
        _freeze_params(model.mlp1)

    if model_args.unfreeze_vit_layers != 0:
        layers = model.vision_model.encoder.layers[model_args.unfreeze_vit_layers :]
        for k, v in layers.named_parameters():
            logger.info(f'Unfreezing ViT layer: {k}')
            v.requires_grad = True

    if model_args.prm_loss_type == 'bayesian_prm':
        if not hasattr(model, 'belief_head') or model.belief_head is None:
            if not hasattr(model, 'init_belief_head'):
                raise RuntimeError(
                    "prm_loss_type='bayesian_prm' requires "
                    "InternVLChatModel.init_belief_head()."
                )
            model.init_belief_head(force_reinit=False)

        # Freeze everything first.
        for param in model.parameters():
            param.requires_grad = False

        # Train only the belief network.
        for param in model.belief_head.parameters():
            param.requires_grad = True

        if dist.get_rank() == 0:
            logger.info(
                "BayesianPRM mode: froze all base/reward parameters and "
                "enabled training only for belief_head."
            )

    # print trainable parameters
    if dist.get_rank() == 0:
        for name, param in model.named_parameters():
            if param.requires_grad:
                logger.info(name)

    # set seed for torch dataloaders
    set_seed(training_args.seed)

    if data_args.use_packed_ds:
        collator = partial(
            packed_collate_fn,
            data_collator=concat_pad_data_collator_prm,
            max_item_length=data_args.max_packed_tokens if data_args.strict_mode else 0,
            micro_num=training_args.train_batch_size,
            len2weight=partial(len2weight, loss_reduction=data_args.loss_reduction),
            loss_reduction_all_gather=data_args.loss_reduction_all_gather,
        )
    else:
        collator = concat_pad_data_collator_prm

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        tokenizer=tokenizer,
        data_collator=collator,
    )
    
    if model_args.prm_loss_type in ('ensemble_prm', 'bayesian_prm'):
        trainer.remove_callback(transformers.integrations.WandbCallback)
        trainer.add_callback(PRMStatsCallback)
        trainer.add_callback(transformers.integrations.WandbCallback)


    # Training
    if training_args.do_train:
        train_result = trainer.train(resume_from_checkpoint=resume_checkpoint)

        metrics = train_result.metrics
        try:
            metrics['train_samples'] = len(train_dataset)
        except:
            metrics['train_samples'] = -1

        trainer.log_metrics('train', metrics)
        trainer.save_metrics('train', metrics)
        #trainer.save_state()


if __name__ == '__main__':
    main()
