# --------------------------------------------------------
# InternVL
# Copyright (c) 2024 OpenGVLab
# Licensed under The MIT License [see LICENSE for details]
# --------------------------------------------------------



import warnings
from typing import List, Optional, Tuple, Union

import math
import torch.distributed as dist
import torch.utils.checkpoint
import transformers
from peft import LoraConfig, get_peft_model
from torch import nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import (AutoModel, GenerationConfig, LlamaForCausalLM,
                          LlamaTokenizer, Qwen2ForCausalLM)
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.utils import ModelOutput, logging

from internvl.conversation import get_conv_template
from internvl.model.internlm2.modeling_internlm2 import InternLM2ForCausalLM
from internvl.model.phi3.modeling_phi3 import Phi3ForCausalLM

from .configuration_internvl_chat import InternVLChatConfig
from .modeling_intern_vit import InternVisionModel, has_flash_attn
from .ensemble_prm_head import EnsembleScalarRewardHead
from .bayesian_prm_head import BayesianBeliefHead

logger = logging.get_logger(__name__)


def version_cmp(v1, v2, op='eq'):
    import operator

    from packaging import version

    op_func = getattr(operator, op)
    return op_func(version.parse(v1), version.parse(v2))


class InternVLChatModel(PreTrainedModel):
    config_class = InternVLChatConfig
    main_input_name = 'pixel_values'
    base_model_prefix = 'language_model'
    _no_split_modules = [
        'InternVisionModel',
        'LlamaDecoderLayer',
        'InternLM2DecoderLayer',
        'Phi3DecoderLayer',
        'Qwen2DecoderLayer',
    ]
    _supports_flash_attn_2 = True
    supports_gradient_checkpointing = True

    def __init__(
        self,
        config: InternVLChatConfig,
        vision_model=None,
        language_model=None,
        use_flash_attn=True,
    ):
        super().__init__(config)

        assert version_cmp(transformers.__version__, '4.37.0', 'ge')
        image_size = config.force_image_size or config.vision_config.image_size
        patch_size = config.vision_config.patch_size
        self.patch_size = patch_size
        self.select_layer = config.select_layer
        self.template = config.template
        self.num_image_token = int(
            (image_size // patch_size) ** 2 * (config.downsample_ratio**2)
        )
        self.downsample_ratio = config.downsample_ratio
        self.ps_version = config.ps_version
        self.llm_arch_name = config.llm_config.architectures[0]
        # Enable Flash Attention if supported, otherwise fall back to eager attention.
        use_flash_attn = use_flash_attn if has_flash_attn else False
        config.vision_config.use_flash_attn = True if use_flash_attn else False
        config.llm_config.attn_implementation = (
            'flash_attention_2' if use_flash_attn else 'eager'
        )

        logger.info(f'num_image_token: {self.num_image_token}')
        logger.info(f'ps_version: {self.ps_version}')
        if vision_model is not None:
            self.vision_model = vision_model
        else:
            self.vision_model = InternVisionModel(config.vision_config)
        if language_model is not None:
            self.language_model = language_model
        else:
            if config.llm_config.architectures[0] == 'LlamaForCausalLM':
                self.language_model = LlamaForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'InternLM2ForCausalLM':
                self.language_model = InternLM2ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Phi3ForCausalLM':
                self.language_model = Phi3ForCausalLM(config.llm_config)
            elif config.llm_config.architectures[0] == 'Qwen2ForCausalLM':
                self.language_model = Qwen2ForCausalLM(config.llm_config)
            else:
                raise NotImplementedError(
                    f'{config.llm_config.architectures[0]} is not implemented.'
                )

        vit_hidden_size = config.vision_config.hidden_size
        llm_hidden_size = config.llm_config.hidden_size

        self.mlp1 = nn.Sequential(
            nn.LayerNorm(vit_hidden_size * int(1 / self.downsample_ratio) ** 2),
            nn.Linear(
                vit_hidden_size * int(1 / self.downsample_ratio) ** 2, llm_hidden_size
            ),
            nn.GELU(),
            nn.Linear(llm_hidden_size, llm_hidden_size),
        )

        self.img_context_token_id = None
        self.conv_template = get_conv_template(self.template)
        if hasattr(config, 'system_message'):
            self.system_message = config.system_message
        else:
            self.system_message = self.conv_template.system_message
        self.num_samples = 0
        self.prm_loss_type = getattr(config, "prm_loss_type", "ensemble_prm")
        self.prm_eps = 1e-6
        self.prm_token_id = None
        self.reward_token_ids = None

        # The ensemble is loaded from the parent checkpoint and frozen during
        # BayesianPRM training. Only the belief head receives gradients.
        self.ensemble_prm_num_heads = int(getattr(config, "ensemble_prm_num_heads", 8))
        self.ensemble_prm_hidden_dim = int(getattr(config, "ensemble_prm_hidden_dim", 128))
        self.ensemble_prm_dropout = float(getattr(config, "ensemble_prm_dropout", 0.0))
        self.ensemble_prm_use_prior_network = bool(getattr(config, "ensemble_prm_use_prior_network", False))
        self.ensemble_prm_prior_scale = float(getattr(config, "ensemble_prm_prior_scale", 1.0))
        self.ensemble_prm_bootstrap_prob = float(getattr(config, "ensemble_prm_bootstrap_prob", 1.0))
        self.ensemble_prm_head = None

        self.belief_hidden_dim = int(getattr(config, "belief_hidden_dim", 256))
        self.belief_dropout = float(getattr(config, "belief_dropout", 0.0))
        self.belief_beta_kl = float(getattr(config, "belief_beta_kl", 0.1))
        self.belief_use_reward_probs = bool(getattr(config, "belief_use_reward_probs", True))
        self.belief_loglik_normalize_by_n = bool(
            getattr(config, "belief_loglik_normalize_by_n", True)
        )
        self.belief_use_conservatism = bool(
            getattr(config, "belief_use_conservatism", False)
        )
        self.belief_conservatism_beta = float(
            getattr(config, "belief_conservatism_beta", 0.1)
        )
        self.belief_head = None

        self.init_ensemble_prm_head()
        if self.prm_loss_type == "bayesian_prm":
            self.init_belief_head()

        if config.use_backbone_lora:
            self.wrap_backbone_lora(
                r=config.use_backbone_lora, lora_alpha=2 * config.use_backbone_lora
            )

        if config.use_llm_lora:
            self.wrap_llm_lora(
                r=config.use_llm_lora, lora_alpha=2 * config.use_llm_lora
            )

    def wrap_backbone_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        lora_config = LoraConfig(
            r=r,
            target_modules=['attn.qkv', 'attn.proj', 'mlp.fc1', 'mlp.fc2'],
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )
        self.vision_model = get_peft_model(self.vision_model, lora_config)
        self.vision_model.print_trainable_parameters()

    def wrap_llm_lora(self, r=128, lora_alpha=256, lora_dropout=0.05):
        # Determine the target modules based on the architecture of the language model
        if self.llm_arch_name == 'InternLM2ForCausalLM':
            target_modules = [
                'attention.wqkv',
                'attention.wo',
                'feed_forward.w1',
                'feed_forward.w2',
                'feed_forward.w3',
            ]
        elif self.llm_arch_name == 'Phi3ForCausalLM':
            target_modules = [
                'mlp.down_proj',
                'mlp.gate_up_proj',
                'self_attn.o_proj',
                'self_attn.qkv_proj',
            ]
        elif self.llm_arch_name in ['Qwen2ForCausalLM', 'LlamaForCausalLM']:
            target_modules = [
                'self_attn.q_proj',
                'self_attn.k_proj',
                'self_attn.v_proj',
                'self_attn.o_proj',
                'mlp.gate_proj',
                'mlp.down_proj',
                'mlp.up_proj',
            ]
        else:
            raise NotImplemented
        lora_config = LoraConfig(
            r=r,
            target_modules=target_modules,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            task_type='CAUSAL_LM',
        )
        self.language_model = get_peft_model(self.language_model, lora_config)
        self.language_model.enable_input_require_grads()
        self.language_model.print_trainable_parameters()

    def init_ensemble_prm_head(self, force_reinit: bool = False):
        """
        Initialize the ensemble scalar reward head.

        The head is kept separate so it can be frozen independently from the
        Bayesian belief network.
        """
        if self.ensemble_prm_head is not None and not force_reinit:
            return self.ensemble_prm_head

        hidden_size = getattr(
            self.language_model.config,
            "hidden_size",
            self.config.llm_config.hidden_size,
        )

        self.ensemble_prm_head = EnsembleScalarRewardHead(
            hidden_size=hidden_size,
            num_heads=self.ensemble_prm_num_heads,
            hidden_dim=self.ensemble_prm_hidden_dim,
            dropout=self.ensemble_prm_dropout,
            use_prior_network=self.ensemble_prm_use_prior_network,
            prior_scale=self.ensemble_prm_prior_scale,
        )

        # If the base model has already been moved to a device/dtype, move the
        # new head accordingly. Avoid touching meta tensors during ZeRO-3 init.
        try:
            ref_param = next(self.language_model.parameters())
            if not getattr(ref_param, "is_meta", False):
                self.ensemble_prm_head.to(
                    device=ref_param.device,
                    dtype=ref_param.dtype,
                )
        except StopIteration:
            pass

        if hasattr(self.ensemble_prm_head, "freeze_prior_network"):
            self.ensemble_prm_head.freeze_prior_network()

        return self.ensemble_prm_head
    
    def init_belief_head(self, force_reinit: bool = False):
        """
        Initialize the BayesianPRM belief head.

        The belief head is trained after the ensemble PRM has been trained and frozen.
        It predicts q_phi(z=m | c) over frozen ensemble reward heads.
        """
        if self.belief_head is not None and not force_reinit:
            return self.belief_head

        hidden_size = getattr(
            self.language_model.config,
            "hidden_size",
            self.config.llm_config.hidden_size,
        )

        self.belief_head = BayesianBeliefHead(
            hidden_size=hidden_size,
            num_heads=self.ensemble_prm_num_heads,
            belief_hidden_dim=self.belief_hidden_dim,
            dropout=self.belief_dropout,
            use_reward_probs=self.belief_use_reward_probs,
        )

        # If the base model has already been moved to a device/dtype, move the
        # new head accordingly. Avoid touching meta tensors during ZeRO-3 init.
        try:
            ref_param = next(self.language_model.parameters())
            if not getattr(ref_param, "is_meta", False):
                self.belief_head.to(
                    device=ref_param.device,
                    dtype=ref_param.dtype,
                )
        except StopIteration:
            pass

        return self.belief_head

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        image_flags: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        statistics: Optional[torch.LongTensor] = None,
        loss_weight: Optional[List] = None,
        loss_reduction_all_gather: Optional[bool] = False,
        prm_counts_k: Optional[torch.Tensor] = None,
        prm_counts_n: Optional[torch.Tensor] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        image_flags = image_flags.squeeze(-1)
        input_embeds = self.language_model.get_input_embeddings()(input_ids).clone()

        vit_embeds = self.extract_feature(pixel_values)
        vit_embeds = vit_embeds[image_flags == 1]
        vit_batch_size = pixel_values.shape[0]

        B, N, C = input_embeds.shape
        input_embeds = input_embeds.reshape(B * N, C)

        if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
            print(
                f'dynamic ViT batch size: {vit_batch_size}, images per sample: {vit_batch_size / B}, dynamic token length: {N}'
            )
            if statistics is not None:
                num_samples, num_padding_tokens, num_padding_images = (
                    statistics.tolist()
                )
                self.num_samples += num_samples
                print(
                    f'total_samples={self.num_samples}, {num_samples=}, {num_padding_tokens=}, {num_padding_images=}'
                )

        input_ids = input_ids.reshape(B * N)
        selected = input_ids == self.img_context_token_id
        try:
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds.reshape(
                -1, C
            )
            ignore_flag = False
        except Exception as e:
            vit_embeds = vit_embeds.reshape(-1, C)
            print(
                f'warning: {e}, input_embeds[selected].shape={input_embeds[selected].shape}, '
                f'vit_embeds.shape={vit_embeds.shape}'
            )
            n_token = selected.sum()
            input_embeds[selected] = input_embeds[selected] * 0.0 + vit_embeds[:n_token]
            ignore_flag = True

        input_embeds = input_embeds.reshape(B, N, C)

        # If we need kappa from '<prm>' hidden states, prefer retrieving only the last hidden state
        # (without storing all layer hidden_states) to avoid OOM for long sequences.
        need_prm_hidden = (
            labels is not None
            and loss_weight is None
            and (
                (prm_counts_k is not None and prm_counts_n is not None)
                or getattr(self, "prm_loss_type", "ensemble_prm")
                in ("ensemble_prm", "bayesian_prm")
            )
        )

        last_hidden_state = None
        if need_prm_hidden:
            # Support PEFT-wrapped models: try common attribute paths to reach the base decoder.
            lm = self.language_model
            base = getattr(lm, "model", None) or getattr(lm, "transformer", None)
            if base is None and hasattr(lm, "base_model"):
                bm = lm.base_model
                base = getattr(bm, "model", None) or getattr(bm, "transformer", None)
            if base is None:
                # Fallback: request full hidden_states (may OOM on long sequences).
                outputs = self.language_model(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=True,
                    return_dict=True,
                )
                logits = outputs.logits
                last_hidden_state = outputs.hidden_states[-1]
            else:
                base_outputs = base(
                    inputs_embeds=input_embeds,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                    output_attentions=output_attentions,
                    output_hidden_states=False,
                    return_dict=True,
                )
                last_hidden_state = getattr(base_outputs, "last_hidden_state", None)
                if last_hidden_state is None:
                    last_hidden_state = base_outputs[0]
                lm_head = self.language_model.get_output_embeddings()
                logits = lm_head(last_hidden_state)
                # Create a lightweight outputs-like object for downstream return.
                outputs = CausalLMOutputWithPast(
                    loss=None,
                    logits=logits,
                    past_key_values=getattr(base_outputs, "past_key_values", None),
                    hidden_states=None,
                    attentions=getattr(base_outputs, "attentions", None),
                )
        else:
            outputs = self.language_model(
                inputs_embeds=input_embeds,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            logits = outputs.logits

        loss = None
        if labels is not None and loss_weight is not None:
            loss_weight = torch.tensor(
                loss_weight, dtype=torch.float32, device=labels.device
            )
            # Shift so that tokens < n predict n
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            shift_weights = loss_weight[..., 1:].contiguous()
            # Flatten the tokens
            loss_fct = CrossEntropyLoss(reduction='none')
            shift_logits = shift_logits.view(-1, self.language_model.config.vocab_size)
            shift_labels = shift_labels.view(-1)
            shift_weights = shift_weights.view(-1)
            # Enable model parallelism
            shift_labels = shift_labels.to(shift_logits.device)
            shift_weights = shift_weights.to(shift_logits.device)
            loss = loss_fct(shift_logits, shift_labels)

            shift_weights_sum = shift_weights.sum()
            if loss_reduction_all_gather:
                dist.all_reduce(shift_weights_sum, op=dist.ReduceOp.AVG)

            loss = loss * shift_weights
            loss = loss.sum() / shift_weights_sum
            if ignore_flag:
                loss = loss * 0.0
        elif labels is not None:
            placeholder_mask = input_ids == self.prm_token_id
            selected_logits = logits.contiguous().view(
                -1, self.language_model.config.vocab_size
            )[placeholder_mask]
            selected_labels = labels.contiguous().view(-1)[placeholder_mask]
            selected_logits = selected_logits[..., self.reward_token_ids]
            if selected_logits.size(0) == 0:
                loss = logits.sum() * 0.0
                if ignore_flag:
                    loss = loss * 0.0
                if not return_dict:
                    output = (logits,) + outputs[1:]
                    return (loss,) + output
                return CausalLMOutputWithPast(
                    loss=loss,
                    logits=logits,
                    past_key_values=outputs.past_key_values,
                    hidden_states=outputs.hidden_states,
                    attentions=outputs.attentions,
                )

            prm_loss_type = getattr(self, "prm_loss_type", "ensemble_prm")

            use_ensemble_prm = prm_loss_type == "ensemble_prm"
            use_bayesian_prm = prm_loss_type == "bayesian_prm"
            
            if use_ensemble_prm:
                if last_hidden_state is None:
                    raise RuntimeError(
                        "ensemble_prm requires last_hidden_state. "
                        "Please make sure output_hidden_states=True when prm_loss_type='ensemble_prm'."
                    )

                if self.ensemble_prm_head is None:
                    raise RuntimeError(
                        "ensemble_prm_head is not initialized. "
                        "Call model.init_ensemble_prm_head() before constructing Trainer/optimizer."
                    )

                flat_prm_mask = placeholder_mask.contiguous().view(-1)

                last_h = last_hidden_state.contiguous().view(
                    -1, last_hidden_state.shape[-1]
                )
                prm_h = last_h[flat_prm_mask]  # [M, H]

                if prm_h.numel() == 0:
                    loss = logits.sum() * 0.0
                    with torch.no_grad():
                        self._prm_last_stats = {
                            "ensemble_prm_loss": 0.0,
                            "ensemble_reward_mean": 0.0,
                            "ensemble_reward_std": 0.0,
                            "ensemble_target_mean": 0.0,
                            "valid_prm_count": 0.0,
                        }
                else:
                    # selected_labels: [M], soft PRM target in [0, 1], usually K / N.
                    target = selected_labels.to(device=prm_h.device, dtype=torch.float32).clamp(0.0, 1.0)

                    # ensemble_logits: [E, M]
                    ensemble_logits = self.ensemble_prm_head(prm_h)

                    # target_expand: [E, M]
                    target_expand = target[None, :].expand_as(ensemble_logits)

                    # loss_per_head: [E, M]
                    loss_per_head = F.binary_cross_entropy_with_logits(
                        ensemble_logits.float(),
                        target_expand.float(),
                        reduction="none",
                    )

                    bootstrap_prob = float(getattr(self, "ensemble_prm_bootstrap_prob", 1.0))

                    if self.training and bootstrap_prob < 1.0:
                        bootstrap_mask = (
                            torch.rand_like(loss_per_head.float()) < bootstrap_prob
                        ).float()

                        denom = bootstrap_mask.sum().clamp_min(1.0)
                        cls_loss = (loss_per_head.float() * bootstrap_mask).sum() / denom
                    else:
                        bootstrap_mask = None
                        cls_loss = loss_per_head.mean()

                    loss = cls_loss

                    with torch.no_grad():
                        ensemble_probs = torch.sigmoid(ensemble_logits.float())  # [E, M]
                        reward_mean = ensemble_probs.mean(dim=0)  # [M]
                        reward_std = ensemble_probs.std(dim=0, unbiased=False)  # [M]

                        stats = {
                            "ensemble_prm_loss": float(cls_loss.detach().item()),
                            "ensemble_reward_mean": float(reward_mean.mean().detach().item()),
                            "ensemble_reward_std": float(reward_std.mean().detach().item()),
                            "ensemble_target_mean": float(target.mean().detach().item()),
                            "ensemble_bootstrap_prob": float(bootstrap_prob),
                            "valid_prm_count": float(prm_h.size(0)),
                        }

                        if bootstrap_mask is not None:
                            stats["ensemble_bootstrap_keep_ratio"] = float(
                                bootstrap_mask.mean().detach().item()
                            )

                        self._prm_last_stats = stats

            elif use_bayesian_prm:
                if prm_counts_k is None or prm_counts_n is None:
                    raise RuntimeError(
                        "bayesian_prm requires prm_counts_k and prm_counts_n. "
                        "Please use VisualPRM-style count supervision."
                    )

                if last_hidden_state is None:
                    raise RuntimeError(
                        "bayesian_prm requires last_hidden_state. "
                        "Please make sure hidden states are available for PRM markers."
                    )

                if self.ensemble_prm_head is None:
                    raise RuntimeError(
                        "ensemble_prm_head is not initialized. "
                        "BayesianPRM requires a frozen ensemble reward head."
                    )

                if self.belief_head is None:
                    raise RuntimeError(
                        "belief_head is not initialized. "
                        "Call model.init_belief_head() before BayesianPRM training."
                    )

                flat_prm_mask = placeholder_mask.contiguous().view(-1)
                last_h = last_hidden_state.contiguous().view(
                    -1, last_hidden_state.shape[-1]
                )
                prm_h_all = last_h[flat_prm_mask]  # [P_all, H]

                selected_k = prm_counts_k.contiguous().view(-1)[placeholder_mask]
                selected_n = prm_counts_n.contiguous().view(-1)[placeholder_mask]

                valid_kn = (
                    (selected_n > 0)
                    & (selected_k >= 0)
                    & (selected_k <= selected_n)
                )

                if (prm_h_all.numel() == 0) or (not valid_kn.any()):
                    loss = logits.sum() * 0.0
                    with torch.no_grad():
                        self._prm_last_stats = {
                            "bayesian_prm_loss": 0.0,
                            "bayesian_expected_loglik": 0.0,
                            "bayesian_kl": 0.0,

                            # Final posterior stats. When conservatism is disabled,
                            # final posterior equals reliability posterior.
                            "bayesian_entropy": 0.0,
                            "bayesian_weight_top1_mean": 0.0,
                            "bayesian_weight_max": 0.0,
                            "bayesian_weight_min": 0.0,
                            "bayesian_reward_mean": 0.0,
                            "bayesian_reward_std": 0.0,
                            "bayesian_uncertainty_mean": 0.0,

                            # Explicit reliability / hybrid stats.
                            "bayesian_rel_reward_mean": 0.0,
                            "bayesian_post_reward_mean": 0.0,
                            "bayesian_rel_entropy": 0.0,
                            "bayesian_post_entropy": 0.0,

                            # Conservatism settings.
                            "bayesian_use_conservatism": float(
                                bool(getattr(self, "belief_use_conservatism", False))
                            ),
                            "bayesian_conservatism_active": 0.0,
                            "bayesian_conservatism_beta": float(
                                getattr(self, "belief_conservatism_beta", 0.1)
                            ),

                            "valid_prm_count": 0.0,
                        }
                else:
                    prm_h = prm_h_all[valid_kn]  # [P, H]
                    selected_k = selected_k[valid_kn].to(
                        device=prm_h.device, dtype=torch.float32
                    )
                    selected_n = selected_n[valid_kn].to(
                        device=prm_h.device, dtype=torch.float32
                    )

                    # Frozen ensemble likelihood source.
                    # ensemble_logits: [E, P]
                    with torch.no_grad():
                        ensemble_logits = self.ensemble_prm_head(prm_h.detach())
                        ensemble_probs = torch.sigmoid(ensemble_logits.float()).clamp(
                            self.prm_eps,
                            1.0 - self.prm_eps,
                        )

                    # mu: [P, E], where E is the number of ensemble heads.
                    mu = ensemble_probs.transpose(0, 1).contiguous()

                    # Train only the belief network through this loss.  We detach the
                    # marker hidden states so the belief ELBO does not update the reward
                    # LLM/projector through prm_h.
                    belief_logits = self.belief_head(
                        prm_h.detach(),
                        mu.detach(),
                    )  # [P, E]

                    # Reliability posterior:
                    # alpha_rel = q_phi(z=m | c_t).
                    #
                    # This is the only posterior used in the ELBO-style training loss.
                    rel_weights = F.softmax(
                        belief_logits.float(),
                        dim=-1,
                    )

                    # Count log-likelihood under each frozen ensemble member:
                    # log p(K | N, mu_m) = K log mu_m + (N-K) log(1-mu_m)
                    # The combinatorial term log C(N,K) is omitted because it is constant
                    # w.r.t. ensemble member m and belief parameters phi.
                    log_lik = (
                        selected_k[:, None] * torch.log(mu)
                        + (selected_n[:, None] - selected_k[:, None])
                        * torch.log(1.0 - mu)
                    )

                    if bool(getattr(self, "belief_loglik_normalize_by_n", True)):
                        log_lik = log_lik / selected_n[:, None].clamp_min(1.0)

                    # Reliability-posterior ELBO loss.
                    # Do NOT use post_weights here; otherwise the belief head would no
                    # longer learn the reliability posterior.
                    expected_loglik = (rel_weights * log_lik).sum(dim=-1)

                    num_heads = rel_weights.shape[-1]
                    log_rel_weights = torch.log(
                        rel_weights.clamp_min(self.prm_eps)
                    )

                    # KL(q_phi(z|c) || Uniform(z)) = sum_m w_m log(M w_m)
                    kl_to_uniform = (
                        rel_weights
                        * (log_rel_weights + math.log(float(num_heads)))
                    ).sum(dim=-1)

                    belief_beta_kl = float(
                        getattr(self, "belief_beta_kl", 0.1)
                    )

                    bayesian_prm_loss = (
                        -expected_loglik + belief_beta_kl * kl_to_uniform
                    ).mean()

                    loss = bayesian_prm_loss

                    with torch.no_grad():
                        # --------------------------------------------------------------
                        # Conservatism-aware belief calibration.
                        #
                        # Treat the reliability-aware belief alpha_rel as the prior and
                        # perform a second-stage exponential reweighting:
                        #
                        #   alpha_post_m
                        #       ∝ alpha_rel_m * exp(-mu_m / beta_2)
                        #
                        # Equivalently, in log space:
                        #
                        #   alpha_post
                        #       = softmax(log(alpha_rel) - mu / beta_2).
                        # --------------------------------------------------------------
                        use_conservatism = bool(
                            getattr(self, "belief_use_conservatism", False)
                        )
                        conservatism_beta = float(
                            getattr(self, "belief_conservatism_beta", 0.1)
                        )

                        if use_conservatism:
                            temperature = max(
                                conservatism_beta,
                                float(self.prm_eps),
                            )

                            # Sequential Bayesian-style update:
                            # reliability belief -> conservatism-aware final belief.
                            post_weights = F.softmax(
                                log_rel_weights - mu / temperature,
                                dim=-1,
                            )

                            conservatism_active = 1.0
                        else:
                            post_weights = rel_weights
                            conservatism_active = 0.0
                        # --------------------------------------------------------------
                        # Reliability reward and final hybrid reward.
                        # --------------------------------------------------------------
                        rel_reward_mean = (rel_weights * mu).sum(dim=-1)  # [P]
                        post_reward_mean = (post_weights * mu).sum(dim=-1)  # [P]

                        # By convention, legacy fields without "rel_" or "post_" refer to
                        # the final posterior used for reward computation.
                        reward_mean = post_reward_mean

                        weighted_var = (
                            post_weights * (mu - reward_mean[:, None]).pow(2)
                        ).sum(dim=-1)
                        weighted_uncertainty = torch.sqrt(
                            weighted_var.clamp_min(0.0)
                        )

                        log_post_weights = torch.log(
                            post_weights.clamp_min(self.prm_eps)
                        )

                        rel_entropy = -(
                            rel_weights * log_rel_weights
                        ).sum(dim=-1)

                        post_entropy = -(
                            post_weights * log_post_weights
                        ).sum(dim=-1)

                        top1_weight = post_weights.max(dim=-1).values

                        stats = {
                            "bayesian_prm_loss": float(
                                bayesian_prm_loss.detach().item()
                            ),
                            "bayesian_expected_loglik": float(
                                expected_loglik.mean().detach().item()
                            ),
                            "bayesian_kl": float(
                                kl_to_uniform.mean().detach().item()
                            ),

                            # Final posterior stats.
                            "bayesian_entropy": float(
                                post_entropy.mean().detach().item()
                            ),
                            "bayesian_weight_top1_mean": float(
                                top1_weight.mean().detach().item()
                            ),
                            "bayesian_weight_max": float(
                                post_weights.max().detach().item()
                            ),
                            "bayesian_weight_min": float(
                                post_weights.min().detach().item()
                            ),
                            "bayesian_reward_mean": float(
                                reward_mean.mean().detach().item()
                            ),
                            "bayesian_reward_std": float(
                                reward_mean.std(unbiased=False).detach().item()
                            ),
                            "bayesian_uncertainty_mean": float(
                                weighted_uncertainty.mean().detach().item()
                            ),

                            # Explicit reliability / hybrid reward stats.
                            "bayesian_rel_reward_mean": float(
                                rel_reward_mean.mean().detach().item()
                            ),
                            "bayesian_post_reward_mean": float(
                                post_reward_mean.mean().detach().item()
                            ),
                            "bayesian_rel_entropy": float(
                                rel_entropy.mean().detach().item()
                            ),
                            "bayesian_post_entropy": float(
                                post_entropy.mean().detach().item()
                            ),

                            # Conservatism settings.
                            "bayesian_use_conservatism": float(
                                use_conservatism
                            ),
                            "bayesian_conservatism_active": float(
                                conservatism_active
                            ),
                            "bayesian_conservatism_beta": float(
                                conservatism_beta
                            ),

                            "bayesian_target_mean": float(
                                (selected_k / selected_n).mean().detach().item()
                            ),
                            "valid_prm_count": float(prm_h.size(0)),
                        }
                        
                        rel_weight_mean = rel_weights.mean(dim=0)  # [E]

                        for head_idx, weight in enumerate(rel_weight_mean):
                            stats[f"bayesian_rel_weight_head_{head_idx}"] = float(
                                weight.item()
                            )

            else:
                raise RuntimeError(
                    "Unsupported PRM mode. Use 'ensemble_prm' for stage 1 "
                    "or 'bayesian_prm' for stage 2."
                )

            if ignore_flag:
                loss = loss * 0.0

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )

    def pixel_shuffle(self, x, scale_factor=0.5):
        n, w, h, c = x.size()
        # N, W, H, C --> N, W, H * scale, C // scale
        x = x.view(n, w, int(h * scale_factor), int(c / scale_factor))
        # N, W, H * scale, C // scale --> N, H * scale, W, C // scale
        x = x.permute(0, 2, 1, 3).contiguous()
        # N, H * scale, W, C // scale --> N, H * scale, W * scale, C // (scale ** 2)
        x = x.view(
            n,
            int(h * scale_factor),
            int(w * scale_factor),
            int(c / (scale_factor * scale_factor)),
        )
        if self.ps_version == 'v1':
            warnings.warn(
                "In ps_version 'v1', the height and width have not been swapped back, "
                'which results in a transposed image.'
            )
        else:
            x = x.permute(0, 2, 1, 3).contiguous()
        return x

    def extract_feature(self, pixel_values):
        if self.select_layer == -1:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=False, return_dict=True
            ).last_hidden_state
        else:
            vit_embeds = self.vision_model(
                pixel_values=pixel_values, output_hidden_states=True, return_dict=True
            ).hidden_states[self.select_layer]
        vit_embeds = vit_embeds[:, 1:, :]

        h = w = int(vit_embeds.shape[1] ** 0.5)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], h, w, -1)
        vit_embeds = self.pixel_shuffle(vit_embeds, scale_factor=self.downsample_ratio)
        vit_embeds = vit_embeds.reshape(vit_embeds.shape[0], -1, vit_embeds.shape[-1])
        vit_embeds = self.mlp1(vit_embeds)
        return vit_embeds

    def batch_chat(
        self,
        tokenizer,
        pixel_values,
        questions,
        generation_config,
        num_patches_list=None,
        history=None,
        return_history=False,
        IMG_START_TOKEN='<img>',
        IMG_END_TOKEN='</img>',
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
        verbose=False,
        image_counts=None,
    ):
        if history is not None or return_history:
            print('Now multi-turn chat is not supported in batch_chat.')
            raise NotImplementedError

        if image_counts is not None:
            num_patches_list = image_counts
            print(
                'Warning: `image_counts` is deprecated. Please use `num_patches_list` instead.'
            )

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.system_message = self.system_message
            template.append_message(template.roles[0], question)
            template.append_message(template.roles[1], None)
            query = template.get_prompt()

            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        responses = tokenizer.batch_decode(generation_output, skip_special_tokens=True)
        responses = [
            response.split(template.sep.strip())[0].strip() for response in responses
        ]
        return responses

    def chat(
        self,
        tokenizer,
        pixel_values,
        question,
        generation_config,
        history=None,
        return_history=False,
        num_patches_list=None,
        IMG_START_TOKEN='<img>',
        IMG_END_TOKEN='</img>',
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
        verbose=False,
    ):

        if history is None and pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = (
                [pixel_values.shape[0]] if pixel_values is not None else []
            )
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)
        template.system_message = self.system_message
        eos_token_id = tokenizer.convert_tokens_to_ids(template.sep.strip())

        history = [] if history is None else history
        for old_question, old_answer in history:
            template.append_message(template.roles[0], old_question)
            template.append_message(template.roles[1], old_answer)
        template.append_message(template.roles[0], question)
        template.append_message(template.roles[1], None)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        generation_config['eos_token_id'] = eos_token_id
        generation_output = self.generate(
            pixel_values=pixel_values,
            input_ids=input_ids,
            attention_mask=attention_mask,
            **generation_config,
        )
        response = tokenizer.batch_decode(generation_output, skip_special_tokens=True)[
            0
        ]
        response = response.split(template.sep.strip())[0].strip()
        history.append((question, response))
        if return_history:
            return response, history
        else:
            query_to_print = query.replace(IMG_CONTEXT_TOKEN, '')
            query_to_print = query_to_print.replace(
                f'{IMG_START_TOKEN}{IMG_END_TOKEN}', '<image>'
            )
            if verbose:
                print(query_to_print, response)
            return response

    @torch.no_grad()
    def generate(
        self,
        pixel_values: Optional[torch.FloatTensor] = None,
        input_ids: Optional[torch.FloatTensor] = None,
        attention_mask: Optional[torch.LongTensor] = None,
        visual_features: Optional[torch.FloatTensor] = None,
        generation_config: Optional[GenerationConfig] = None,
        output_hidden_states: Optional[bool] = None,
        **generate_kwargs,
    ) -> torch.LongTensor:

        assert self.img_context_token_id is not None
        if pixel_values is not None:
            if visual_features is not None:
                vit_embeds = visual_features
            else:
                vit_embeds = self.extract_feature(pixel_values)
            input_embeds = self.language_model.get_input_embeddings()(input_ids)
            B, N, C = input_embeds.shape
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = input_ids == self.img_context_token_id
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds.reshape(-1, C).to(input_embeds.device)

            input_embeds = input_embeds.reshape(B, N, C)
        else:
            input_embeds = self.language_model.get_input_embeddings()(input_ids)

        outputs = self.language_model.generate(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            generation_config=generation_config,
            output_hidden_states=output_hidden_states,
            use_cache=True,
            **generate_kwargs,
        )

        return outputs

    def prm(
        self,
        tokenizer,
        pixel_values,
        question,
        num_patches_list=None,
        IMG_START_TOKEN='<img>',
        IMG_END_TOKEN='</img>',
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
        PRM_TOKEN='<prm>',
        REWARD_TOKENS=['Yes', 'No'],
        verbose=False,
    ):
        prm_token_id = tokenizer.convert_tokens_to_ids(PRM_TOKEN)
        reward_token_ids = tokenizer.convert_tokens_to_ids(REWARD_TOKENS)
        if pixel_values is not None and '<image>' not in question:
            question = '<image>\n' + question

        if num_patches_list is None:
            num_patches_list = (
                [pixel_values.shape[0]] if pixel_values is not None else []
            )
        assert pixel_values is None or len(pixel_values) == sum(num_patches_list)

        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        template = get_conv_template(self.template)

        template.append_message(template.roles[0], '')
        template.append_message(template.roles[1], question)
        query = template.get_prompt()

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        for num_patches in num_patches_list:
            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace('<image>', image_tokens, 1)

        model_inputs = tokenizer(query, return_tensors='pt')
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        with torch.no_grad():
            logits = self.forward(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=torch.tensor([1] * pixel_values.shape[0], dtype=torch.long),
                return_dict=True,
            ).logits
            placeholder_mask = input_ids == prm_token_id
            selected_logits = logits[placeholder_mask]
            selected_logits = selected_logits[..., reward_token_ids]
            mu = torch.sigmoid(selected_logits[..., 0])
            output = mu
        return output

    def batch_prm(
        self,
        tokenizer,
        pixel_values,
        questions,
        num_patches_list=None,
        IMG_START_TOKEN='<img>',
        IMG_END_TOKEN='</img>',
        IMG_CONTEXT_TOKEN='<IMG_CONTEXT>',
        PRM_TOKEN='<prm>',
        REWARD_TOKENS=['Yes', 'No'],
        verbose=False,
    ):
        prm_token_id = tokenizer.convert_tokens_to_ids(PRM_TOKEN)
        reward_token_ids = tokenizer.convert_tokens_to_ids(REWARD_TOKENS)
        img_context_token_id = tokenizer.convert_tokens_to_ids(IMG_CONTEXT_TOKEN)
        self.img_context_token_id = img_context_token_id

        if verbose and pixel_values is not None:
            image_bs = pixel_values.shape[0]
            print(f'dynamic ViT batch size: {image_bs}')

        queries = []
        for idx, num_patches in enumerate(num_patches_list):
            question = questions[idx]
            if pixel_values is not None and '<image>' not in question:
                question = '<image>\n' + question
            template = get_conv_template(self.template)
            template.append_message(template.roles[0], '')
            template.append_message(template.roles[1], question)
            query = template.get_prompt()

            image_tokens = (
                IMG_START_TOKEN
                + IMG_CONTEXT_TOKEN * self.num_image_token * num_patches
                + IMG_END_TOKEN
            )
            query = query.replace('<image>', image_tokens, 1)
            queries.append(query)

        tokenizer.padding_side = 'left'
        model_inputs = tokenizer(queries, return_tensors='pt', padding=True)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        input_ids = model_inputs['input_ids'].to(device)
        attention_mask = model_inputs['attention_mask'].to(device)
        with torch.no_grad():
            logits = self.forward(
                pixel_values=pixel_values,
                input_ids=input_ids,
                attention_mask=attention_mask,
                image_flags=torch.tensor([1] * pixel_values.shape[0], dtype=torch.long),
                return_dict=True,
            ).logits
            placeholder_mask = input_ids == prm_token_id
            selected_logits = logits[placeholder_mask]
            selected_logits = selected_logits[..., reward_token_ids]
            mu = torch.sigmoid(selected_logits[..., 0])
            output = mu
        return output

    @property
    def lm_head(self):
        return self.language_model.get_output_embeddings()

    def get_input_embeddings(self):
        return self.language_model.get_input_embeddings()

    def get_output_embeddings(self):
        return self.language_model.get_output_embeddings()
