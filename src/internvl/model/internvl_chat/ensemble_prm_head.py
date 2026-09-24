# Copyright (c) OpenMMLab.
# A lightweight ensemble scalar reward head for PRM training.

import math
from typing import Iterator, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


class EnsembleScalarRewardHead(nn.Module):
    """
    Ensemble scalar reward head for process reward modeling.

    A shared language-model backbone provides hidden states, and multiple
    head-specific MLPs predict scalar correctness logits.

    When ``use_prior_network=True``, the final prediction of each ensemble
    member is the sum of:

        learned_logit + prior_scale * frozen_prior_logit

    The prior network is randomly initialized and permanently frozen. Its
    purpose is to induce different residual learning problems for different
    ensemble members, thereby discouraging ensemble collapse.

    Input:
        hidden_states:
            Shape [M, H] or [B, L, H], where H is ``hidden_size``.

    Output:
        logits:
            Shape [E, M] or [E, B, L], where E is ``num_heads``.
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        hidden_dim: int = 128,
        dropout: float = 0.0,
        activation: str = "gelu",
        use_prior_network: bool = False,
        prior_scale: float = 1.0,
    ):
        super().__init__()

        if num_heads <= 0:
            raise ValueError(
                f"num_heads must be positive, got {num_heads}"
            )
        if hidden_size <= 0:
            raise ValueError(
                f"hidden_size must be positive, got {hidden_size}"
            )
        if hidden_dim <= 0:
            raise ValueError(
                f"hidden_dim must be positive, got {hidden_dim}"
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                f"dropout must be in [0, 1), got {dropout}"
            )
        if not math.isfinite(prior_scale):
            raise ValueError(
                f"prior_scale must be finite, got {prior_scale}"
            )
        if prior_scale < 0.0:
            raise ValueError(
                f"prior_scale must be non-negative, got {prior_scale}"
            )

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.hidden_dim = int(hidden_dim)
        self.dropout = float(dropout)
        self.activation = str(activation)

        self.use_prior_network = bool(use_prior_network)
        self.prior_scale = float(prior_scale)

        # ================================================================
        # Trainable ensemble branch
        #
        # Keep the original parameter names unchanged so that existing
        # ensemble checkpoints remain compatible:
        #   norm, w1, b1, w2, b2
        # ================================================================
        self.norm = nn.LayerNorm(self.hidden_size)

        # Each ensemble member owns one [hidden_size, hidden_dim] matrix.
        self.w1 = nn.Parameter(
            torch.empty(
                self.num_heads,
                self.hidden_size,
                self.hidden_dim,
            )
        )
        self.b1 = nn.Parameter(
            torch.zeros(
                self.num_heads,
                self.hidden_dim,
            )
        )

        # Each ensemble member owns one [hidden_dim, 1] matrix.
        self.w2 = nn.Parameter(
            torch.empty(
                self.num_heads,
                self.hidden_dim,
                1,
            )
        )
        self.b2 = nn.Parameter(
            torch.zeros(
                self.num_heads,
                1,
            )
        )

        # ================================================================
        # Frozen randomized prior branch
        #
        # It has the same computation structure as the learned branch:
        # LayerNorm -> head-specific MLP -> scalar logit.
        #
        # These parameters are stored in the state_dict but must never be
        # updated by the optimizer.
        # ================================================================
        if self.use_prior_network:
            self.prior_norm = nn.LayerNorm(self.hidden_size)

            self.prior_w1 = nn.Parameter(
                torch.empty(
                    self.num_heads,
                    self.hidden_size,
                    self.hidden_dim,
                )
            )
            self.prior_b1 = nn.Parameter(
                torch.zeros(
                    self.num_heads,
                    self.hidden_dim,
                )
            )

            self.prior_w2 = nn.Parameter(
                torch.empty(
                    self.num_heads,
                    self.hidden_dim,
                    1,
                )
            )
            self.prior_b2 = nn.Parameter(
                torch.zeros(
                    self.num_heads,
                    1,
                )
            )
        else:
            # Register None parameters so the module interface remains clear,
            # while no additional state_dict keys are introduced when the
            # prior network is disabled.
            self.prior_norm = None
            self.register_parameter("prior_w1", None)
            self.register_parameter("prior_b1", None)
            self.register_parameter("prior_w2", None)
            self.register_parameter("prior_b2", None)

        self.reset_parameters()

    @staticmethod
    def _reset_mlp_parameters(
        w1: nn.Parameter,
        b1: nn.Parameter,
        w2: nn.Parameter,
        b2: nn.Parameter,
    ) -> None:
        """
        Initialize every ensemble member independently.

        Calling Xavier initialization separately for each head consumes a
        different part of the RNG stream, so the heads receive independent
        random initializations under a fixed global seed.
        """
        for head_idx in range(w1.shape[0]):
            nn.init.xavier_uniform_(w1[head_idx])
            nn.init.xavier_uniform_(w2[head_idx])

        nn.init.zeros_(b1)
        nn.init.zeros_(b2)

    def reset_parameters(self) -> None:
        """
        Initialize both the learned branch and the randomized prior branch.
        """
        self.norm.reset_parameters()

        self._reset_mlp_parameters(
            self.w1,
            self.b1,
            self.w2,
            self.b2,
        )

        if self.use_prior_network:
            self.prior_norm.reset_parameters()

            self._reset_mlp_parameters(
                self.prior_w1,
                self.prior_b1,
                self.prior_w2,
                self.prior_b2,
            )

            # reset_parameters() may be called after construction, so always
            # restore the frozen status of the prior branch afterward.
            self.freeze_prior_network()

    def learned_parameters(self) -> Iterator[nn.Parameter]:
        """
        Yield only the trainable ensemble-branch parameters.

        This helper will be used later in the training script, instead of
        indiscriminately enabling gradients for every parameter in the
        complete ensemble head.
        """
        yield from self.norm.parameters()
        yield self.w1
        yield self.b1
        yield self.w2
        yield self.b2

    def prior_parameters(self) -> Iterator[nn.Parameter]:
        """
        Yield only the randomized-prior parameters.
        """
        if not self.use_prior_network:
            return

        yield from self.prior_norm.parameters()
        yield self.prior_w1
        yield self.prior_b1
        yield self.prior_w2
        yield self.prior_b2

    def freeze_prior_network(self) -> None:
        """
        Permanently disable parameter gradients for the randomized prior.

        Note:
            This freezes only the prior parameters. It deliberately does not
            detach hidden_states, so gradients may still flow from the final
            loss through the fixed prior computation into a trainable shared
            backbone.
        """
        if not self.use_prior_network:
            return

        for param in self.prior_parameters():
            param.requires_grad_(False)

    def prior_is_frozen(self) -> bool:
        """
        Return True when every prior parameter is frozen.
        """
        if not self.use_prior_network:
            return True

        return all(
            not param.requires_grad
            for param in self.prior_parameters()
        )

    def _activate(self, x: torch.Tensor) -> torch.Tensor:
        if self.activation == "gelu":
            return F.gelu(x)
        if self.activation == "relu":
            return F.relu(x)
        if self.activation == "silu":
            return F.silu(x)

        raise ValueError(
            f"Unsupported activation: {self.activation}"
        )

    def _forward_branch(
        self,
        hidden_states: torch.Tensor,
        norm: nn.LayerNorm,
        w1: torch.Tensor,
        b1: torch.Tensor,
        w2: torch.Tensor,
        b2: torch.Tensor,
        dropout: float,
    ) -> torch.Tensor:
        """
        Run one parallel ensemble MLP branch.

        Args:
            hidden_states:
                [M, H] or [B, L, H].
            norm, w1, b1, w2, b2:
                Parameters of either the learned branch or prior branch.
            dropout:
                Dropout probability for this branch.

        Returns:
            [E, M] or [E, B, L].
        """
        if hidden_states.size(-1) != self.hidden_size:
            raise ValueError(
                f"Expected hidden size {self.hidden_size}, "
                f"but got {hidden_states.size(-1)}"
            )

        original_shape = hidden_states.shape[:-1]

        # [M, H] or [B, L, H] -> [M_total, H]
        x = norm(hidden_states)
        x = x.reshape(-1, self.hidden_size)

        # Match branch parameter dtype. This matters for bf16/fp16 training.
        x = x.to(dtype=w1.dtype)

        # [M_total, H] x [E, H, D] -> [E, M_total, D]
        x = torch.einsum(
            "mh,ehd->emd",
            x,
            w1,
        )
        x = x + b1[:, None, :]

        x = self._activate(x)

        if dropout > 0.0:
            x = F.dropout(
                x,
                p=dropout,
                training=self.training,
            )

        # [E, M_total, D] x [E, D, 1] -> [E, M_total, 1]
        logits = torch.einsum(
            "emd,edo->emo",
            x,
            w2,
        )
        logits = logits + b2[:, None, :]
        logits = logits.squeeze(-1)

        # [E, M_total] -> [E, *original_shape]
        logits = logits.reshape(
            self.num_heads,
            *original_shape,
        )

        return logits

    def forward_components(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        Optional[torch.Tensor],
        torch.Tensor,
    ]:
        """
        Return learned, prior, and final logits separately.

        Returns:
            learned_logits:
                [E, ...]
            prior_logits:
                [E, ...], or None when prior network is disabled.
                This is the unscaled prior output.
            final_logits:
                learned_logits + prior_scale * prior_logits
        """
        learned_logits = self._forward_branch(
            hidden_states=hidden_states,
            norm=self.norm,
            w1=self.w1,
            b1=self.b1,
            w2=self.w2,
            b2=self.b2,
            dropout=self.dropout,
        )

        if not self.use_prior_network:
            return learned_logits, None, learned_logits

        # Do not wrap this in torch.no_grad(), and do not detach hidden_states.
        # The prior parameters remain frozen, but its computation may still
        # contribute gradients to a trainable shared backbone.
        prior_logits = self._forward_branch(
            hidden_states=hidden_states,
            norm=self.prior_norm,
            w1=self.prior_w1,
            b1=self.prior_b1,
            w2=self.prior_w2,
            b2=self.prior_b2,
            dropout=0.0,
        )

        final_logits = (
            learned_logits
            + self.prior_scale * prior_logits
        )

        return learned_logits, prior_logits, final_logits

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        """
        Return the final logits used by the PRM loss and inference code.
        """
        _, _, final_logits = self.forward_components(hidden_states)
        return final_logits

    @staticmethod
    def probs_mean_std(
        logits: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convert ensemble logits to reward mean and ensemble uncertainty.

        Args:
            logits:
                [E, ...]

        Returns:
            mean:
                [...]
            std:
                [...]
        """
        probs = torch.sigmoid(logits)
        mean = probs.mean(dim=0)
        std = probs.std(dim=0, unbiased=False)
        return mean, std