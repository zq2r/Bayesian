# --------------------------------------------------------
# BayesianPRM belief head
# --------------------------------------------------------
#
# This module implements a lightweight belief network for BayesianPRM.
# It is trained after the ensemble PRM is trained and frozen.
#
# Given a PRM marker hidden state h_i and frozen ensemble reward
# probabilities mu_{i,1:M}, the belief head outputs logits for
# q_phi(z_i=m | c_i), where z_i indexes the frozen reward model/head.

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn


class BayesianBeliefHead(nn.Module):
    """
    Lightweight contextual belief network over frozen ensemble PRM heads.

    Args:
        hidden_size:
            Dimension of the PRM marker hidden state.
        num_heads:
            Number of frozen ensemble reward heads/models.
        belief_hidden_dim:
            Hidden dimension of the belief MLP.
        dropout:
            Dropout rate inside the belief MLP.
        use_reward_probs:
            If True, concatenate frozen ensemble reward probabilities
            mu_{1:M} to the PRM hidden state.

    Input:
        prm_hidden_states:
            Tensor of shape [P, H], where P is the number of valid PRM markers.
        ensemble_probs:
            Optional tensor of ensemble reward probabilities.
            Accepted shapes:
              - [P, M]
              - [M, P]

    Output:
        belief_logits:
            Tensor of shape [P, M].
            Softmax over dim=-1 gives q_phi(z=m | c).
    """

    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        belief_hidden_dim: int = 256,
        dropout: float = 0.0,
        use_reward_probs: bool = True,
    ) -> None:
        super().__init__()

        if hidden_size <= 0:
            raise ValueError(f"hidden_size must be positive, got {hidden_size}")
        if num_heads <= 0:
            raise ValueError(f"num_heads must be positive, got {num_heads}")
        if belief_hidden_dim <= 0:
            raise ValueError(
                f"belief_hidden_dim must be positive, got {belief_hidden_dim}"
            )

        self.hidden_size = int(hidden_size)
        self.num_heads = int(num_heads)
        self.belief_hidden_dim = int(belief_hidden_dim)
        self.dropout = float(dropout)
        self.use_reward_probs = bool(use_reward_probs)

        input_dim = self.hidden_size + (
            self.num_heads if self.use_reward_probs else 0
        )

        self.net = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, self.belief_hidden_dim),
            nn.GELU(),
            nn.Dropout(self.dropout),
            nn.Linear(self.belief_hidden_dim, self.num_heads),
        )

    def forward(
        self,
        prm_hidden_states: torch.Tensor,
        ensemble_probs: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if prm_hidden_states.dim() != 2:
            raise ValueError(
                "prm_hidden_states must have shape [P, H], "
                f"got {tuple(prm_hidden_states.shape)}"
            )

        x = prm_hidden_states

        if self.use_reward_probs:
            if ensemble_probs is None:
                raise ValueError(
                    "ensemble_probs must be provided when use_reward_probs=True"
                )

            if ensemble_probs.dim() != 2:
                raise ValueError(
                    "ensemble_probs must have shape [P, M] or [M, P], "
                    f"got {tuple(ensemble_probs.shape)}"
                )

            # Accept both [P, M] and [M, P].
            if (
                ensemble_probs.shape[0] == self.num_heads
                and ensemble_probs.shape[1] == x.shape[0]
            ):
                ensemble_probs = ensemble_probs.transpose(0, 1)
            elif (
                ensemble_probs.shape[0] == x.shape[0]
                and ensemble_probs.shape[1] == self.num_heads
            ):
                pass
            else:
                raise ValueError(
                    "ensemble_probs shape is incompatible with prm_hidden_states "
                    f"and num_heads: ensemble_probs={tuple(ensemble_probs.shape)}, "
                    f"prm_hidden_states={tuple(x.shape)}, "
                    f"num_heads={self.num_heads}"
                )

            ensemble_probs = ensemble_probs.to(dtype=x.dtype, device=x.device)
            x = torch.cat([x, ensemble_probs], dim=-1)

        return self.net(x)