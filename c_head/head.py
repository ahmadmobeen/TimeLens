"""Boundary-regression head + L1/gIoU loss for NS-P1 method bet C.

The *unconditioned* AdaVTG-LLM-style baseline (C step 1): a plain MLP that reads a
frozen TimeLens-7B last-layer hidden state ``h`` at the generation position and
regresses a moment ``(center, width)``, both normalized to clip duration. No
length-conditioning and no cross-attention to visual tokens — the cleanest test of
whether a frozen head can close the residual sub-second center floor
(``runbook-C-boundary-head.md`` kill-switch; ``C-baseline-adavtg-spec.md`` §6.6a).
Length-conditioning is added later as the C arm.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn


class BoundaryHead(nn.Module):
    """``LayerNorm -> MLP(d -> d -> 2) -> sigmoid`` => (center, width) in (0,1).

    The input ``LayerNorm`` is essential: TimeLens-7B last-layer states carry massive
    outlier activations (the LLM "massive-activation" phenomenon), so regressing from the
    raw state blows up the logits, saturates the output sigmoid, and freezes training
    (observed in the 3-clip smoke). The final layer is zero-initialized so the head starts
    at ``sigmoid(0)=0.5`` for both center and width — a neutral, non-saturated start.
    """

    def __init__(
        self, hidden_size: int = 3584, mlp_dim: int | None = None, dropout: float = 0.0
    ) -> None:
        super().__init__()
        mlp_dim = mlp_dim or hidden_size
        self.norm = nn.LayerNorm(hidden_size)
        self.fc1 = nn.Linear(hidden_size, mlp_dim)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(mlp_dim, 2)
        nn.init.zeros_(self.fc2.weight)  # start at sigmoid(0)=0.5, stable gradients
        nn.init.zeros_(self.fc2.bias)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        """``h``: ``[B, hidden_size]`` -> ``[B, 2]`` = (center, width), each in (0, 1)."""
        x = self.drop(self.act(self.fc1(self.norm(h))))
        return torch.sigmoid(self.fc2(x))


def cw_to_startend(cw: torch.Tensor) -> torch.Tensor:
    """(center, width) -> (start, end), grad-preserving. ``cw``: [B,2] -> [B,2]."""
    center, width = cw[:, 0], cw[:, 1]
    return torch.stack([center - width / 2.0, center + width / 2.0], dim=-1)


def giou_1d(
    pred_se: torch.Tensor, gt_se: torch.Tensor, eps: float = 1e-7
) -> Tuple[torch.Tensor, torch.Tensor]:
    """1-D generalized IoU on (start, end) intervals. Returns ``(iou, giou)``, each [B]."""
    ps, pe = pred_se[:, 0], pred_se[:, 1]
    gs, ge = gt_se[:, 0], gt_se[:, 1]
    inter = (torch.min(pe, ge) - torch.max(ps, gs)).clamp(min=0)
    len_p = (pe - ps).clamp(min=0)
    len_g = (ge - gs).clamp(min=0)
    union = len_p + len_g - inter
    iou = inter / union.clamp(min=eps)
    enclose = (torch.max(pe, ge) - torch.min(ps, gs)).clamp(min=eps)
    giou = iou - (enclose - union) / enclose
    return iou, giou


def boundary_loss(
    pred_cw: torch.Tensor,
    gt_cw: torch.Tensor,
    lambda_l1: float = 1.0,
    lambda_giou: float = 1.0,
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """``L1(center,width) + gIoU`` loss on duration-normalized (c,w). Returns (loss, parts).

    L1 is the summed abs error ``|Δc| + |Δw|`` (AdaVTG Eq. 6); gIoU is on the (start,end)
    interval derived from (c,w). Default weights 1/1; ablate L1-only (TimeRefine) later.
    """
    l1 = (pred_cw - gt_cw).abs().sum(dim=-1).mean()  # |Δc| + |Δw|
    _, giou = giou_1d(cw_to_startend(pred_cw), cw_to_startend(gt_cw))
    giou_loss = (1.0 - giou).mean()
    loss = lambda_l1 * l1 + lambda_giou * giou_loss
    return loss, {"l1": float(l1.detach()), "giou_loss": float(giou_loss.detach())}


class CrossAttnBoundaryHead(nn.Module):
    """AdaVTG-style head (arm 2): a post-answer query token cross-attends over the clip's
    pooled visual tokens, then regresses (center, width).

    query ``q`` [B,H] (post-answer state) attends over K/V = pooled visual tokens ``kv``
    [B,P,H] across ``n_layers`` cross-attention blocks (residual + norm), then MLP→2→sigmoid.
    Input LayerNorms tame the LLM-state outliers (same rationale as BoundaryHead); the final
    layer is zero-initialized so the head starts at sigmoid(0)=0.5.
    """

    def __init__(
        self, hidden_size: int = 3584, n_heads: int = 8, n_layers: int = 2, dropout: float = 0.0
    ) -> None:
        super().__init__()
        self.q_norm = nn.LayerNorm(hidden_size)
        self.kv_norm = nn.LayerNorm(hidden_size)
        self.attn = nn.ModuleList(
            [nn.MultiheadAttention(hidden_size, n_heads, dropout=dropout, batch_first=True)
             for _ in range(n_layers)]
        )
        self.post = nn.ModuleList([nn.LayerNorm(hidden_size) for _ in range(n_layers)])
        self.mlp = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_size, 2),
        )
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, q: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        """q: [B,H] query; kv: [B,P,H] visual tokens -> [B,2] = (center,width) in (0,1)."""
        x = self.q_norm(q).unsqueeze(1)  # [B,1,H]
        kv = self.kv_norm(kv)            # [B,P,H]
        for attn, norm in zip(self.attn, self.post):
            a, _ = attn(x, kv, kv, need_weights=False)
            x = norm(x + a)
        return torch.sigmoid(self.mlp(x.squeeze(1)))
