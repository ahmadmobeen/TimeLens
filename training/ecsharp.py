"""NS-P1 loss-1 Stage 2 = EC-Sharp.

A query-conditioned per-frame soft-argmax expected-center head, trained JOINTLY with the LoRA SFT
so gradients flow into the (LoRA-adapted) backbone. The *frozen* version of this exact head was RED
(|dc|~2.3s pooled / per-frame swept layers 14/21/28 all failed to beat the 1.05s text-decode floor —
see ``probe_perframe.PFHead``). EC-Sharp's bet: UNFREEZING the backbone (via LoRA) lets the
representation move so the differentiable soft-argmax center becomes sub-second. If it works, the
<2s floor was an objective/readout limitation, not a fundamental representational one.

Deploy = the head's soft-argmax center (see ``eval_ecsharp.py``), NOT the text decode. The standard
span-format CE is kept as a co-training signal that keeps the backbone's grounding intact while the
auxiliary center/sharpness/width losses reshape the per-frame representation.

Cite DSNT (arXiv 1801.07372): differentiable soft-argmax of a (here temporal) heatmap.

Everything is gated by env ``ECSHARP=1``; off => the plain SFT path is untouched.

Loss (all in SECONDS for a natural, duration-uniform scale):
    L = CE_text
      + w_center * Huber(center_pred_sec, center_gt_sec ; delta=ECS_HUBER_CENTER)
      + w_width  * Huber(log width_pred_sec, log width_gt_sec ; delta=0.5)          # log-ratio
      + w_sharp  * Huber(std_pred_sec, target_sec ; delta=0.5)                       # anti-collapse
where target_sec is the GT half-width (width_prop, default) or a fixed sub-second sigma.
"""
from __future__ import annotations

import os
from typing import Optional, Tuple

import torch
import torch.nn as nn

from training.trainer import QwenSFTTrainer


def _flag(name: str, default: str = "0") -> bool:
    return os.environ.get(name, default).lower() not in ("0", "", "false", "no")


class ECSharpConfig:
    """All knobs read from env so the recipe stays in the launch command (ablatable)."""

    def __init__(self) -> None:
        self.enabled = _flag("ECSHARP")
        self.w_center = float(os.environ.get("ECS_W_CENTER", "1.0"))
        self.w_width = float(os.environ.get("ECS_W_WIDTH", "0.5"))
        self.w_sharp = float(os.environ.get("ECS_W_SHARP", "0.5"))
        self.sigma_mode = os.environ.get("ECS_SIGMA_MODE", "width_prop")  # width_prop | fixed
        self.sigma_fixed = float(os.environ.get("ECS_SIGMA_FIXED", "0.5"))  # seconds
        self.huber_center = float(os.environ.get("ECS_HUBER_CENTER", "1.0"))  # seconds
        self.layer = int(os.environ.get("ECS_LAYER", "-1"))  # hidden_states index; -1 = last

    def __repr__(self) -> str:
        return (
            "ECSharpConfig("
            f"enabled={self.enabled}, w_center={self.w_center}, w_width={self.w_width}, "
            f"w_sharp={self.w_sharp}, sigma_mode={self.sigma_mode}, sigma_fixed={self.sigma_fixed}, "
            f"huber_center={self.huber_center}, layer={self.layer})"
        )


class ECSharpHead(nn.Module):
    """Query(h_q)-conditioned soft-argmax over per-frame states -> (center, width, var), all in
    normalized [0,1] clip-fraction units. Mirrors ``probe_perframe.PFHead`` so a positive result
    here (vs that frozen head's RED) isolates the effect of joint (LoRA) backbone training.

    Reductions (softmax / expected value / variance) run in fp32 for stability regardless of the
    module's parameter dtype (bf16 under deepspeed); the qkv projections run in the param dtype.
    """

    def __init__(self, d: int = 3584) -> None:
        super().__init__()
        self.vn = nn.LayerNorm(d)
        self.qn = nn.LayerNorm(d)
        self.qp = nn.Linear(d, d)
        self.kp = nn.Linear(d, d)
        self.wn = nn.LayerNorm(2 * d)
        self.wf = nn.Linear(2 * d, 1)
        self.scale = d ** -0.5

    def forward(
        self, V: torch.Tensor, mask: torch.Tensor, pos: torch.Tensor, hq: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # V [B,T,d]; mask [B,T]; pos [B,T] in [0,1]; hq [B,d]
        dt = self.qp.weight.dtype
        V = V.to(dt)
        hq = hq.to(dt)
        q = self.qp(self.qn(hq))                                  # [B,d]
        k = self.kp(self.vn(V))                                   # [B,T,d]
        logit = torch.einsum("btd,bd->bt", k, q).float() * self.scale  # [B,T] fp32
        logit = logit.masked_fill(mask < 0.5, float("-inf"))
        p = torch.softmax(logit, dim=-1)                          # [B,T] fp32
        posf = pos.float()
        center = (p * posf).sum(-1)                               # [B] fp32, normalized
        var = (p * (posf - center[:, None]) ** 2).sum(-1)         # [B] fp32
        m = mask.to(dt)
        vmean = (V * m[..., None]).sum(1) / m.sum(1, keepdim=True).clamp(min=1)  # [B,d]
        w = torch.sigmoid(self.wf(self.wn(torch.cat([hq, vmean], dim=-1))).float()).squeeze(-1)
        return center, w, var


def vis_token_ids(model_config) -> list[int]:
    ids = [
        t
        for t in (
            getattr(model_config, "image_token_id", None),
            getattr(model_config, "video_token_id", None),
        )
        if t is not None
    ]
    return ids or [151655, 151656]  # Qwen2.5-VL <|image_pad|>/<|video_pad|> fallback


def build_perframe_query(
    hidden: torch.Tensor,
    input_ids: torch.Tensor,
    grid_thw: torch.Tensor,
    vis_ids: torch.Tensor,
    labels: Optional[torch.Tensor] = None,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Extract, WITH grad, per-frame visual states + the query vector from a live forward pass.

    hidden [B,S,d] (a chosen layer's states); input_ids [B,S]; grid_thw [T,3] (TimeLens feeds
    frames-as-images => one grid row per frame => T = grid_thw.shape[0]); vis_ids the video/image
    pad token ids. labels [B,S] (train) locates the answer boundary so hq = the LAST PROMPT token
    (the position whose output predicts the first answer token) — identical to eval, where the
    prompt-only input's last token is that same position (so no train/eval query mismatch).

    Returns (V[1,T,d], mask[1,T], pos[1,T], hq[1,d]) or None if the token/frame layout is degenerate.
    Assumes B==1 (the SFT recipe's per_device_train_batch_size)."""
    B, S, d = hidden.shape
    assert B == 1, f"EC-Sharp expects per_device_train_batch_size=1, got B={B}"
    ids = input_ids[0]
    vis_pos = torch.isin(ids, vis_ids).nonzero(as_tuple=True)[0]
    n_vis = int(vis_pos.numel())
    T = int(grid_thw.shape[0])
    if n_vis == 0 or T == 0 or n_vis % T != 0:
        return None
    spatial = n_vis // T
    vtok = hidden[0].index_select(0, vis_pos)                 # [n_vis,d] grad-carrying
    V = vtok.view(T, spatial, d).mean(1).unsqueeze(0)         # [1,T,d]
    mask = torch.ones(1, T, device=hidden.device, dtype=hidden.dtype)
    pos = ((torch.arange(T, device=hidden.device).float() + 0.5) / T).unsqueeze(0)  # [1,T]
    if labels is not None:
        ans = (labels[0] != -100).nonzero(as_tuple=True)[0]
        hq_pos = (int(ans[0].item()) - 1) if ans.numel() else (S - 1)
    else:
        hq_pos = S - 1
    hq_pos = max(0, min(hq_pos, S - 1))
    hq = hidden[0, hq_pos].unsqueeze(0)                       # [1,d]
    return V, mask, pos, hq


def _huber(pred: torch.Tensor, target: torch.Tensor, delta: float) -> torch.Tensor:
    e = (pred - target).abs()
    return torch.where(e < delta, 0.5 * e * e / delta, e - 0.5 * delta).mean()


def ecsharp_loss(
    center: torch.Tensor, w: torch.Tensor, var: torch.Tensor,
    ecs_target: torch.Tensor, cfg: ECSharpConfig,
) -> Tuple[torch.Tensor, dict]:
    """ecs_target [B,3] = (center_sec, width_sec, duration_sec). center/w/var normalized [0,1]."""
    c_sec = ecs_target[:, 0].float()
    w_sec = ecs_target[:, 1].float().clamp(min=1e-3)
    dur = ecs_target[:, 2].float().clamp(min=1e-3)

    center_sec = center * dur
    width_sec = (w * dur).clamp(min=1e-3)
    std_sec = var.clamp(min=0).sqrt() * dur

    center_loss = _huber(center_sec, c_sec, cfg.huber_center)
    width_loss = _huber(torch.log(width_sec), torch.log(w_sec), 0.5)
    if cfg.sigma_mode == "fixed":
        target_sec = torch.full_like(std_sec, cfg.sigma_fixed)
    else:  # width_prop: distribution spread should match the moment's half-width
        target_sec = (w_sec / 2.0).clamp(min=0.1)
    sharp_loss = _huber(std_sec, target_sec, 0.5)

    total = cfg.w_center * center_loss + cfg.w_width * width_loss + cfg.w_sharp * sharp_loss
    parts = {
        "ec_center": float(center_loss.detach()),
        "ec_width": float(width_loss.detach()),
        "ec_sharp": float(sharp_loss.detach()),
        "ec_dc_sec": float((center_sec - c_sec).abs().mean().detach()),  # raw |Δcenter| seconds
        "ec_std_sec": float(std_sec.mean().detach()),
    }
    return total, parts


class ECSharpTrainer(QwenSFTTrainer):
    """QwenSFTTrainer + a jointly-trained EC-Sharp auxiliary loss. The head is a submodule of the
    model (attached in the launch script) so create_optimizer (inherited) and deepspeed manage its
    params; we keep a direct reference for calling it in compute_loss."""

    def __init__(self, *args, ec_head=None, ec_cfg=None, vis_ids=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.ec_head = ec_head
        self.ec_cfg = ec_cfg or ECSharpConfig()
        self.ec_vis_ids = vis_ids  # tensor
        self._ec_log: dict = {}

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        ecs_target = inputs.pop("ecs_target", None)
        outputs = model(**inputs, output_hidden_states=True)
        ce = outputs.loss
        loss = ce
        if ecs_target is not None and self.ec_head is not None:
            hidden = outputs.hidden_states[self.ec_cfg.layer]  # [B,S,d]
            grid = inputs.get("video_grid_thw", inputs.get("image_grid_thw"))
            vis_ids = self.ec_vis_ids.to(hidden.device)
            built = build_perframe_query(
                hidden, inputs["input_ids"], grid, vis_ids, labels=inputs.get("labels")
            )
            if built is not None:
                V, mask, pos, hq = built
                center, w, var = self.ec_head(V, mask, pos, hq)
                aux, parts = ecsharp_loss(center, w, var, ecs_target.to(center.device), self.ec_cfg)
                loss = ce + aux
                self._ec_log = {"ce": float(ce.detach()), "ec_aux": float(aux.detach()), **parts}
        return (loss, outputs) if return_outputs else loss

    def log(self, logs, start_time=None):
        if self._ec_log:
            logs = {**logs, **self._ec_log}
        try:
            super().log(logs, start_time)
        except TypeError:  # older signature
            super().log(logs)
