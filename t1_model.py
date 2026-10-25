"""T1 stage 1a head: score every second of a long timeline for one query.

Frozen inputs per second: SigLIP's pooled image vector and a 3x3 grid of its patch
tokens. Frozen inputs per query: SigLIP's text token states and pooled text vector.

  x_t = proj(pooled_t) + prior(zero-shot SigLIP logit_t, its z-score in the timeline)
        + query-conditioned attention-pool over the 9 grid cells of second t
  4 x [ dilated depthwise temporal conv | cross-attn to query tokens |
        self-attn over the whole timeline | FFN ]
  logit_t = linear(x_t)

The zero-shot SigLIP logit is an explicit input, so the head starts from what B1
already had; everything it scores beyond that is learned. There is no positional
encoding: the dilated convs give local order (+-60 s receptive field), attention is
order-free, and nothing ties the score to absolute position in the timeline.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

DIM_IN = 1152


class Block(nn.Module):
    def __init__(self, d: int, heads: int, dilation: int, dropout: float):
        super().__init__()
        self.heads = heads
        self.n1, self.conv = nn.LayerNorm(d), nn.Conv1d(d, d, 9, padding=4 * dilation,
                                                        dilation=dilation, groups=d)
        self.n2, self.xq, self.xkv, self.xo = (nn.LayerNorm(d), nn.Linear(d, d),
                                               nn.Linear(d, 2 * d), nn.Linear(d, d))
        self.n3, self.qkv, self.so = nn.LayerNorm(d), nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.n4 = nn.LayerNorm(d)
        self.ffn = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Dropout(dropout),
                                 nn.Linear(4 * d, d))
        self.drop = nn.Dropout(dropout)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        return x.view(b, n, self.heads, d // self.heads).transpose(1, 2)

    def forward(self, x: torch.Tensor, q: torch.Tensor, qmask: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        x = x + self.drop(self.conv(self.n1(x).transpose(1, 2)).transpose(1, 2))
        h = self.n2(x)
        k, v = self.xkv(q).chunk(2, dim=-1)
        att = F.scaled_dot_product_attention(
            self._split(self.xq(h)), self._split(k), self._split(v),
            attn_mask=qmask[:, None, None, :])
        x = x + self.drop(self.xo(att.transpose(1, 2).reshape(b, n, d)))
        qq, kk, vv = self.qkv(self.n3(x)).chunk(3, dim=-1)
        att = F.scaled_dot_product_attention(self._split(qq), self._split(kk), self._split(vv))
        x = x + self.drop(self.so(att.transpose(1, 2).reshape(b, n, d)))
        return x + self.drop(self.ffn(self.n4(x)))


class T1Head(nn.Module):
    def __init__(self, d: int = 512, layers: int = 4, heads: int = 8,
                 dropout: float = 0.1, use_grid: bool = True):
        super().__init__()
        self.use_grid = use_grid
        self.f_in = nn.Sequential(nn.LayerNorm(DIM_IN), nn.Linear(DIM_IN, d))
        self.g_in = nn.Sequential(nn.LayerNorm(DIM_IN), nn.Linear(DIM_IN, d))
        self.q_in = nn.Sequential(nn.LayerNorm(DIM_IN), nn.Linear(DIM_IN, d))
        self.qp_in = nn.Sequential(nn.LayerNorm(DIM_IN), nn.Linear(DIM_IN, d))
        self.prior_in = nn.Linear(2, d)
        self.blocks = nn.ModuleList(Block(d, heads, 2 ** i, dropout) for i in range(layers))
        self.out = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))

    def forward(self, feats: torch.Tensor, qtok: torch.Tensor, qmask: torch.Tensor,
                prior: torch.Tensor) -> torch.Tensor:
        """feats [B, L, 10, 1152] (pooled, grid x9); qtok [B, 64, 1152]; qmask [B, 64]
        True = real token; prior [B, L] zero-shot SigLIP logits.  ->  logits [B, L]."""
        z = (prior - prior.mean(-1, keepdim=True)) / (prior.std(-1, keepdim=True) + 1e-4)
        x = self.f_in(feats[:, :, 0]) + self.prior_in(torch.stack([prior / 10.0, z], -1))
        q = self.q_in(qtok)
        if self.use_grid:
            qv = (q * qmask[..., None]).sum(1) / qmask.sum(1, keepdim=True)    # [B, d]
            g = self.g_in(feats[:, :, 1:])                                    # [B, L, 9, d]
            w = torch.einsum("blkd,bd->blk", g, qv) / g.shape[-1] ** 0.5
            x = x + (w.softmax(-1)[..., None] * g).sum(2)
        for blk in self.blocks:
            x = blk(x, q, qmask)
        return self.out(x).squeeze(-1)


def zero_shot_logits(pooled: torch.Tensor, qpool: torch.Tensor, scale: float,
                     bias: float) -> torch.Tensor:
    """SigLIP's own image-text logit per second: [B, L, 1152] x [B, 1152] -> [B, L]."""
    return (F.normalize(pooled.float(), dim=-1)
            @ F.normalize(qpool.float(), dim=-1)[..., None]).squeeze(-1) * scale + bias
