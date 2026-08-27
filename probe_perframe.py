"""NS-P1 round-1 #A (TRUE per-frame gate): query-conditioned soft-argmax over PER-FRAME features.

Reads the un-pooled per-frame caches from cache_perframe.py
({per_clip:[{layers:{L:[T,d]}, h_ans, T}], meta:[{video_id,query,gt_start,gt_end,duration}]}) and runs
the SAME query(h_ans)-conditioned soft-argmax as probe_softargmax.py, but over the T real frames
(variable length, padded + masked), swept across layers. This resolves the pooling confound of the
pooled-256 probe (which was RED at |dc|~2.3s).

GATE: if some layer's per-frame soft-argmax beats the model's own text decode (~1.05s |dc|) on the
<2s bin => sub-second center IS extractable from the frozen per-frame representation (readout/loss
branch is alive). If no layer beats it => localizes the floor to the backbone (finetune branch) and
supports the input-Nyquist hypothesis.

Run from the repo ROOT (main venv has torch + benchmarks + needlescan):
  env -u VIRTUAL_ENV CUDA_VISIBLE_DEVICES=0 uv run python research/TimeLens/probe_perframe.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))

from c_head import config as C  # noqa: E402
from benchmarks.datasets.public.charades_timelens import CharadesTimeLensBenchmark  # noqa: E402
from benchmarks.length_profile import signed_length_profile  # noqa: E402
from benchmarks.length_strata import LengthStratifiedGrounding  # noqa: E402
from needlescan.base.types import Prediction, Sample  # noqa: E402

LAYERS = (14, 21, 28)


def read_clips(cache_dir, tags):
    """Read the .pt caches ONCE; return (clips, metas, cw[N,3=(c_frac,w_frac,dur)]) after dropping degenerate GT."""
    clips, metas = [], []
    for tag in tags:
        for f in sorted(Path(cache_dir).glob(f"{tag}.shard*of*.pt")):
            d = torch.load(f, map_location="cpu")
            clips += d["per_clip"]
            metas += d["meta"]
    kc, km, cw = [], [], []
    for c, m in zip(clips, metas):
        dur, s, e = float(m["duration"]), float(m["gt_start"]), float(m["gt_end"])
        if dur <= 0 or e <= s:
            continue
        kc.append(c)
        km.append(m)
        cw.append([min(max(((s + e) / 2.0) / dur, 0.0), 1.0), min(max((e - s) / dur, 1e-4), 1.0), dur])
    return kc, km, torch.tensor(cw, dtype=torch.float32)


def pad_layer(clips, layer):
    """Pad variable-T per-frame features for one layer -> V[N,Tmax,d], mask, pos (frame time-fraction), hans."""
    Tmax = max(c["T"] for c in clips)
    N = len(clips)
    d = clips[0]["layers"][layer].shape[1]
    V = torch.zeros(N, Tmax, d)
    mask = torch.zeros(N, Tmax)
    pos = torch.zeros(N, Tmax)
    hans = torch.zeros(N, d)
    for i, c in enumerate(clips):
        T = c["T"]
        V[i, :T] = c["layers"][layer].float()
        mask[i, :T] = 1.0
        pos[i, :T] = (torch.arange(T).float() + 0.5) / T
        hans[i] = c["h_ans"].float()
    return V, mask, pos, hans


class PFHead(torch.nn.Module):
    def __init__(self, d: int = 3584) -> None:
        super().__init__()
        self.vn = torch.nn.LayerNorm(d)
        self.qn = torch.nn.LayerNorm(d)
        self.qp = torch.nn.Linear(d, d)
        self.kp = torch.nn.Linear(d, d)
        self.wn = torch.nn.LayerNorm(2 * d)
        self.wf = torch.nn.Linear(2 * d, 1)
        self.scale = d ** -0.5

    def forward(self, V, mask, pos, hans):
        q = self.qp(self.qn(hans))                              # [B,d]
        k = self.kp(self.vn(V))                                 # [B,T,d]
        logit = torch.einsum("btd,bd->bt", k, q) * self.scale   # [B,T]
        logit = logit.masked_fill(mask < 0.5, float("-inf"))
        p = torch.softmax(logit, dim=-1)                        # [B,T] over real frames
        center = (p * pos).sum(-1)
        var = (p * (pos - center[:, None]) ** 2).sum(-1)
        vmean = (V * mask[..., None]).sum(1) / mask.sum(1, keepdim=True).clamp(min=1)
        w = torch.sigmoid(self.wf(self.wn(torch.cat([hans, vmean], dim=-1)))).squeeze(-1)
        return center, w, var


def huber(a, b, delta=0.05):
    e = (a - b).abs()
    return torch.where(e < delta, 0.5 * e * e / delta, e - 0.5 * delta).mean()


def train_head(V, mask, pos, hans, cw, arm, device, epochs=80, bs=128, lr=5e-4, lam=1.0, sigma_s=0.3, seed=0):
    torch.manual_seed(seed)
    head = PFHead(V.shape[-1]).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    N = cw.shape[0]
    tr = torch.randperm(N, generator=torch.Generator().manual_seed(seed))[max(1, N // 20):]
    loader = DataLoader(TensorDataset(V[tr], mask[tr], pos[tr], hans[tr], cw[tr]), batch_size=bs, shuffle=True)
    for _ in range(epochs):
        for Vb, mb, pb, hb, cwb in loader:
            Vb, mb, pb, hb, cwb = Vb.to(device), mb.to(device), pb.to(device), hb.to(device), cwb.to(device)
            center, w, var = head(Vb, mb, pb, hb)
            loss = huber(center, cwb[:, 0]) + huber(w, cwb[:, 1])
            if arm == "unimodal":
                tv = (sigma_s / cwb[:, 2]).clamp(max=0.5) ** 2
                loss = loss + lam * (var - tv).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head


@torch.no_grad()
def score_head(head, V, mask, pos, hans, metas, device):
    head.eval()
    center, w, var = head(V.to(device), mask.to(device), pos.to(device), hans.to(device))
    center, w, var = center.cpu(), w.cpu(), var.cpu()
    samples, preds = [], []
    for i, m in enumerate(metas):
        dur = float(m["duration"])
        c, wid = float(center[i]) * dur, float(w[i]) * dur
        s = max(0.0, c - wid / 2.0)
        e = min(dur, c + wid / 2.0)
        if e <= s:
            e = min(dur, s + 0.1)
        samples.append(Sample(
            sample_id=f"{m['video_id']}::{i}", video_id=m["video_id"], query=m["query"],
            duration=dur, target_spans=((float(m["gt_start"]), float(m["gt_end"])),)))
        preds.append(Prediction(pred_spans=((s, e),)))
    rows = LengthStratifiedGrounding(CharadesTimeLensBenchmark()).to_long_table(preds, samples)
    sp = signed_length_profile(rows)
    b = next((x for x in sp if x["bin"] == "[0,2)"), {})
    std_s = float((var.sqrt() * torch.tensor([float(m["duration"]) for m in metas])).mean())
    return b, std_s


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cd = str(C.CACHE_DIR)
    print(f"device={device}; reading per-frame caches ...", flush=True)
    tr_clips, _, tr_cw = read_clips(cd, ["train_natural.perframe_lim600", "train_speedaug.perframe_lim600"])
    ev_clips, ev_metas, _ = read_clips(cd, ["eval.perframe"])
    print(f"train N={len(tr_clips)}  eval<2s N={len(ev_clips)}  "
          f"T range train=[{min(c['T'] for c in tr_clips)},{max(c['T'] for c in tr_clips)}] "
          f"eval=[{min(c['T'] for c in ev_clips)},{max(c['T'] for c in ev_clips)}]", flush=True)
    print(f"\n{'layer':>6}{'arm':>10}{'<2s|dc|':>9}{'<2sR@0.7':>9}{'predstd':>9}")
    for layer in LAYERS:
        Vtr, mtr, ptr, htr = pad_layer(tr_clips, layer)
        Vev, mev, pev, hev = pad_layer(ev_clips, layer)
        for arm in ("plain", "unimodal"):
            head = train_head(Vtr, mtr, ptr, htr, tr_cw, arm, device)
            b, std_s = score_head(head, Vev, mev, pev, hev, ev_metas, device)
            print(f"{layer:>6}{arm:>10}{b.get('med_abs_center_err', float('nan')):>9.2f}"
                  f"{b.get('R@1_IoU0.7', float('nan')):>9.3f}{std_s:>9.2f}", flush=True)
    print("\nBaselines <2s: text-decode |dc|=1.05s R@0.7=0.015 ; pooled-256 probe |dc|~2.3s (RED).")
    print("GATE: any layer <2s |dc| < ~1.0s => sub-second center IS in per-frame features (readout branch).")


if __name__ == "__main__":
    main()
