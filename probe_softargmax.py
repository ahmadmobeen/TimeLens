"""NS-P1 round-1 experiment #A (cheap gate): query-conditioned soft-argmax center head.

Tests whether sub-second CENTER is extractable from the FROZEN pooled visual stream
``vis[256,3584]`` ALREADY on disk (post-answer caches; no new GPU caching). The per-token
saliency is CONDITIONED on the cached answer-aware state ``h_ans[3584]`` (attention query over
the visual tokens) -> softmax over time -> expected center. Two arms:
  plain     = Huber(center) + Huber(width)
  unimodal  = + lambda * |Var_p[t] - sigma*^2|  (anti-collapse: force a NARROW, committed peak)

Why query-conditioned: a soft-argmax over vis alone is query-AGNOSTIC and can only learn the
dataset-mean center. Conditioning the attention on h_ans makes it "where in time do the visual
tokens match the answered moment" -- the readout that could recover sub-second center the text
decode discarded. (Distinct from the falsified C `xattn` arm, which REGRESSED (c,w) via an MLP on
pooled cross-attn; here the center IS the soft-argmax expectation + a sharpness penalty.)

Decisive gate (methodology-directions-round1.md #A):
  - Neither arm beats the model's own text-decode |dCenter| (~1.05 s) on <2 s
      => pooled visual features do not carry extractable sub-second center (publishable negative).
  - It does (esp. unimodal > plain)
      => signal EXISTS; lever is readout/objective, not backbone => greenlight EC-Sharp / zoom.

Caveat: vis[256] is order-preserving avg-pooled over ~14k visual tokens -> coarse temporal proxy
for true per-frame hiddens. GREEN is strong; RED is suggestive (confirm with per-frame caching).

Run from the repo ROOT (main venv has torch + benchmarks + needlescan):
  env -u VIRTUAL_ENV uv run python research/TimeLens/probe_softargmax.py [--smoke]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # so `import c_head.*` resolves when run from repo root

from c_head import config as C  # noqa: E402
from benchmarks.datasets.public.charades_timelens import CharadesTimeLensBenchmark  # noqa: E402
from benchmarks.length_profile import signed_length_profile  # noqa: E402
from benchmarks.length_strata import LengthStratifiedGrounding  # noqa: E402
from needlescan.base.types import Prediction, Sample  # noqa: E402

NTOK = 256


def load_cache(cache_dir, tags):
    """Concatenate vis[N,256,3584], h_ans[N,3584] + (center,width,dur) targets across tags."""
    vis, hans, metas = [], [], []
    for tag in tags:
        for f in sorted(Path(cache_dir).glob(f"{tag}.shard*of*.pt")):
            d = torch.load(f, map_location="cpu")
            if not len(d["meta"]):
                continue
            vis.append(d["vis"])
            hans.append(d["h_ans"])
            metas.extend(d["meta"])
    vis = torch.cat(vis)
    hans = torch.cat(hans)
    cw, keep = [], []
    for i, m in enumerate(metas):
        dur, s, e = float(m["duration"]), float(m["gt_start"]), float(m["gt_end"])
        if dur <= 0 or e <= s:
            continue
        cw.append([min(max(((s + e) / 2.0) / dur, 0.0), 1.0), min(max((e - s) / dur, 1e-4), 1.0), dur])
        keep.append(i)
    return vis[keep], hans[keep], torch.tensor(cw, dtype=torch.float32), [metas[i] for i in keep]


class QSoftArgmaxCenter(torch.nn.Module):
    """h_ans-conditioned attention over visual tokens -> soft-argmax expected center; + width head."""

    def __init__(self, d: int = 3584, p: int = NTOK) -> None:
        super().__init__()
        self.vnorm = torch.nn.LayerNorm(d)
        self.qnorm = torch.nn.LayerNorm(d)
        self.qproj = torch.nn.Linear(d, d)
        self.kproj = torch.nn.Linear(d, d)
        self.wnorm = torch.nn.LayerNorm(2 * d)
        self.wfc = torch.nn.Linear(2 * d, 1)
        self.scale = d ** -0.5
        self.register_buffer("pos", (torch.arange(p).float() + 0.5) / p)

    def forward(self, vis, hans):  # vis [B,P,d], hans [B,d] -> center[B], width[B], var[B]
        q = self.qproj(self.qnorm(hans))                       # [B,d]
        k = self.kproj(self.vnorm(vis))                        # [B,P,d]
        logit = torch.einsum("bpd,bd->bp", k, q) * self.scale  # [B,P]
        pr = torch.softmax(logit, dim=-1)
        center = (pr * self.pos).sum(-1)
        var = (pr * (self.pos[None, :] - center[:, None]) ** 2).sum(-1)
        w = torch.sigmoid(self.wfc(self.wnorm(torch.cat([hans, vis.mean(1)], dim=-1)))).squeeze(-1)
        return center, w, var


def huber(a, b, delta=0.05):
    e = (a - b).abs()
    return torch.where(e < delta, 0.5 * e * e / delta, e - 0.5 * delta).mean()


def train_arm(vis, hans, cw, arm, device, epochs, bs=256, lr=5e-4, lam=1.0, sigma_s=0.3, seed=0):
    torch.manual_seed(seed)
    head = QSoftArgmaxCenter().to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=lr, weight_decay=1e-4)
    n = cw.shape[0]
    tr = torch.randperm(n, generator=torch.Generator().manual_seed(seed))[max(1, n // 20):]
    loader = DataLoader(TensorDataset(vis[tr], hans[tr], cw[tr]), batch_size=bs, shuffle=True)
    for _ in range(epochs):
        for vb, hb, cwb in loader:
            vb, hb, cwb = vb.float().to(device), hb.float().to(device), cwb.to(device)
            center, w, var = head(vb, hb)
            loss = huber(center, cwb[:, 0]) + huber(w, cwb[:, 1])
            if arm == "unimodal":
                tv = (sigma_s / cwb[:, 2]).clamp(max=0.5) ** 2
                loss = loss + lam * (var - tv).abs().mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
    return head


@torch.no_grad()
def score_arm(head, cache_dir, device):
    vis, hans, _, metas = load_cache(cache_dir, ["eval.postans"])
    head.eval()
    center, w, var = head(vis.float().to(device), hans.float().to(device))
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
    mean_std_s = float((var.sqrt() * torch.tensor([float(m["duration"]) for m in metas])).mean())
    return signed_length_profile(rows), mean_std_s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=80)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cd = str(C.CACHE_DIR)
    print(f"device={device}; loading train cache (natural + speedaug)...", flush=True)
    vis, hans, cw, _ = load_cache(cd, ["train_natural.postans", "train_speedaug.postans_limit750"])
    print(f"train N={cw.shape[0]}  vis={tuple(vis.shape)}  hans={tuple(hans.shape)}", flush=True)
    epochs = 3 if args.smoke else args.epochs
    for arm in ("plain", "unimodal"):
        head = train_arm(vis, hans, cw, arm, device, epochs)
        sp, mean_std_s = score_arm(head, cd, device)
        print(f"\n========== arm={arm}  epochs={epochs}  mean_pred_temporal_std={mean_std_s:.2f}s ==========")
        print(f"{'bin':>9}{'n':>6}{'|dctr|':>8}{'R@0.5':>7}{'R@0.7':>7}{'mIoU':>7}{'med_log2':>10}")
        for d in sp:
            if d["n"] == 0:
                continue
            print(f"{d['bin']:>9}{int(d['n']):>6}{d.get('med_abs_center_err', float('nan')):>8.2f}"
                  f"{d.get('R@1_IoU0.5', float('nan')):>7.3f}{d.get('R@1_IoU0.7', float('nan')):>7.3f}"
                  f"{d.get('mIoU', float('nan')):>7.3f}{d.get('med_log2_ratio', float('nan')):>10.3f}", flush=True)
    print("\nBaselines on <2s: text-decode |dctr|=1.05s R@0.7=0.015 ; frozen (c,w) head |dctr|=1.82s R@0.7=0.000")
    print("GATE: a soft-argmax arm with <2s |dctr| < ~1.0s (esp. unimodal > plain) => signal EXISTS in visual features.")


if __name__ == "__main__":
    main()
