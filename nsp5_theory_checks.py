#!/usr/bin/env python3
"""Theory checks 3 and 4 (CPU only; draft-theory/THEORY.md).

3. The temperature shift (P2): at a finite budget the temperature that maximizes S = E log(1 + N p(E))
   is sharper than the likelihood fit when raising the temperature moves mass from questions the
   forecast hits to those it misses, i.e. when X = d log p(E) / d tau is negatively associated with
   p(E). Computed on the calibration split at the fit (3.58): binned means of X by log p(E), and
   dS_N/dtau = E[X w], w = N p / (1 + N p), at N = 16, 64, 256.
4. Concentration pays only where stacking is possible (P4): top-k maximizes the chance of a first
   frame on the evidence among placements with at most one frame per second, so CaliFrame can lead
   it only where the stack depth D = N p_max exceeds one. CaliFrame minus same-score top-k
   (q8_kelly_eps0 minus q8opt_topk_nr, 64 frames) by bins of D, LVBench and CG-Bench pooled,
   intervals from 5,000 draws of videos (seed 0). D uses the scores only.

    .venv/bin/python nsp5_theory_checks.py          # from research/TimeLens
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "..", "docs", "research", "projects", "NS-P5-adaptive-sampling",
                     "paper", "cvpr2027", "tools")
sys.path.insert(0, TOOLS)
import calibration_checks as cc                            # noqa: E402  (chdirs to TimeLens)
from nsp5_compare import video_of                          # noqa: E402
from nsp5_method_figures import nll                        # noqa: E402
from nsp5_pooled import load, path                         # noqa: E402

TAU, N, H = 3.58, 64, 0.02
PRIOR = "data/t1/priors/qwen3vl_emb_8b/"
EDGES = (0, 1, 2, 4, 8, 16, np.inf)


def check3() -> None:
    z, m, _ = cc.calsplit("maxopt")
    lp = np.array([-nll(c, x, TAU) for c, x in zip(z, m)])
    X = np.array([(nll(c, x, TAU - H) - nll(c, x, TAU + H)) / (2 * H) for c, x in zip(z, m)])
    print(f"== check 3, calibration split ({len(z)} questions), tau = {TAU}")
    print(f"   mean X = {X.mean():+.4f}; corr(X, log p(E)) = {np.corrcoef(X, lp)[0, 1]:+.3f}")
    q = np.quantile(lp, np.linspace(0, 1, 6))
    for a, b in zip(q[:-1], q[1:]):
        sel = (lp >= a) & (lp <= b)
        print(f"   log p(E) in [{a:6.2f}, {b:6.2f}]: n = {sel.sum():3d}, mean X = {X[sel].mean():+.4f}")
    for n in (16, 64, 256):
        w = n * np.exp(lp) / (1 + n * np.exp(lp))
        print(f"   N = {n:3d}: dS_N/dtau at the fit = {np.mean(X * w):+.4f}")


def check4() -> None:
    rows = []
    for b, f, pre in (("lvbench", "curves_maxopt.npz", "lvbench:"), ("cgbench", "cgbench_curves_maxopt.npz", "")):
        curves = np.load(PRIOR + f)
        A, B = load(path("q8_kelly_eps0", b)), load(path("q8opt_topk_nr", b))
        for k in sorted(set(A) & set(B)):
            ck = pre + k if pre + k in curves.files else k
            if ck in curves.files:
                s = curves[ck].astype(float) / TAU
                p = np.exp(s - s.max())
                rows.append((b + video_of(k), N * p.max() / p.sum(), A[k] - B[k]))
    vid = [r[0] for r in rows]
    D = np.array([r[1] for r in rows])
    d = np.array([r[2] for r in rows])
    uv, inv = np.unique(vid, return_inverse=True)
    rng = np.random.default_rng(0)
    print(f"== check 4, LVBench and CG-Bench ({len(rows)} questions), N = {N}")
    for lo, hi in zip(EDGES[:-1], EDGES[1:]):
        sel = (D >= lo) & (D < hi)
        sums = np.bincount(inv[sel], weights=d[sel], minlength=len(uv))
        cnts = np.bincount(inv[sel], minlength=len(uv)).astype(float)
        s, c = sums[cnts > 0], cnts[cnts > 0]
        idx = rng.integers(0, len(s), size=(5000, len(s)))
        a, b = np.percentile(s[idx].sum(1) / c[idx].sum(1), [2.5, 97.5])
        print(f"   D in [{lo:g}, {hi:g}): n = {sel.sum():5d}, CaliFrame minus top-k "
              f"{100 * d[sel].mean():+.2f} [{100 * a:+.2f}, {100 * b:+.2f}]")


if __name__ == "__main__":
    check3()
    check4()
