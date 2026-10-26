"""Fuse SigLIP and Qwen3-VL-Embedding-8B per-second curves into the frozen A3 score.

The rule fixed in METHODS-LOG (A3): per second, SigLIP-so400m's max over (stem + option_k)
divided by 1.59 plus the 8B embedder's divided by 3.58, i.e. the sum of the two
log-posteriors at their 5-fold-CV temperatures. The normalising constants drop out of every
allocation rule, so they are not added. Frames are then allocated from softmax(fused / 1.76)
by eval_lvbench_foveate.py --arm curve.

Arithmetic in float64, stored as float32: this reproduces the LVBench file the A3 results
were computed from (data/t1/curves/fused_siglip_q8b_maxopt.npz) bit for bit.

    .venv/bin/python nsp5_fuse_curves.py \\
        --siglip data/t1/priors/siglip_so400m/videomme_curves_maxopt.npz \\
        --q8b data/t1/priors/qwen3vl_emb_8b/videomme_curves_maxopt.npz \\
        --out data/t1/priors/fused_siglip_q8b/videomme_curves_maxopt.npz
"""
import argparse
import os

import numpy as np


def fuse(siglip: np.ndarray, q8b: np.ndarray, tau_siglip: float, tau_q8b: float) -> np.ndarray:
    return (siglip.astype(np.float64) / tau_siglip + q8b.astype(np.float64) / tau_q8b).astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--siglip", required=True, help="npz of SigLIP max-over-options curves")
    ap.add_argument("--q8b", required=True, help="npz of Qwen3-VL-Embedding-8B max-over-options curves")
    ap.add_argument("--out", required=True)
    ap.add_argument("--tau_siglip", type=float, default=1.59)
    ap.add_argument("--tau_q8b", type=float, default=3.58)
    args = ap.parse_args()

    if os.path.exists(args.out):
        raise SystemExit(f"{args.out} exists; running jobs may be reading it (np.load is lazy)")
    s, q = np.load(args.siglip), np.load(args.q8b)
    if set(s.files) != set(q.files):
        raise SystemExit(f"curve files cover different questions: {len(s.files)} vs {len(q.files)}")
    out = {}
    for k in s.files:
        a, b = s[k], q[k]
        if len(a) != len(b):
            raise SystemExit(f"{k}: SigLIP curve has {len(a)} seconds, 8B curve {len(b)}")
        out[k] = fuse(a, b, args.tau_siglip, args.tau_q8b)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    np.savez(args.out, **out)
    print(f"fused {len(out)} curves (tau {args.tau_siglip} / {args.tau_q8b}) -> {args.out}")


if __name__ == "__main__":
    main()
