#!/usr/bin/env python3
"""Study AB's knob choice: the stacking depth m of stacked top-k (the ceil(N / m) highest-scoring
seconds, m frames evenly inside each; `--alloc topk_stack --stack m`), chosen on the calibration
split by the log coverage C of N frames, as Study X5 chose its knobs (nsp5_x5_knob.py). C is the
mean over questions of log(1 + n_E), n_E the frames whose second lies on the annotated evidence; the
8B per-option scores; the harness's own placement code (eval_lvbench_foveate.curve_times). No answer
model runs.

    .venv/bin/python nsp5_ab_stack_knob.py [N]       # from research/TimeLens; N defaults to 64
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "..", "docs", "research", "projects", "NS-P5-adaptive-sampling",
                     "paper", "cvpr2027", "tools")
os.chdir(HERE)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 64
sys.argv_saved, sys.argv = sys.argv, sys.argv[:1]      # the eval module parses no args at import
import eval_lvbench_foveate as ev                          # noqa: E402
sys.argv = sys.argv_saved
sys.path.insert(0, TOOLS)
import calibration_checks as cc                            # noqa: E402
os.chdir(HERE)

DEPTHS = (1, 2, 4, 8, 16)


def log_coverage(z: list, m: list, alloc: str, **kw) -> float:
    vals = []
    for c, mask in zip(z, m):
        times = ev.curve_times(c, float(len(c)), N, alloc, tau=3.58, eps=0.0, **kw)
        secs = np.clip(np.floor(np.asarray(times)).astype(int), 0, len(c) - 1)
        vals.append(np.log1p(mask[secs].sum()))
    return float(np.mean(vals))


def main() -> None:
    z, m, vids = cc.calsplit("maxopt")
    print(f"calibration split: {len(z)} questions, {len(set(vids))} videos; {N} frames")
    cov = {d: log_coverage(z, m, "topk_stack", stack=d) for d in DEPTHS}
    for d in DEPTHS:
        print(f"  topk_stack stack = {d:2d}   C = {cov[d]:.4f}")
    print(f"  (CaliFrame, kelly tau 3.58 eps 0: C = {log_coverage(z, m, 'kelly'):.4f})")
    print("chosen by C: stack =", max(DEPTHS, key=lambda d: cov[d]))


if __name__ == "__main__":
    main()
