#!/usr/bin/env python3
"""Study X5's registered knob choice, recomputed: the log coverage C of 64 frames on the
calibration split for top-k with non-maximum suppression (radius r in {2, 5, 10, 30, 60} s) and key
clips (clip length c in {2, 4, 8} s), on the 8B per-option scores, with the harness's own placement
code (eval_lvbench_foveate.curve_times). C is the mean over questions of log(1 + n_E), n_E the frames
whose second lies on the annotated evidence. No answer model runs.

    .venv/bin/python nsp5_x5_knob.py          # from research/TimeLens
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "..", "docs", "research", "projects", "NS-P5-adaptive-sampling",
                     "paper", "cvpr2027", "tools")
os.chdir(HERE)
sys.argv_saved, sys.argv = sys.argv, sys.argv[:1]      # the eval module parses no args at import
import eval_lvbench_foveate as ev                          # noqa: E402
sys.argv = sys.argv_saved
sys.path.insert(0, TOOLS)
import calibration_checks as cc                            # noqa: E402
os.chdir(HERE)

N = 64
GRID = [("topk_nms", "nms_radius", r) for r in (2, 5, 10, 30, 60)] + \
       [("keyclip", "clip_len", c) for c in (2, 4, 8)]


def log_coverage(z: list, m: list, alloc: str, knob: str, value) -> float:
    vals = []
    for c, mask in zip(z, m):
        times = ev.curve_times(c, float(len(c)), N, alloc, tau=3.58, eps=0.0, **{knob: value})
        secs = np.clip(np.floor(np.asarray(times)).astype(int), 0, len(c) - 1)
        vals.append(np.log1p(mask[secs].sum()))
    return float(np.mean(vals))


def main() -> None:
    z, m, vids = cc.calsplit("maxopt")
    print(f"calibration split: {len(z)} questions, {len(set(vids))} videos; {N} frames")
    best = {}
    for alloc, knob, value in GRID:
        cov = log_coverage(z, m, alloc, knob, value)
        print(f"  {alloc:9s} {knob} = {value:3d}   C = {cov:.4f}")
        if cov > best.get(alloc, (None, -1.0))[1]:
            best[alloc] = (value, cov)
    print("chosen by C:", {a: v for a, (v, _) in best.items()})


if __name__ == "__main__":
    main()
