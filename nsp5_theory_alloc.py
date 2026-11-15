#!/usr/bin/env python3
"""Theory check 2: predicted accuracy differences between placement rules and across budgets.

Under the saturating law (accuracy = a + b log(1 + k), k the frames whose second lies on the
annotated evidence; Study V), a placement's expected accuracy is a + b S, S = E log(1 + k). This
script computes S on the calibration split (872 CG-Bench questions on 101 videos no evaluation
question uses; the 8B per-option scores) for each rule with the harness's own placement code
(eval_lvbench_foveate.curve_times) at N = 16, 32, 64, 128 and 256, and prints the predicted
differences b * dS beside the measured ones (Qwen2.5-VL-7B, LVBench and CG-Bench pooled, paired,
videos resampled, 5,000 draws) where the arms exist. b is Study V's controlled slope (mean of
LVBench and CG-Bench). Nothing is fitted on an evaluation set.

    .venv/bin/python nsp5_theory_alloc.py          # from research/TimeLens
"""
import json
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
from nsp5_pooled import pooled                             # noqa: E402
os.chdir(HERE)

BUDGETS = (16, 32, 64, 128, 256)
RULES = {"CaliFrame": ("kelly", {}), "top-k": ("topk", {}), "stacked top-k (4)": ("topk_stack", {"stack": 4}),
         "capped (1 per second)": ("kelly_cap1", {}), "water-filling": ("waterfill", {})}
LC = ("lvbench", "cgbench")
# measured arms: (budget, rule) -> LVBench stem (X1 stem after ':' when it differs)
ARMS = {(64, "CaliFrame"): "q8_kelly_eps0", (128, "CaliFrame"): "n128_q8_kelly_eps0",
        (64, "top-k"): "q8opt_topk_nr", (128, "top-k"): "n128_q8opt_topk",
        (64, "stacked top-k (4)"): "t_stack4", (128, "stacked top-k (4)"): "n128_t_stack4",
        (64, "capped (1 per second)"): "t_cap1", (128, "capped (1 per second)"): "n128_t_cap1"}


def frames_on_evidence(z: np.ndarray, m: np.ndarray, n: int, alloc: str, kw: dict) -> int:
    if alloc == "uniform":
        times = [(i + 0.5) * len(z) / n for i in range(n)]
    else:
        times = ev.curve_times(z, float(len(z)), n, alloc, tau=3.58, eps=0.0, **kw)
    secs = np.clip(np.floor(np.asarray(times)).astype(int), 0, len(z) - 1)
    return int(m[secs].sum())


def main() -> None:
    z, m, vids = cc.calsplit("maxopt")
    v = json.load(open(os.path.join("logs", "nsp5", "analysis", "v_dose.json")))
    beta = float(np.mean([v[b]["fit"]["forms"]["log"]["lpm_points"][0] for b in LC]))
    print(f"calibration split: {len(z)} questions, {len(set(vids))} videos; b = {beta:.2f} points "
          f"per unit log(1 + k)")
    S = {}
    for n in BUDGETS:
        for rule, (alloc, kw) in list(RULES.items()) + [("uniform", ("uniform", {}))]:
            k = np.array([frames_on_evidence(c, x, n, alloc, kw) for c, x in zip(z, m)])
            S[(n, rule)] = (float(np.mean(np.log1p(k))), float(np.mean(k)), float(np.mean(k > 0)))
    print(f"\n{'N':>4} " + " ".join(f"{r[:14]:>15}" for r in list(RULES) + ["uniform"]) + "   (S; mean k; P(k>0))")
    for n in BUDGETS:
        print(f"{n:4d} " + " ".join(f"{S[(n, r)][0]:6.3f} {S[(n, r)][1]:4.1f} {S[(n, r)][2]:.2f}"
                                     for r in list(RULES) + ["uniform"]))
    print("\nPredicted CaliFrame minus each rule, b * dS (points); measured where the arms exist:")
    for rule in list(RULES)[1:] + ["uniform"]:
        cells = []
        for n in BUDGETS:
            pred = beta * (S[(n, "CaliFrame")][0] - S[(n, rule)][0])
            arm_a, arm_b = ARMS.get((n, "CaliFrame")), ARMS.get((n, rule))
            meas = ""
            if arm_a and arm_b:
                _, d, (lo, hi) = pooled(arm_a, arm_b, benches=LC)
                meas = f" | meas {d:+.2f} [{lo:+.2f}, {hi:+.2f}]"
            cells.append(f"N={n}: pred {pred:+.2f}{meas}")
        print(f"  vs {rule}:")
        for c in cells:
            print(f"      {c}")


if __name__ == "__main__":
    main()
