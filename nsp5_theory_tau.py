#!/usr/bin/env python3
"""Theory check 1: does accuracy across temperatures follow from the calibration split's likelihood?

Horse-race identity: placing N frames in proportion to p puts N p(E) of them on the evidence E, so
with accuracy a + b log(frames on the evidence) the expected accuracy is a + b (log N - NLL(p)),
NLL the expected negative log of p's mass on the evidence. Between two temperatures the predicted
accuracy difference is -b (NLL(tau) - NLL(3.58)). With the saturating law a + b log(1 + k) the
prediction uses S(tau) = E log(1 + N p(E)) instead. Both are computed on the calibration split
(872 CG-Bench questions on 101 videos no evaluation question uses), so nothing is fitted on an
evaluation set; b is Study V's controlled slope. The measured differences come from the logged
temperature grid (Qwen2.5-VL-7B, 64 frames, the five benchmarks).

    .venv/bin/python nsp5_theory_tau.py            # from research/TimeLens
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "..", "..", "docs", "research", "projects", "NS-P5-adaptive-sampling",
                     "paper", "cvpr2027", "tools")
sys.path.insert(0, TOOLS)
import calibration_checks as cc                            # noqa: E402  (chdirs to TimeLens)
from nsp5_method_figures import nll                        # noqa: E402
from nsp5_pooled import BENCH, load, path, pooled          # noqa: E402

N = 64
GRID = {1.0: "q8_kelly_eps0_tau100", 1.79: "q8_kelly_eps0_tau179", 2.5: "q8_kelly_eps0_tau250",
        3.58: "q8_kelly_eps0", 5.0: "q8_kelly_eps0_tau500", 7.16: "q8_kelly_eps0_tau716",
        10.0: "q8_kelly_eps0_tau1000"}


def mass(z: np.ndarray, m: np.ndarray, t: float) -> float:
    return float(np.exp(-nll(z, m, t)))


def accuracy(arm: str, benches: tuple) -> float:
    right = n = 0
    for b in benches:
        rows = load(path(arm, b))
        right += sum(rows.values())
        n += len(rows)
    return 100.0 * right / n


def main() -> None:
    z, m, vids = cc.calsplit("maxopt")
    v = json.load(open(os.path.join("logs", "nsp5", "analysis", "v_dose.json")))
    slopes = {b: v[b]["fit"]["forms"]["log"]["lpm_points"][0] for b in ("lvbench", "cgbench")}
    beta = float(np.mean(list(slopes.values())))
    print(f"calibration split: {len(z)} questions, {len(set(vids))} videos; N = {N}")
    print(f"Study V controlled slope (points per unit log(1+k)): LVBench {slopes['lvbench']:.2f}, "
          f"CG-Bench {slopes['cgbench']:.2f}; mean {beta:.2f}")
    rows = []
    for t, arm in GRID.items():
        nl = float(np.mean([nll(c, x, t) for c, x in zip(z, m)]))
        s = float(np.mean([np.log1p(N * mass(c, x, t)) for c, x in zip(z, m)]))
        rows.append((t, arm, nl, s, accuracy(arm, BENCH), accuracy(arm, ("lvbench", "cgbench"))))
    ref = next(r for r in rows if r[0] == 3.58)
    print(f"\n{'tau':>5} {'NLL':>6} {'S':>6} | {'pred dNLL':>9} {'pred dS':>8} | "
          f"{'meas 5 bench':>12} {'meas LV+CG':>10}")
    pred_l, pred_s, meas5, meas2 = [], [], [], []
    for t, arm, nl, s, a5, a2 in rows:
        pl, ps = -beta * (nl - ref[2]), beta * (s - ref[3])
        d5, d2 = a5 - ref[4], a2 - ref[5]
        pred_l.append(pl), pred_s.append(ps), meas5.append(d5), meas2.append(d2)
        print(f"{t:5.2f} {nl:6.3f} {s:6.3f} | {pl:+9.2f} {ps:+8.2f} | {d5:+12.2f} {d2:+10.2f}")
    for name, pred in (("log k (NLL)", pred_l), ("log(1+k) (S)", pred_s)):
        p, q5, q2 = np.array(pred), np.array(meas5), np.array(meas2)
        k5 = float(p @ q5 / (p @ p)) if p @ p > 0 else float("nan")
        k2 = float(p @ q2 / (p @ p)) if p @ p > 0 else float("nan")
        print(f"{name}: corr with measured, 5 benchmarks {np.corrcoef(p, q5)[0, 1]:+.3f}, LV+CG "
              f"{np.corrcoef(p, q2)[0, 1]:+.3f}; measured = {k5:.2f} x predicted (5), {k2:.2f} x (LV+CG)")
    print("\nPaired intervals of the measured differences (videos resampled, 5,000 draws):")
    for t, arm, *_ in rows:
        if arm != "q8_kelly_eps0":
            _, d, (lo, hi) = pooled(arm, "q8_kelly_eps0")
            print(f"  tau {t:5.2f} minus 3.58: {d:+.2f} [{lo:+.2f}, {hi:+.2f}]")


if __name__ == "__main__":
    main()
