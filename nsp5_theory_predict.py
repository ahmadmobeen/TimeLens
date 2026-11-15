#!/usr/bin/env python3
"""Theory check 5 (supersedes the slope and coverage of checks 1-2): accuracy gaps predicted from the
frames each placement puts on the annotated evidence, with Study V's controlled slope over its
registered population.

Law (Study V): accuracy = a + b log(1 + k), k = frames on the evidence. b is the slope of the dose
means on log(1 + k) over the registered population (questions whose logged k equals the dose in every
arm), which equals the question-fixed-effects linear slope because every question sees every dose.
(The 13.2 / 16.1 in v_dose.json keep only questions whose answer changes with the dose; they inflate
the linear slope by population / informative questions.) For each arm and benchmark,
S = mean log(1 + in_span), in_span the harness's logged count of frames within the evidence +-0.5 s;
the predicted gap between two arms is b (S_A - S_B), pooled over benchmarks by question counts, and is
printed beside the measured paired gap (videos resampled, 5,000 draws). No answer-model output enters
S; b comes from the dose arms, which ran on the same benchmarks.

    .venv/bin/python nsp5_theory_predict.py          # from research/TimeLens
"""
import json
import math

import numpy as np

from nsp5_pooled import path, pooled

LC = ("lvbench", "cgbench")
DOSES = {"lvbench": (0, 1, 2, 4, 8, 16, 32), "cgbench": (0, 1, 4, 16)}
CAL = "q8_kelly_eps0"
PAIRS = [  # (label, CaliFrame-side arm, other arm)
    ("uniform, 16", "n16_q8_kelly_eps0", "n16_uniform"), ("uniform, 32", "n32_q8_kelly_eps0", "n32_uniform"),
    ("uniform, 64", CAL, "e3_uniform:uniform"), ("uniform, 128", "n128_q8_kelly_eps0", "n128_uniform"),
    ("uniform, 256", "n256_q8_kelly_eps0", "n256_uniform"),
    ("top-k, 64", CAL, "q8opt_topk_nr"), ("top-k, 128", "n128_q8_kelly_eps0", "n128_q8opt_topk"),
    ("stacked top-k, 64", CAL, "t_stack4"), ("stacked top-k, 128", "n128_q8_kelly_eps0", "n128_t_stack4"),
    ("capped, 64", CAL, "t_cap1"), ("capped, 128", "n128_q8_kelly_eps0", "n128_t_cap1"),
    ("water-filling, 64", CAL, "y_waterfill"),
    ("GMM-EVA's selection, 16", "n16_q8_kelly_eps0", "n16_f2c_topk"),
    ("GMM-EVA's selection, 32", "n32_q8_kelly_eps0", "n32_f2c_topk"),
    ("GMM-EVA's selection, 64", CAL, "f2c_topk_nr"),
    ("GMM-EVA's selection, 128", "n128_q8_kelly_eps0", "n128_f2c_topk"),
    ("GMM-EVA's selection, 256", "n256_q8_kelly_eps0", "n256_f2c_topk"),
] + [(f"tau {t}", f"q8_kelly_eps0_tau{s}", CAL) for t, s in
     (("1.00", "100"), ("1.79", "179"), ("2.50", "250"), ("5.00", "500"), ("7.16", "716"), ("10.0", "1000"))]


def rows(arm: str, bench: str) -> dict:
    try:
        with open(path(arm, bench), encoding="utf-8") as fh:
            return {r["key"]: r for r in (json.loads(l) for l in fh if l.strip())}
    except FileNotFoundError:
        return {}


def slope(bench: str) -> tuple:
    arms = {k: rows(f"v_dose{k}", bench) for k in DOSES[bench]}
    keys = set.intersection(*(set(a) for a in arms.values()))
    pop = [q for q in keys if all(arms[k][q].get("in_span") == k for k in DOSES[bench])]
    x = np.array([math.log1p(k) for k in DOSES[bench]])
    y = np.array([100 * np.mean([arms[k][q]["correct"] for q in pop]) for k in DOSES[bench]])
    b = float(np.polyfit(x, y, 1)[0])
    return b, len(pop), len(keys)


def coverage(arm: str, bench: str) -> tuple:
    r = rows(arm, bench)
    ks = [v["in_span"] for v in r.values() if v.get("in_span") is not None]
    return (float(np.mean(np.log1p(ks))), len(ks)) if ks else (float("nan"), 0)


def main() -> None:
    B = {}
    for bench in LC:
        b, n, tot = slope(bench)
        B[bench] = b
        print(f"{bench}: controlled slope b = {b:.2f} points per unit log(1 + k) "
              f"(registered population {n} of {tot} questions)")
    print(f"\n{'comparison':26s} {'pred':>6} | {'measured':>22} | inside")
    inside = []
    for label, a, o in PAIRS:
        num = den = 0.0
        ok = True
        for bench in LC:
            sa, na = coverage(a, bench)
            so, no = coverage(o, bench)
            if not na or not no or math.isnan(sa) or math.isnan(so):
                ok = False
                break
            n = min(na, no)
            num += B[bench] * (sa - so) * n
            den += n
        if not ok:
            print(f"{label:26s}   (arm or in_span missing)")
            continue
        pred = num / den
        try:
            _, d, (lo, hi) = pooled(a, o, benches=LC)
        except FileNotFoundError:
            print(f"{label:26s} {pred:+6.2f} | (no measured arm)")
            continue
        hit = lo <= pred <= hi
        inside.append((label, hit))
        print(f"{label:26s} {pred:+6.2f} | {d:+6.2f} [{lo:+6.2f}, {hi:+6.2f}] | {'yes' if hit else 'no'}")
    for group in ("uniform", "top-k", "stacked", "capped", "water", "GMM", "tau"):
        g = [h for lbl, h in inside if lbl.startswith(group)]
        if g:
            print(f"  {group}: {sum(g)} of {len(g)} predictions inside the measured interval")


if __name__ == "__main__":
    main()
