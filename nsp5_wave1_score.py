"""Score wave 1's GPU studies (NeedleScan METHODS-LOG 2026-11-12, registered in da7940b) once their
arms are merged: paired differences in plain accuracy with 95% intervals from the bootstrap that
resamples videos within each benchmark, 5,000 draws, seed 0 (nsp5_pooled.pooled).

    .venv/bin/python nsp5_wave1_score.py S     fresh CG-Bench questions: S1 M2 - GMM-EVA's selection, S2 M2 - uniform
    .venv/bin/python nsp5_wave1_score.py S128  the same at 128 frames: S3, and S4 (128 minus 64 per question)
    .venv/bin/python nsp5_wave1_score.py T     stacking, LVBench + CG-Bench at 64 and 128 frames: T1-T3
    .venv/bin/python nsp5_wave1_score.py U     scorer x query x rule on the five benchmarks: U1-U3, Shapley split
"""
from __future__ import annotations

import sys
from itertools import permutations

from nsp5_pooled import BENCH, load, path, pooled

LC = ("lvbench", "cgbench")
# Study U's grid: (scorer, query, rule) -> arm; BOLT as released is the first cell, M2 the last
GRID = {("CLIP", "question", "power"): "bolt_own", ("CLIP", "question", "softmax"): "kelly_clip_eps0",
        ("CLIP", "options", "power"): "u_clipopt_its", ("CLIP", "options", "softmax"): "u_clipopt_kelly_eps0",
        ("8B", "question", "power"): "u_q8stem_its", ("8B", "question", "softmax"): "u_q8stem_kelly_eps0",
        ("8B", "options", "power"): "bolt_its_q8", ("8B", "options", "softmax"): "q8_kelly_eps0"}
BASE, TOP = ("CLIP", "question", "power"), ("8B", "options", "softmax")
FACTORS = ("scorer", "query", "rule")


def show(label: str, a: str, r: str, benches: tuple) -> tuple:
    rows, d, (lo, hi) = pooled(a, r, benches=benches)
    n = sum(x[1] for x in rows)
    acc_a = sum(x[2] * x[1] for x in rows) / n
    acc_r = sum(x[3] * x[1] for x in rows) / n
    print(f"{label}: {a} minus {r}: {acc_a:.2f} vs {acc_r:.2f}, {d:+.2f} [{lo:+.2f}, {hi:+.2f}] (n={n})")
    for b, k, x, y, dd in rows:
        print(f"    {b:<14} n={k:5d}  {x:5.2f} vs {y:5.2f}  {dd:+6.2f}")
    return d, lo, hi


def accuracy(arm: str, benches: tuple = BENCH) -> float:
    right = n = 0
    for b in benches:
        rows = load(path(arm, b))
        right += sum(rows.values())
        n += len(rows)
    return 100 * right / n


def study_s() -> None:
    fresh = ("cgbench_fresh",)
    show("S1 (primary)", "q8_kelly_eps0", "f2c_topk_nr", fresh)
    show("S2", "q8_kelly_eps0", "uniform", fresh)


def study_s128() -> None:
    """Study S part 2 (registered 2026-11-13): S3 at 128 frames, and S4, the per-question difference
    at 128 frames minus that at 64 on the same new questions, videos resampled (5,000 draws, seed 0)."""
    import numpy as np
    from nsp5_compare import video_of
    fresh = ("cgbench_fresh",)
    show("S3 (primary, 128 frames)", "n128_q8_kelly_eps0", "n128_f2c_topk", fresh)
    arms = [load(path(a, "cgbench_fresh")) for a in
            ("n128_q8_kelly_eps0", "n128_f2c_topk", "q8_kelly_eps0", "f2c_topk_nr")]
    keys = sorted(set.intersection(*(set(a) for a in arms)))
    d = np.array([(arms[0][k] - arms[1][k]) - (arms[2][k] - arms[3][k]) for k in keys])
    _, inv = np.unique([video_of(k) for k in keys], return_inverse=True)
    sums, cnts = np.bincount(inv, weights=d), np.bincount(inv).astype(float)
    idx = np.random.default_rng(0).integers(0, len(sums), size=(5000, len(sums)))
    lo, hi = np.percentile(sums[idx].sum(1) / cnts[idx].sum(1), [2.5, 97.5])
    print(f"S4 (128 minus 64 frames): {100 * d.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] (n={len(keys)})")


def study_t() -> None:
    arms = {64: ("q8_kelly_eps0", "t_cap1", "q8opt_topk_nr", "t_stack4"),
            128: ("n128_q8_kelly_eps0", "n128_t_cap1", "n128_q8opt_topk", "n128_t_stack4")}
    for n, (m2, cap, top, stack) in arms.items():
        print(f"== {n} frames, LVBench and CG-Bench pooled")
        show(f"T1 (primary, N={n}) capped M2 - top-k", cap, top, LC)
        show(f"T2 (N={n}) M2 - capped M2", m2, cap, LC)
        show(f"T3 (N={n}) stacked top-k - top-k", stack, top, LC)


def study_u() -> None:
    print("== U1 (primary): calibrated softmax minus BOLT's power, per scorer x query cell")
    for scorer in ("CLIP", "8B"):
        for query in ("question", "options"):
            show(f"U1 {scorer}, {query}", GRID[(scorer, query, "softmax")],
                 GRID[(scorer, query, "power")], BENCH)
    print("== U2: options minus question only, under M2's placement")
    for scorer in ("CLIP", "8B"):
        show(f"U2 {scorer}", GRID[(scorer, "options", "softmax")], GRID[(scorer, "question", "softmax")],
             BENCH)
    print("== U3: the 8B embedder minus CLIP, with options and the softmax")
    show("U3", GRID[("8B", "options", "softmax")], GRID[("CLIP", "options", "softmax")], BENCH)
    acc = {cell: accuracy(arm) for cell, arm in GRID.items()}
    print("== pooled accuracy per cell")
    for cell, a in acc.items():
        print(f"    {', '.join(cell):30s} {GRID[cell]:22s} {a:6.2f}")

    def value(switched: frozenset) -> float:
        return acc[tuple(TOP[i] if FACTORS[i] in switched else BASE[i] for i in range(3))]

    shapley = {f: 0.0 for f in FACTORS}
    orders = list(permutations(FACTORS))
    for order in orders:
        on: set = set()
        for f in order:
            shapley[f] += (value(frozenset(on | {f})) - value(frozenset(on))) / len(orders)
            on.add(f)
    total = acc[TOP] - acc[BASE]
    print(f"== Shapley split of BOLT as released ({acc[BASE]:.2f}) to M2 ({acc[TOP]:.2f}), {total:+.2f} points")
    for f, v in shapley.items():
        print(f"    {f:7s} {v:+6.2f}  ({100 * v / total:.0f}%)")


if __name__ == "__main__":
    {"S": study_s, "S128": study_s128, "T": study_t, "U": study_u}[sys.argv[1]]()
