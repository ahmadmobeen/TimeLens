"""Selector-by-budget trend (Study G, METHODS-LOG 2026-11-05). For each budget N, two arms A_N and
B_N answer the same questions; d(N) is A minus B, plain accuracy pooled over the five benchmarks.
The primary statistic is the OLS slope of d(N) on log2 N, with a 95% interval from a paired
bootstrap that resamples videos within each benchmark, the same draw for every N so the budgets
stay paired (CONVENTIONS section 7); 5,000 draws, seed 0. Each d(N) is printed with its own
interval and a Holm-adjusted two-sided p-value across the budgets.

    .venv/bin/python nsp5_trend.py n16_f2c_kelly_eps0:n16_f2c_topk:16 \\
        n32_f2c_kelly_eps0:n32_f2c_topk:32 f2c_kelly_eps0:f2c_topk_nr:64 \\
        n128_f2c_kelly_eps0:n128_f2c_topk:128

An arm is named by its LVBench stem; on the other benchmarks its file is x1_<bench>_<stem>.
"""
import sys

import numpy as np

from nsp5_compare import video_of
from nsp5_pooled import BENCH, load, path

DRAWS, SEED = 5000, 0


def main() -> None:
    specs = [s.split(":") for s in sys.argv[1:]]
    budgets = np.array([float(n) for _, _, n in specs])
    x = np.log2(budgets)
    per_bench = []                       # per benchmark: (video sums [n_budgets, n_videos], video counts)
    for b in BENCH:
        A = [load(path(a, b)) for a, _, _ in specs]
        B = [load(path(r, b)) for _, r, _ in specs]
        keys = sorted(set.intersection(*(set(t) for t in A + B)))
        _, inv = np.unique([video_of(k) for k in keys], return_inverse=True)
        sums = np.stack([np.bincount(inv, weights=np.array([a[k] - r[k] for k in keys]))
                         for a, r in zip(A, B)])
        per_bench.append((sums, np.bincount(inv).astype(float), len(keys)))
    total = sum(n for _, _, n in per_bench)

    def stat(draw=None):
        s = np.zeros(len(specs))
        c = 0.0
        for j, (sums, cnts, _) in enumerate(per_bench):
            idx = slice(None) if draw is None else draw[j]
            s += sums[:, idx].sum(1)
            c += cnts[idx].sum()
        d = 100 * s / c
        return d, np.polyfit(x, d, 1)[0]

    d0, slope0 = stat()
    rng = np.random.default_rng(SEED)
    ds, slopes = [], []
    for _ in range(DRAWS):
        draw = [rng.integers(0, len(cnts), len(cnts)) for _, cnts, _ in per_bench]
        d, sl = stat(draw)
        ds.append(d)
        slopes.append(sl)
    ds, slopes = np.array(ds), np.array(slopes)
    print(f"questions per budget: {total}")
    p = [min(1.0, 2 * min(np.mean(ds[:, i] <= 0), np.mean(ds[:, i] >= 0))) for i in range(len(specs))]
    order = np.argsort(p)
    holm = np.empty(len(p))
    run = 0.0
    for rank, i in enumerate(order):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        holm[i] = run
    for i, (a, r, n) in enumerate(specs):
        lo, hi = np.percentile(ds[:, i], [2.5, 97.5])
        print(f"  N={n:>4}: {a} minus {r}: {d0[i]:+.2f} [{lo:+.2f}, {hi:+.2f}]  Holm p={holm[i]:.3f}")
    lo, hi = np.percentile(slopes, [2.5, 97.5])
    print(f"slope of the difference per doubling of N: {slope0:+.3f} [{lo:+.3f}, {hi:+.3f}]")


if __name__ == "__main__":
    main()
