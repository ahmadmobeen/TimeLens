"""Pooled paired difference over the five NS-P5 benchmarks (LVBench and the four X1 benchmarks):
plain accuracy over every question, with a bootstrap that resamples videos within each benchmark
(stratified by benchmark, clustered by video: CONVENTIONS section 7). The primary test of the
prior-art pre-registration (METHODS-LOG, 2026-10-28) and of the Kelly-vs-top-k pooling.

    .venv/bin/python nsp5_pooled.py q8_kelly f2c_kelly bolt_its_q8 aks_ada_q8 bolt_own aks_own

prints the first arm minus each later one. An arm is named by its LVBench stem; on the other
benchmarks its file is x1_<bench>_<stem>. 'a:b' names an arm whose LVBench stem a differs from
its X1 stem b (the 8B top-k is ge1_opt_topk:q8opt_topk). --x1 pools the four X1 benchmarks
only (9,211 questions), for a rule fitted on LVBench.
"""
import json
import sys

import numpy as np

from nsp5_compare import video_of

L = "logs/nsp5"
SIZE = {"lvbench": 1534, "videomme": 2700, "lvb": 1337, "mlvu": 2174, "cgbench": 3000}
BENCH = tuple(SIZE)


def path(name, b):
    lvb, _, x1 = name.partition(":")
    return f"{L}/{lvb}.jsonl" if b == "lvbench" else f"{L}/x1_{b}_{x1 or lvb}.jsonl"


def load(p):
    rows = (json.loads(l) for l in open(p, encoding="utf-8") if l.strip())
    return {r["key"]: float(r["correct"]) for r in rows}


def pooled(a, r, n=5000, seed=0, benches=BENCH):
    """a minus r: per-benchmark rows (bench, n, acc a, acc r, diff) and the pooled diff with its
    95 % interval, all in points."""
    rows, strata = [], []
    for b in benches:
        A, R = load(path(a, b)), load(path(r, b))
        ks = sorted(set(A) & set(R))
        d = np.array([A[k] - R[k] for k in ks])
        _, inv = np.unique([video_of(k) for k in ks], return_inverse=True)
        strata.append((np.bincount(inv, weights=d), np.bincount(inv).astype(float)))
        rows.append((b, len(ks), 100 * np.mean([A[k] for k in ks]), 100 * np.mean([R[k] for k in ks]),
                     100 * d.mean()))
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(n):
        s = c = 0.0
        for sums, cnts in strata:
            idx = rng.integers(0, len(sums), len(sums))
            s += sums[idx].sum()
            c += cnts[idx].sum()
        means.append(s / c)
    diff = sum(x[4] * x[1] for x in rows) / sum(x[1] for x in rows)
    return rows, diff, tuple(100 * np.percentile(means, [2.5, 97.5]))


def main():
    args = [x for x in sys.argv[1:] if x != "--x1"]
    benches = BENCH[1:] if "--x1" in sys.argv else BENCH
    full = sum(SIZE[b] for b in benches)
    a, others = args[0], args[1:]
    for r in others:
        rows, diff, (lo, hi) = pooled(a, r, benches=benches)
        tot = sum(x[1] for x in rows)
        acc = lambda i: sum(x[i] * x[1] for x in rows) / tot
        print(f"{a} minus {r} (CI: videos resampled within each benchmark)")
        for b, k, x, y, d in rows:
            print(f"  {b:<9} n={k:5d}  {x:5.2f} vs {y:5.2f}  {d:+6.2f}")
        flag = "" if tot == full else f"  INCOMPLETE: {tot} of {full} questions"
        print(f"  pooled    n={tot:5d}  {acc(2):5.2f} vs {acc(3):5.2f}  {diff:+6.2f} [{lo:+.2f}, {hi:+.2f}]{flag}")


if __name__ == "__main__":
    main()
