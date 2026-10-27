"""MLVU M-Avg (mean of per-task accuracies) for two arms, with a task-stratified paired bootstrap.

    .venv/bin/python nsp5_mavg.py logs/nsp5/x1_mlvu_uniform_q3.jsonl logs/nsp5/x1_mlvu_a3_kelly_q3.jsonl
"""
import collections
import json
import sys

import numpy as np

import bench
from nsp5_compare import boot_ci


def load(p):
    return {json.loads(l)["key"]: json.loads(l)["correct"] for l in open(p)}


def main() -> None:
    ref, arm = load(sys.argv[1]), load(sys.argv[2])
    task = {it.key: it.meta.get("task") or it.key.split(":")[1] for it in bench.load("mlvu")}
    by = collections.defaultdict(list)
    for k in ref:
        if k in arm:
            by[task[k]].append((float(ref[k]), float(arm[k])))
    print(f"{sys.argv[2].split('/')[-1]} vs {sys.argv[1].split('/')[-1]}")
    for t, v in sorted(by.items()):
        a = np.array(v)
        print(f"  {t:12s} n={len(v):4d}  ref {100*a[:,0].mean():5.1f}  arm {100*a[:,1].mean():5.1f}  diff {100*(a[:,1]-a[:,0]).mean():+5.1f}")
    arrs = [np.array(v) for v in by.values()]
    mavg = lambda col: np.mean([a[:, col].mean() for a in arrs])
    rng = np.random.default_rng(0)
    boots = [np.mean([(a[idx, 1] - a[idx, 0]).mean() for a in arrs for idx in [rng.integers(0, len(a), len(a))]])
             for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    print(f"  M-Avg      ref {100*mavg(0):5.2f}  arm {100*mavg(1):5.2f}  diff {100*(mavg(1)-mavg(0)):+5.2f} [{100*lo:+.2f}, {100*hi:+.2f}]")
    d = np.concatenate([a[:, 1] - a[:, 0] for a in arrs]); blo, bhi = boot_ci(d)
    print(f"  plain acc  diff {100*d.mean():+5.2f} [{100*blo:+.2f}, {100*bhi:+.2f}]  (n={len(d)})")


if __name__ == "__main__":
    main()
