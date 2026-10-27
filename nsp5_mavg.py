"""MLVU M-Avg (mean of per-task accuracies) for two arms, with a paired bootstrap that resamples
videos within each task (stratified by task, clustered by video: CONVENTIONS section 7).

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
    items = {it.key: it for it in bench.load("mlvu")}
    by = collections.defaultdict(list)   # task -> [(ref, arm, video)]
    for k in ref:
        if k in arm:
            by[items[k].meta.get("task") or k.split(":")[1]].append((float(ref[k]), float(arm[k]), items[k].vid))
    print(f"{sys.argv[2].split('/')[-1]} vs {sys.argv[1].split('/')[-1]} (CI: by video within task)")
    rng = np.random.default_rng(0)
    task_boot, rows = [], []
    for t, v in sorted(by.items()):
        r = np.array([x[0] for x in v]); a = np.array([x[1] for x in v])
        print(f"  {t:18s} n={len(v):4d}  ref {100*r.mean():5.1f}  arm {100*a.mean():5.1f}  diff {100*(a-r).mean():+5.1f}")
        _, inv = np.unique([x[2] for x in v], return_inverse=True)
        sums, cnts = np.bincount(inv, weights=a - r), np.bincount(inv).astype(float)
        idx = rng.integers(0, len(sums), size=(5000, len(sums)))
        task_boot.append(sums[idx].sum(1) / cnts[idx].sum(1))
        rows.append((r.mean(), a.mean()))
    boots = np.mean(task_boot, axis=0)
    lo, hi = np.percentile(boots, [2.5, 97.5])
    mr, ma = np.mean([x[0] for x in rows]), np.mean([x[1] for x in rows])
    print(f"  M-Avg              ref {100*mr:5.2f}  arm {100*ma:5.2f}  diff {100*(ma-mr):+5.2f} [{100*lo:+.2f}, {100*hi:+.2f}]")
    d = np.array([x[1] - x[0] for v in by.values() for x in v]); g = [x[2] for v in by.values() for x in v]
    blo, bhi = boot_ci(d, groups=g)
    print(f"  plain acc          diff {100*d.mean():+5.2f} [{100*blo:+.2f}, {100*bhi:+.2f}]  (n={len(d)})")


if __name__ == "__main__":
    main()
