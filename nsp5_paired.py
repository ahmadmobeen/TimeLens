"""Paired accuracy differences with nsp5_compare's bootstrap, overall and per Video-MME duration split.

    .venv/bin/python nsp5_paired.py videomme logs/nsp5/x1_videomme_uniform_q3.jsonl logs/nsp5/x1_videomme_a3_kelly_q3.jsonl
"""
import collections, json, sys
import numpy as np
import bench
from nsp5_compare import boot_ci, video_of

def load(p):
    return {json.loads(l)["key"]: json.loads(l)["correct"] for l in open(p)}

def report(ref_path, arm_path, keys=None, label=""):
    ref, arm = load(ref_path), load(arm_path)
    ks = [k for k in ref if k in arm and (keys is None or k in keys)]
    r = np.array([ref[k] for k in ks], float); a = np.array([arm[k] for k in ks], float)
    d = a - r; lo, hi = boot_ci(d, groups=[video_of(k) for k in ks])
    print(f"  {label:10s} n={len(ks):5d}  ref {100*r.mean():5.2f}  arm {100*a.mean():5.2f}  diff {100*d.mean():+5.2f} [{100*lo:+.2f}, {100*hi:+.2f}]")

if __name__ == "__main__":
    bench_name, ref_path, arm_path = sys.argv[1:4]
    print(f"{arm_path.split('/')[-1]} vs {ref_path.split('/')[-1]}")
    report(ref_path, arm_path, label="overall")
    if bench_name == "videomme":
        split = collections.defaultdict(set)
        for it in bench.load("videomme"):
            split[it.meta["split"]].add(it.key)
        print("  splits:", {k: len(v) for k, v in split.items()})
        for s in sorted(split):
            report(ref_path, arm_path, split[s], label=s)
