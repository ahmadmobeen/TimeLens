"""Compare LVBench arms question by question (NS-P5).

Every arm is a JSONL from eval_lvbench_foveate.py. Differences are PAIRED: the same
question under two arms, so an arm is not credited for the luck of which questions it got
right. The 95 % bootstrap CI resamples whole VIDEOS (CONVENTIONS section 7): one video
contributes several questions, and resampling questions alone understates the interval
(LVBench: ~15 questions per video). --cluster question gives the per-question interval
NS-P5 reported before 2026-10-27.

  .venv/bin/python nsp5_compare.py --ref logs/nsp5/b1_topk_siglip.jsonl \\
      logs/nsp5/e3_uniform.jsonl logs/nsp5/a1_topk.jsonl ...

Breakdowns: by evidence-span length, by LVBench question type, and -- where the arm
logged it -- by frames placed on the evidence (in_span).
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

LVB = ("/home/mobeen/codes/active_repos/NeedleScan-vtg-landscape/research/TimeLens/"
       "data/lvbench/data/train-00000-of-00001.parquet")


def load(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as fh:
        return {r["key"]: r for r in map(json.loads, fh) if r}


_VID: dict[str, str] = {}


def video_of(key: str) -> str:
    """The video a question belongs to. LVBench keys are '<video>>>><uid>'; bench.py keys are
    '<bench>:<id>' and are looked up in bench.load(<bench>)."""
    if ">>>" in key:
        return key.split(">>>")[0]
    if key not in _VID:
        import bench
        _VID.update({it.key: it.vid for it in bench.load(key.split(":", 1)[0])})
    return _VID[key]


def boot_ci(diff: np.ndarray, n: int = 5000, seed: int = 0,
            groups=None) -> tuple[float, float]:
    """95 % percentile interval of the mean paired difference. With `groups` (one label per
    question, e.g. its video) whole groups are resampled, the cluster bootstrap of CONVENTIONS
    section 7; without, single questions are."""
    rng = np.random.default_rng(seed)
    diff = np.asarray(diff, float)
    if groups is None:
        means = rng.choice(diff, size=(n, len(diff)), replace=True).mean(1)
    else:
        _, inv = np.unique(np.asarray(groups), return_inverse=True)
        sums, cnts = np.bincount(inv, weights=diff), np.bincount(inv).astype(float)
        idx = rng.integers(0, len(sums), size=(n, len(sums)))
        means = sums[idx].sum(1) / cnts[idx].sum(1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ref", required=True, help="reference arm for paired differences")
    ap.add_argument("arms", nargs="+")
    ap.add_argument("--cluster", choices=["video", "question"], default="video",
                    help="bootstrap unit for the CI (CONVENTIONS section 7: video)")
    args = ap.parse_args()

    import pandas as pd
    df = pd.read_parquet(LVB)
    qtype = {f"{r.key}>>>{r.uid}": "/".join(r.question_type) for r in df.itertuples()}
    ref = load(args.ref)
    arms = {os.path.basename(p).replace(".jsonl", ""): load(p) for p in args.arms}
    common = set(ref)
    for a in arms.values():
        common &= set(a)
    keys = sorted(common)
    groups = [video_of(k) for k in keys] if args.cluster == "video" else None
    print(f"{len(keys)} questions common to all arms and the reference "
          f"({os.path.basename(args.ref)}); CI: paired bootstrap by {args.cluster}\n")

    rc = np.array([ref[k]["correct"] for k in keys], float)
    print(f"{'arm':<22}{'acc':>8}{'vs ref':>9}{'95% CI':>18}{'win/loss':>11}"
          f"{'in_span':>9}{'n=0':>7}")
    print(f"{'(ref) ' + os.path.basename(args.ref)[:15]:<22}{rc.mean() * 100:>8.2f}")
    for name, a in arms.items():
        c = np.array([a[k]["correct"] for k in keys], float)
        d = c - rc
        lo, hi = boot_ci(d, groups=groups)
        ins = [a[k].get("in_span") for k in keys]
        has = all(x is not None for x in ins)
        ins_s = f"{np.mean(ins):>9.1f}{np.mean(np.array(ins) == 0) * 100:>6.0f}%" if has else ""
        print(f"{name:<22}{c.mean() * 100:>8.2f}{d.mean() * 100:>+9.2f}"
              f"{f'[{lo * 100:+.2f}, {hi * 100:+.2f}]':>18}"
              f"{f'{int((d > 0).sum())}/{int((d < 0).sum())}':>11}{ins_s}")

    span = np.array([ref[k]["span"][1] - ref[k]["span"][0] for k in keys])
    bins = [("instant", span == 0), ("1-30 s", (span > 0) & (span <= 30)),
            ("31-300 s", (span > 30) & (span <= 300)), (">300 s", span > 300)]
    print(f"\nby evidence-span length (accuracy %; n per bin)")
    print(f"{'bin':<12}{'n':>6}{'ref':>8}" + "".join(f"{n[:12]:>14}" for n in arms))
    for label, m in bins:
        row = f"{label:<12}{m.sum():>6}{rc[m].mean() * 100:>8.1f}"
        for a in arms.values():
            row += f"{np.mean([a[k]['correct'] for k, mm in zip(keys, m) if mm]) * 100:>14.1f}"
        print(row)

    types = sorted({t for k in keys for t in qtype.get(k, "").split("/") if t})
    print(f"\nby question type (a question can carry several types)")
    print(f"{'type':<28}{'n':>6}{'ref':>8}" + "".join(f"{n[:12]:>14}" for n in arms))
    for t in types:
        m = np.array([t in qtype.get(k, "").split("/") for k in keys])
        row = f"{t[:27]:<28}{m.sum():>6}{rc[m].mean() * 100:>8.1f}"
        for a in arms.values():
            row += f"{np.mean([a[k]['correct'] for k, mm in zip(keys, m) if mm]) * 100:>14.1f}"
        print(row)


if __name__ == "__main__":
    main()
