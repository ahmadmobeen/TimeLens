"""Offline evaluation of confidence-gated two-look policies on REAL per-question outcomes.

A first-look arm with p_letters logged; one or more fallback arms whose answers are
already known per question. Policy G(theta): keep the first look's answer when its
confidence max(p_letters) >= theta, otherwise take the fallback's answer. theta is
chosen by 5-fold cross-validation BY VIDEO (fit on four folds, applied to the fifth),
so no question is scored with a threshold fitted on itself.

Cost is reported as looks per question: 1 + share of questions that fall back.

  .venv/bin/python nsp5_gate_offline.py --first logs/nsp5/a2_topk.jsonl \\
      --fallback logs/nsp5/e3_uniform.jsonl logs/nsp5/a2_kelly.jsonl
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np

from nsp5_compare import boot_ci, load


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--first", required=True)
    ap.add_argument("--fallback", nargs="+", required=True)
    ap.add_argument("--folds", type=int, default=5)
    args = ap.parse_args()
    F = load(args.first)
    fbs = {os.path.basename(p).replace(".jsonl", ""): load(p) for p in args.fallback}
    keys = sorted(k for k in F if "p_letters" in F[k] and all(k in fb for fb in fbs.values()))
    conf = np.array([max(F[k]["p_letters"]) for k in keys])
    c1 = np.array([F[k]["correct"] for k in keys], float)
    vids = sorted({k.split(">>>")[0] for k in keys})
    random.Random(0).shuffle(vids)
    fold_of = {v: i % args.folds for i, v in enumerate(vids)}
    fold = np.array([fold_of[k.split(">>>")[0]] for k in keys])
    grid = np.quantile(conf, np.linspace(0, 1, 41))
    print(f"{len(keys)} questions with first-look confidence; first look alone "
          f"{c1.mean() * 100:.2f}%\n")
    for name, fb in fbs.items():
        c2 = np.array([fb[k]["correct"] for k in keys], float)
        oracle = np.maximum(c1, c2).mean()
        out, looks = np.zeros(len(keys)), np.zeros(len(keys))
        for f in range(args.folds):
            tr, te = fold != f, fold == f
            accs = [np.where(conf[tr] >= t, c1[tr], c2[tr]).mean() for t in grid]
            t = grid[int(np.argmax(accs))]
            out[te] = np.where(conf[te] >= t, c1[te], c2[te])
            looks[te] = 1 + (conf[te] < t)
        lo, hi = boot_ci(out - c1)
        print(f"fallback {name:<16} alone {c2.mean() * 100:6.2f}%  gated (CV) "
              f"{out.mean() * 100:6.2f}%  vs first {(out - c1).mean() * 100:+.2f} "
              f"[{lo * 100:+.2f}, {hi * 100:+.2f}]  looks/q {looks.mean():.2f}  "
              f"| per-question oracle pick {oracle * 100:.2f}%")
        if all("p_letters" in fb[k] for k in keys):
            # Two looks always taken; combine beliefs instead of switching between them.
            p1 = np.array([F[k]["p_letters"] for k in keys])
            p2 = np.array([fb[k]["p_letters"] for k in keys])
            gold = np.array(["ABCD".index(F[k]["gold_letter"]) for k in keys])
            for label, pred in (("mean of beliefs", (p1 + p2).argmax(1)),
                                ("more confident look", np.where(p1.max(1) >= p2.max(1),
                                                                 p1.argmax(1), p2.argmax(1)))):
                ok = (pred == gold).astype(float)
                lo, hi = boot_ci(ok - c1)
                print(f"    both looks, {label:<20} {ok.mean() * 100:6.2f}%  vs first "
                      f"{(ok - c1).mean() * 100:+.2f} [{lo * 100:+.2f}, {hi * 100:+.2f}]  looks/q 2.00")


if __name__ == "__main__":
    main()
