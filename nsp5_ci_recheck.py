"""Recompute every recorded NS-P5 paired CI both ways: per question (as recorded) and clustered
by video (CONVENTIONS section 7), and flag any comparison whose zero-exclusion verdict changes.

    .venv/bin/python nsp5_ci_recheck.py
"""
import collections

import numpy as np

import bench
from nsp5_compare import boot_ci, load, video_of

L = "logs/nsp5/"
SUB = {}
for it in bench.load("videomme"):
    if it.meta["split"] == "long":
        SUB.setdefault("videomme-long", set()).add(it.key)
for it in bench.load("lvb"):
    if str(it.meta["duration_group"]) == "3600":
        SUB.setdefault("lvb-15-60min", set()).add(it.key)

ROWS = [  # label, reference arm, compared arm, subset
    ("LVBench E3: oracle s15 vs uniform", "e3_uniform", "e3_gauss_s15", None),
    ("LVBench B1 vs uniform", "e3_uniform", "b1_topk_siglip", None),
    ("LVBench A1-topk vs B1 (A1 primary)", "b1_topk_siglip", "a1_topk", None),
    ("LVBench A2-kelly vs B1", "b1_topk_siglip", "a2_kelly", None),
    ("LVBench A2-kelly vs uniform", "e3_uniform", "a2_kelly", None),
    ("LVBench A2 placebo vs uniform", "e3_uniform", "a2_kelly_placebo", None),
    ("LVBench A3-kelly vs A2-kelly (A3 primary)", "a2_kelly", "a3_kelly", None),
    ("LVBench A3-topk vs A2-topk", "a2_topk", "a3_topk", None),
    ("LVBench A3-kelly vs uniform", "e3_uniform", "a3_kelly", None),
    ("LVBench A3 placebo vs uniform", "e3_uniform", "a3_kelly_placebo", None),
    ("LVBench A3-kelly vs A3 placebo", "a3_kelly_placebo", "a3_kelly", None),
    ("G1 Qwen3: A2-kelly vs uniform", "g1_q3vl_uniform", "g1_q3vl_a2_kelly", None),
    ("G1 Qwen3: A3-kelly vs A2-kelly", "g1_q3vl_a2_kelly", "g1_q3vl_a3_kelly", None),
    ("G1 Qwen3: A3-topk vs A2-kelly", "g1_q3vl_a2_kelly", "g1_q3vl_a3_topk", None),
    ("G1 Qwen3: A3-kelly vs uniform", "g1_q3vl_uniform", "g1_q3vl_a3_kelly", None),
    ("GE1: A3-kelly vs GE1-stem (primary)", "ge1_stem_topk", "a3_kelly", None),
    ("GE1: GE1-opt vs GE1-stem", "ge1_stem_topk", "ge1_opt_topk", None),
    ("GE1: A3-kelly vs GE1-opt", "ge1_opt_topk", "a3_kelly", None),
    ("G2 Qwen3.5: A3-kelly vs uniform", "g2_q35_uniform", "g2_q35_a3_kelly", None),
    ("G2 Qwen3.5: A3-kelly vs B1", "g2_q35_b1", "g2_q35_a3_kelly", None),
    ("X1 Video-MME long: A3 vs B1 (primary)", "x1_videomme_b1", "x1_videomme_a3_kelly", "videomme-long"),
    ("X1 Video-MME all: A3 vs B1", "x1_videomme_b1", "x1_videomme_a3_kelly", None),
    ("X1 Video-MME long: A3 vs A2", "x1_videomme_a2_kelly", "x1_videomme_a3_kelly", "videomme-long"),
    ("X1 Video-MME long: A3 vs Q8-opt", "x1_videomme_q8opt_topk", "x1_videomme_a3_kelly", "videomme-long"),
    ("X1 Video-MME all: A3 vs Q8-opt", "x1_videomme_q8opt_topk", "x1_videomme_a3_kelly", None),
    ("X1 Video-MME long, Qwen3: A3 vs uniform", "x1_videomme_uniform_q3", "x1_videomme_a3_kelly_q3", "videomme-long"),
    ("X1 LVB all: A3 vs B1 (primary)", "x1_lvb_b1", "x1_lvb_a3_kelly", None),
    ("X1 LVB 15-60 min: A3 vs B1", "x1_lvb_b1", "x1_lvb_a3_kelly", "lvb-15-60min"),
    ("X1 LVB all: A3 vs Q8-opt", "x1_lvb_q8opt_topk", "x1_lvb_a3_kelly", None),
    ("X1 LVB 15-60 min: A3 vs Q8-opt", "x1_lvb_q8opt_topk", "x1_lvb_a3_kelly", "lvb-15-60min"),
    ("X1 LVB all: A3 vs A2", "x1_lvb_a2_kelly", "x1_lvb_a3_kelly", None),
    ("X1 LVB 15-60 min, Qwen3: A3 vs uniform", "x1_lvb_uniform_q3", "x1_lvb_a3_kelly_q3", "lvb-15-60min"),
    ("X1 CG-Bench, Qwen3: A3 vs uniform", "x1_cgbench_uniform_q3", "x1_cgbench_a3_kelly_q3", None),
]


def main() -> None:
    print(f"{'comparison':44s}{'n':>6}{'diff':>8}  {'CI by question':>18}  {'CI by video':>18}  verdict")
    for label, r, a, sub in ROWS:
        ref, arm = load(L + r + ".jsonl"), load(L + a + ".jsonl")
        ks = sorted(k for k in ref if k in arm and (sub is None or k in SUB[sub]))
        d = np.array([float(arm[k]["correct"]) - float(ref[k]["correct"]) for k in ks])
        q = boot_ci(d)
        v = boot_ci(d, groups=[video_of(k) for k in ks])
        ex = lambda ci: ci[0] > 0 or ci[1] < 0
        flag = "CHANGES" if ex(q) != ex(v) else ("excl. 0" if ex(v) else "incl. 0")
        fmt = lambda ci: f"[{100*ci[0]:+.2f}, {100*ci[1]:+.2f}]"
        print(f"{label:44s}{len(ks):>6}{100*d.mean():>+8.2f}  {fmt(q):>18}  {fmt(v):>18}  {flag}")


if __name__ == "__main__":
    main()
