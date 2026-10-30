"""Study B (METHODS-LOG 2026-10-30, pre-registered): choose the hedge per question.

M's own calibrated probability over the video's seconds, p = softmax(z / 3.58) with z the 8B
option curve cut to the video's duration as the harness cuts it, gives each question a spread
h = H(p) / log T: 0 when all mass sits on one second, 1 when p is uniform. The rule places frames
by top-k when h <= h* (a concentrated probability) and by M's Kelly placement otherwise. Both
placements were run question by question on all five benchmarks, so the rule's answer to a
question is the logged answer of the placement it picks.

h* is fitted on LVBench alone, over its deciles of h plus the two fixed rules (always Kelly,
always top-k): the highest LVBench accuracy wins, and a tie goes to the lower h* (toward M).
The rule is then frozen and tested on the four X1 benchmarks pooled (nsp5_pooled.py --x1).

    .venv/bin/python nsp5_switch.py

writes logs/nsp5/q8_switch.jsonl and logs/nsp5/x1_<bench>_q8_switch.jsonl: each row is the chosen
placement's row with "arm", "routed" and "h" set, so the registered tools score the rule as an
arm.
"""
import json

import numpy as np

from nsp5_pooled import BENCH, pooled

L, P = "logs/nsp5", "data/t1/priors/qwen3vl_emb_8b"
TAU = 3.58
KELLY = {b: f"{L}/q8_kelly.jsonl" if b == "lvbench" else f"{L}/x1_{b}_q8_kelly.jsonl" for b in BENCH}
TOPK = {b: f"{L}/ge1_opt_topk.jsonl" if b == "lvbench" else f"{L}/x1_{b}_q8opt_topk.jsonl"
        for b in BENCH}


def load(p):
    return {r["key"]: r for r in map(json.loads, open(p, encoding="utf-8")) if r}


def spread(z, dur):
    """H(p) / log T for p = softmax(z / TAU) over the video's seconds (0 for a one-second video)."""
    z = np.asarray(z, dtype=np.float64)[:max(1, int(np.ceil(dur)))]
    if len(z) < 2:
        return 0.0
    p = np.exp((z - z.max()) / TAU)
    p /= p.sum()
    p = p[p > 0]
    return float(-(p * np.log(p)).sum() / np.log(len(z)))


def features(b):
    curves = np.load(f"{P}/{'' if b == 'lvbench' else b + '_'}curves_maxopt.npz")
    pre = "lvbench:" if b == "lvbench" else ""
    kelly, topk = load(KELLY[b]), load(TOPK[b])
    assert set(kelly) == set(topk), f"{b}: the two arms answered different questions"
    h = {k: spread(curves[pre + k], r["duration_s"]) for k, r in kelly.items()}
    return kelly, topk, h


def fit(kelly, topk, h):
    """The threshold with the highest accuracy here, over the deciles of h and the two fixed rules
    (-inf: always Kelly; inf: always top-k); a tie goes to the lower threshold."""
    keys = sorted(h)
    hv = np.array([h[k] for k in keys])
    ck = np.array([float(kelly[k]["correct"]) for k in keys])
    ct = np.array([float(topk[k]["correct"]) for k in keys])
    grid = [-np.inf] + list(np.quantile(hv, np.arange(1, 10) / 10)) + [np.inf]
    table = [(t, 100 * np.where(hv <= t, ct, ck).mean(), 100 * np.mean(hv <= t)) for t in grid]
    return max(table, key=lambda x: (x[1], -x[0]))[0], table


def write(b, kelly, topk, h, t):
    out = f"{L}/q8_switch.jsonl" if b == "lvbench" else f"{L}/x1_{b}_q8_switch.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        for k in sorted(h):
            routed = "topk" if h[k] <= t else "kelly"
            row = dict((topk if routed == "topk" else kelly)[k], arm="switch", routed=routed,
                       h=round(h[k], 6))
            fh.write(json.dumps(row) + "\n")
    return 100 * np.mean([v <= t for v in h.values()])


def main():
    data = {b: features(b) for b in BENCH}
    t, table = fit(*data["lvbench"])
    print("LVBench fit (h*, accuracy %, share routed to top-k %):")
    for x, acc, share in table:
        print(f"  {x:8.4f}  {acc:6.2f}  {share:5.1f}{'  <- chosen' if x == t else ''}")
    for b in BENCH:
        print(f"{b:<9} routed to top-k: {write(b, *data[b], t):5.1f}%")
    print("Test, the four X1 benchmarks pooled (LVBench was the fit):")
    for r in ("q8_kelly", "ge1_opt_topk:q8opt_topk"):
        rows, diff, (lo, hi) = pooled("q8_switch", r, benches=BENCH[1:])
        print(f"  switch minus {r}: {diff:+.2f} [{lo:+.2f}, {hi:+.2f}]  n={sum(x[1] for x in rows)}")


if __name__ == "__main__":
    main()
