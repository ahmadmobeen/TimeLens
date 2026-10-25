"""Self-test for T1's data plumbing and evaluation maths. CPU only, no real features.

A wrong label offset or a leaky distractor pool would not crash training; it would
quietly produce a good-looking number. These checks pin the invariants down.

Run: .venv/bin/python selftest_t1.py
"""
from __future__ import annotations

import random

import numpy as np
import torch

from t1_data import Clusters, Query, make_timeline, span_seconds
from t1_train import derangement, dist_to_span, loss_fn, summarize, window_scores

FAILS: list[str] = []


def check(name: str, ok: bool) -> None:
    print(f"  [{'ok' if ok else 'FAIL'}] {name}")
    if not ok:
        FAILS.append(name)


class FakeBank:
    """Duck-typed Bank: video i has T_i seconds; feature row = global second index."""

    def __init__(self, lengths: list[int], emb: torch.Tensor):
        self.vids = [f"v{i}" for i in range(len(lengths))]
        self.T = dict(zip(self.vids, lengths))
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist()
        self.start = dict(zip(self.vids, starts))
        self.row = {v: i for i, v in enumerate(self.vids)}
        self.vid_emb = torch.nn.functional.normalize(emb, dim=-1)
        self.owner = np.concatenate([[i] * n for i, n in enumerate(lengths)])


def main() -> None:
    print("span_seconds")
    check("instant span keeps one second", span_seconds(5.0, 5.0, 30) == (5, 6))
    check("fractional span covers overlapped seconds", span_seconds(4.2, 7.0, 30) == (4, 7))
    check("span past the end is clipped", span_seconds(28.5, 40.0, 30) == (28, 30))

    print("make_timeline")
    rng = np.random.default_rng(0)
    lengths = rng.integers(20, 40, size=300).tolist()
    emb = torch.randn(300, 8)
    bank = FakeBank(lengths, emb)
    allowed = np.ones(300, bool)
    allowed[[3, 7, 11]] = False                          # 3 = queried video, 7/11 excluded
    q = Query("x:0", "v3", 4.2, 9.0)
    ok_len = ok_lab = ok_excl = ok_intact = ok_dup = True
    for mode in ("random", "cluster"):
        cl = Clusters(bank, np.ones(300, bool), per_cluster=30, seed=0)
        for s in range(50):
            tl = make_timeline(q, bank, allowed, 1500, mode, random.Random(s), cl)
            owners = bank.owner[tl.idx.numpy()]
            ok_len &= len(tl.idx) == 1500
            lo, hi = span_seconds(q.a, q.b, bank.T["v3"])
            ok_lab &= (tl.label.nonzero().squeeze(-1).tolist()
                       == list(range(tl.src_lo + lo, tl.src_lo + hi)))
            ok_lab &= bool((owners[tl.label.numpy()] == 3).all())
            ok_excl &= not ({7, 11} & set(owners.tolist()))
            ok_intact &= ((owners == 3).sum() == bank.T["v3"]
                          and (tl.idx[tl.src_lo:tl.src_hi] - bank.start["v3"]).tolist()
                          == list(range(bank.T["v3"])))
            seen = [owners[0]] + [b for a, b in zip(owners[:-1], owners[1:]) if a != b]
            ok_dup &= len(seen) == len(set(seen))
        check(f"[{mode}] exact length", ok_len)
        check(f"[{mode}] labels = the queried span, inside the queried video", ok_lab)
        check(f"[{mode}] excluded videos never appear", ok_excl)
        check(f"[{mode}] queried video appears once, whole and in order", ok_intact)
        check(f"[{mode}] no distractor repeats", ok_dup)
    cl = Clusters(bank, np.ones(300, bool), per_cluster=30, seed=0)
    tl = make_timeline(q, bank, allowed, 600, "cluster", random.Random(0), cl)
    own = cl.assign[3]
    others = [o for o in set(bank.owner[tl.idx.numpy()].tolist()) if o != 3]
    share = np.mean([cl.assign[o] == own for o in others])
    avail = sum(cl.assign[v] == own and allowed[v] for v in range(300))
    check(f"cluster mode draws its own cluster first ({share:.0%} same-cluster, "
          f"{avail} available)", share > 0.9 or len(others) > avail)

    print("window scoring and metrics")
    z = np.full(3600, -5.0)
    z[1800:1806] = 5.0                                   # a perfect detector
    w = window_scores(z, 3600.0, 1803.0)
    g = [x["score"] for x in w if x["cand"] == "gt"][0]
    check("perfect curve: GT window outscores every other window",
          all(g > x["score"] for x in w if x["cand"] != "gt"))
    seg = z[1683:1923]
    check("window score is log-sum-exp", abs(g - np.log(np.exp(seg).sum())) < 1e-6)
    check("argmax distance 0 inside the span", dist_to_span(z, 1800, 1806) == 0.0)
    check("argmax distance measured to the nearest span edge",
          dist_to_span(z, 1900, 1910) == 100.0)
    per = []
    for c in range(40):
        zz = np.random.default_rng(c).normal(size=3600)
        zz[1000:1005] += 50
        per.append({"windows": window_scores(zz, 3600.0, 1002.0), "dist": 0.0})
    s = summarize(per)
    check(f"separable items: AUROC(hard) {s['auroc_hard']:.2f}, paired {s['paired_win']:.2f}",
          s["auroc_hard"] == 1.0 and s["paired_win"] == 1.0)
    per = [{"windows": window_scores(np.random.default_rng(c).normal(size=3600), 3600.0,
                                     1800.0), "dist": 999.0} for c in range(400)]
    s = summarize(per)
    check(f"noise items: AUROC(hard) {s['auroc_hard']:.2f} near 0.5",
          abs(s["auroc_hard"] - 0.5) < 0.1)

    print("derangement and loss")
    p = derangement(50, 0)
    check("no fixed points", all(p[i] != i for i in range(50)) and sorted(p) == list(range(50)))
    grp = [i // 5 for i in range(50)]
    p = derangement(50, 1, group=grp)
    check("placebo never comes from the same group", all(grp[p[i]] != grp[i] for i in range(50)))
    forbid = np.zeros((50, 50), bool)
    forbid[np.arange(50), (np.arange(50) + 1) % 50] = True     # i may not draw i+1
    forbid[:10, :10] = True                                     # nor a paraphrase block
    p = derangement(50, 2, forbid=forbid)
    check("placebo never draws a forbidden (paraphrase) partner",
          not any(forbid[i, p[i]] for i in range(50)) and sorted(p) == list(range(50)))
    lab = torch.zeros(2, 100, dtype=torch.bool)
    lab[:, 40:45] = True
    good = torch.where(lab, 5.0, -5.0)
    check("loss prefers logits peaked on the evidence",
          loss_fn(good, lab) < loss_fn(torch.zeros(2, 100), lab) < loss_fn(-good, lab))

    print(f"\n{'ALL PASS' if not FAILS else f'{len(FAILS)} FAILED: {FAILS}'}")
    raise SystemExit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
