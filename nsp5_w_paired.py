"""Paired differences with their bootstrap draws, shared by Study W's CPU analyses W5, W6 and the
Holm part of W8 (NS-P5, METHODS-LOG 2026-11-12, "Reviewer-requested experiments, wave 1").

boot() is nsp5_pooled.pooled with three additions: it takes any per-question values (correctness,
or an indicator such as "picked option X"), any subset of questions, and it returns every draw.
It consumes the random stream exactly as pooled() does (videos resampled within each benchmark,
5,000 draws, seed 0, benchmarks in BENCH order), so on the same questions its interval is the
interval the paper's tables print. Keys must be unique across benchmarks (they are: LVBench keys
are '<video>>>><uid>', the others '<bench>:<id>').
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from decimal import ROUND_HALF_UP, Decimal

import numpy as np

from nsp5_compare import video_of
from nsp5_pooled import BENCH, path

OURS = "q8_kelly_eps0"             # M2, CaliFrame
UNIFORM = "e3_uniform:uniform"     # LVBench stem e3_uniform, X1 stem uniform
GMM = "f2c_topk_nr"                # GMM-EVA's selection
OUT_DIR = "logs/nsp5/analysis"
NAMES = {"lvbench": "LVBench", "videomme": "Video-MME", "lvb": "LongVideoBench", "mlvu": "MLVU",
         "cgbench": "CG-Bench"}
DRAWS, SEED = 5000, 0


def rows(arm: str, bench: str) -> dict[str, dict]:
    """Every logged row of an arm on a benchmark, by question key."""
    with open(path(arm, bench), encoding="utf-8") as fh:
        return {r["key"]: r for r in (json.loads(line) for line in fh if line.strip())}


def correct(arm: str, benches=BENCH) -> dict[str, float]:
    """Correctness (0/1) of an arm on every question of the given benchmarks, one flat dict."""
    return {k: float(r["correct"]) for b in benches for k, r in rows(arm, b).items()}


def by_bench(keys) -> dict[str, list[str]]:
    """Group question keys by benchmark, in BENCH order (the order boot() draws in)."""
    out: dict[str, list[str]] = {b: [] for b in BENCH}
    for k in keys:
        b = "lvbench" if ">>>" in k else k.split(":", 1)[0]
        out[b].append(k)
    return {b: ks for b, ks in out.items() if ks}


@dataclass(frozen=True)
class Paired:
    """a minus r over a set of questions, in points."""
    n: int
    videos: int
    a: float                 # mean of a over the questions
    r: float
    diff: float              # the pooled (question-weighted) difference, as pooled() computes it
    lo: float                # 95 % percentile interval of the draws
    hi: float
    draws: np.ndarray        # [DRAWS] pooled differences
    per: dict                # bench -> {"n", "a", "r", "diff"}
    equal: float             # benchmark-equal average of the per-benchmark differences
    equal_lo: float
    equal_hi: float

    def text(self) -> str:
        return f"{self.diff:+6.2f} [{self.lo:+6.2f}, {self.hi:+6.2f}]"

    def record(self) -> dict:
        """JSON-ready summary (draws left out)."""
        return {"n": self.n, "videos": self.videos, "a": round(self.a, 3), "r": round(self.r, 3),
                "diff": round(self.diff, 3), "ci": [round(self.lo, 3), round(self.hi, 3)]}


def _strata(va: dict, vr: dict, keys: dict[str, list[str]]) -> tuple[list, dict]:
    strata, per = [], {}
    for b in BENCH:
        ks = sorted(keys.get(b, ()))
        if not ks:
            continue
        d = np.array([va[k] - vr[k] for k in ks])
        _, inv = np.unique([video_of(k) for k in ks], return_inverse=True)
        strata.append((np.bincount(inv, weights=d), np.bincount(inv).astype(float)))
        # kept as np.float64, as pooled() keeps them: Python 3.12's sum() adds plain floats with
        # compensated summation but np.float64 with plain addition, and the pooled difference
        # below must be pooled()'s to the last bit
        per[b] = {"n": len(ks), "a": 100 * np.mean([va[k] for k in ks]),
                  "r": 100 * np.mean([vr[k] for k in ks]), "diff": 100 * d.mean()}
    return strata, per


def boot(va: dict, vr: dict, keys: dict[str, list[str]], n: int = DRAWS, seed: int = SEED) -> Paired:
    """va minus vr over keys (bench -> question keys) with the video bootstrap of pooled()."""
    strata, per = _strata(va, vr, keys)
    rng = np.random.default_rng(seed)
    pooled_draws = np.empty(n)
    S, C = np.empty((n, len(strata))), np.empty((n, len(strata)))
    for i in range(n):
        s = c = 0.0
        for j, (sums, cnts) in enumerate(strata):      # the loop of nsp5_pooled.pooled, verbatim
            idx = rng.integers(0, len(sums), len(sums))
            S[i, j], C[i, j] = sums[idx].sum(), cnts[idx].sum()
            s += S[i, j]
            c += C[i, j]
        pooled_draws[i] = s / c
    tot = sum(x["n"] for x in per.values())
    diff = sum(x["diff"] * x["n"] for x in per.values()) / tot
    lo, hi = (float(x) for x in 100 * np.percentile(pooled_draws, [2.5, 97.5]))
    equal_draws = 100 * (S / C).mean(1)
    e_lo, e_hi = (float(x) for x in np.percentile(equal_draws, [2.5, 97.5]))
    return Paired(n=tot, videos=sum(len(s[0]) for s in strata),
                  a=sum(x["a"] * x["n"] for x in per.values()) / tot,
                  r=sum(x["r"] * x["n"] for x in per.values()) / tot,
                  diff=diff, lo=lo, hi=hi, draws=100 * pooled_draws, per=per,
                  equal=float(np.mean([x["diff"] for x in per.values()])), equal_lo=e_lo, equal_hi=e_hi)


def p_boot(draws: np.ndarray) -> float:
    """Two-sided bootstrap p-value of 'no difference', the inversion of the percentile interval:
    twice the smaller share of draws at or beyond zero, with the (1 + count) / (B + 1) correction
    (Davison and Hinkley 1997, ch. 4). Its floor is 2 / (B + 1), 0.0004 at 5,000 draws."""
    b = len(draws)
    below = (1 + int(np.sum(draws <= 0))) / (b + 1)
    above = (1 + int(np.sum(draws >= 0))) / (b + 1)
    return min(1.0, 2 * min(below, above))


def p_normal(diff: float, draws: np.ndarray) -> float:
    """Two-sided p-value of the normal approximation with the bootstrap standard error."""
    return math.erfc(abs(diff / float(np.std(draws, ddof=1))) / math.sqrt(2))


def holm(p: list[float]) -> list[float]:
    """Holm's step-down adjusted p-values (family-wise error rate), in the input order."""
    m = len(p)
    adj, run = [0.0] * m, 0.0
    for rank, i in enumerate(sorted(range(m), key=lambda j: p[j])):
        run = max(run, min(1.0, (m - rank) * p[i]))
        adj[i] = run
    return adj


def half_up(x: float) -> str:
    """A signed difference rounded half-up to 0.1, as the paper's tables print it."""
    s = str(Decimal(repr(float(x))).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP))
    return "0.0" if Decimal(s) == 0 else (s if s.startswith("-") else "+" + s)


def write_outputs(stem: str, data: dict, lines: list[str]) -> None:
    """logs/nsp5/analysis/<stem>.json (every number) and <stem>.txt (the printed tables)."""
    os.makedirs(OUT_DIR, exist_ok=True)
    with open(os.path.join(OUT_DIR, f"{stem}.json"), "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=1)
    with open(os.path.join(OUT_DIR, f"{stem}.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")
    print("\n".join(lines))
    print(f"-> {OUT_DIR}/{stem}.json, {OUT_DIR}/{stem}.txt")
