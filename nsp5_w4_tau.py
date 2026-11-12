"""NS-P5 Study W, item W4: how stable the calibrated temperature is (CPU only, no answer model).

Pre-registered in METHODS-LOG 2026-11-12 ("Reviewer-requested experiments, wave 1", Study W): "W4
temperature stability: video bootstrap of the split fit (2,000 draws, continuous optimum); learning
curve over 5, 10, 20, 40 and 101 calibration videos (500 draws each) with the loss in NLL and in S on
LVBench and CG-Bench; clue intervals widened or narrowed by -2 to +15 s; each benchmark's accuracy at
0.036 against the best of the seven temperatures."

Data and loss are the paper's (tools/calibration_checks.py): the 8B per-option maximum curves, stored
as 100 x cosine (so tau 3.58 here is 0.0358 on cosines), evidence seconds from t1_data.span_seconds,
and the loss -log p(E) with p = softmax(z / tau). A fit is the continuous minimizer of the mean loss
over the questions of a draw: the best point of an 801-point grid in log tau (0.1 to 100), refined
by the parabola through it and its two neighbours; on the whole split it is checked against a
golden-section search on the exact loss. S = mean log(1 + N p(E)) at N = 64, as in s_law.py.

  W4a  2,000 draws of the 101 calibration videos with replacement (seed 0), refit in each.
  W4b  learning curve: k = 5, 10, 20, 40 videos drawn without replacement and k = 101 with
       replacement (without, it is the split itself), 500 draws each (seed 0). Each draw's fit is
       scored on LVBench and CG-Bench by the mean loss of their evidence (NLL) and by S, relative
       to tau 3.58, and through each benchmark's 64-frame S line (logs/nsp5/s_law.json).
  W4c  every clue interval [a, b] of the split moved to [a - d, b + d], d in -2..+15 s (an interval
       that a negative d would invert shrinks to its midpoint); refit by the continuous optimum
       (2,000-draw video bootstrap) and by the paper's five-fold procedure (tau_fit).
  W4d  each benchmark's accuracy at tau 3.58 against the best of the seven temperatures run
       (q8_kelly_eps0_tau*): paired differences, videos resampled within each benchmark (5,000
       draws, seed 0, nsp5_pooled.pooled), and pooled over the five benchmarks.

    cd research/TimeLens && .venv/bin/python nsp5_w4_tau.py | tee logs/nsp5/analysis/w4_tau.log

Writes logs/nsp5/analysis/w4_tau.json and caches the per-question log p(E) tables in
logs/nsp5/analysis/w_tau_tables.npz (nsp5_w7_objective.py reuses them).
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")      # BLAS threads of the bootstrap products (CPU is shared)

import numpy as np  # noqa: E402

TL = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(TL, "..", "..", "docs", "research", "projects",
                                     "NS-P5-adaptive-sampling", "paper", "cvpr2027", "tools"))
sys.path.insert(0, TL)
sys.path.insert(0, TOOLS)
import calibration_checks as cc  # noqa: E402  (chdirs to research/TimeLens)
import bench  # noqa: E402
from nsp5_compare import video_of  # noqa: E402
from nsp5_method_figures import tau_fit  # noqa: E402
from nsp5_pooled import BENCH, SIZE, load, path, pooled  # noqa: E402
from t1_data import span_seconds  # noqa: E402

OUT_DIR = os.path.join("logs", "nsp5", "analysis")
OUT = os.path.join(OUT_DIR, "w4_tau.json")
CACHE = os.path.join(OUT_DIR, "w_tau_tables.npz")
S_LAW = os.path.join("logs", "nsp5", "s_law.json")
TAU = cc.TAU                                   # 3.58 on the stored scale, 0.0358 on cosines
N = 64
GRID = np.exp(np.linspace(np.log(0.1), np.log(100.0), 801))
LOG_GRID = np.log(GRID)
ARMS = {1.0: "q8_kelly_eps0_tau100", 1.79: "q8_kelly_eps0_tau179", 2.5: "q8_kelly_eps0_tau250",
        3.58: "q8_kelly_eps0", 5.0: "q8_kelly_eps0_tau500", 7.16: "q8_kelly_eps0_tau716",
        10.0: "q8_kelly_eps0_tau1000"}
M2 = ARMS[3.58]
REF = np.array(sorted(ARMS))                   # exact columns at the seven run temperatures
I_TAU = int(np.flatnonzero(REF == TAU)[0])
DELTAS = (0, -2, -1, 1, 2, 5, 10, 15)          # seconds added at each end of every clue interval
SIZES = (5, 10, 20, 40, 101)
PROCS = 8
CHUNK = 4_000_000                              # elements per exp block

_JOB: dict = {}


# ---------------------------------------------------------------------------------------- data
def cal_masks(z: list, deltas: tuple) -> list:
    """Per question of the calibration split (calsplit's order and filter), one evidence mask per
    delta, every clue interval [a, b] moved to [a - d, b + d]; an inverted one shrinks to its middle."""
    curves = np.load(os.path.join(cc.PRIOR, "cgbench_cal_curves_maxopt.npz"))
    names = set(curves.files)
    items = [it for it in bench.load("cgbench_cal") if it.key in names and it.spans]
    assert len(items) == len(z), (len(items), len(z))
    out = []
    for it, c in zip(items, z):
        row = []
        for d in deltas:
            m = np.zeros(len(c), bool)
            for a, b in it.spans:
                lo_t, hi_t = a - d, b + d
                if lo_t > hi_t:
                    lo_t = hi_t = (a + b) / 2
                lo, hi = span_seconds(lo_t, hi_t, len(c))
                m[lo:hi] = True
            row.append(m)
        out.append(row)
    return out


def load_sets() -> dict:
    """name -> (curves, mask variants per question, video per question); 'cal' has one variant per
    delta in DELTAS (the first, delta 0, is calsplit's own mask), the benchmarks one."""
    z, m, v = cc.calsplit()
    masks = cal_masks(z, DELTAS)
    assert all(np.array_equal(a[0], b) for a, b in zip(masks, m)), "delta 0 differs from calsplit"
    sets = {"cal": (z, masks, v)}
    for name, fn in (("lvbench", cc.lvbench), ("cgbench", cc.cgbench)):
        zb, mb, vb = fn()
        sets[name] = (zb, [[x] for x in mb], vb)
    return sets


def fingerprint(sets: dict) -> np.ndarray:
    return np.array([[len(z), sum(float(c.sum()) for c in z), sum(int(x.sum()) for r in ms for x in r)]
                     for z, ms, _ in sets.values()])


# ------------------------------------------------------------------------------- log p(E) tables
def lse(z: np.ndarray, taus: np.ndarray) -> np.ndarray:
    """log sum_t exp(z_t / tau) for every tau."""
    zmax = z.max()
    d = z - zmax
    out = np.empty(len(taus))
    step = max(1, CHUNK // len(z))
    for i in range(0, len(taus), step):
        t = taus[i:i + step]
        out[i:i + step] = zmax / t + np.log(np.exp(d[None, :] / t[:, None]).sum(1))
    return out


def _chunk(bounds: tuple) -> tuple:
    lo, hi = bounds
    z, masks, taus = _JOB["z"], _JOB["masks"], _JOB["taus"]
    out = np.empty((hi - lo, len(masks[lo]), len(taus)))
    for i in range(lo, hi):
        a = lse(z[i], taus)
        for j, m in enumerate(masks[i]):
            out[i - lo, j] = lse(z[i][m], taus) - a
    return lo, out


def logpe_table(z: list, masks: list, taus: np.ndarray) -> np.ndarray:
    """[questions, mask variants, taus]: log of the softmax(z / tau) mass on each mask (exact)."""
    _JOB.update(z=z, masks=masks, taus=np.asarray(taus, float))
    bounds = [(i, min(len(z), i + 16)) for i in range(0, len(z), 16)]
    out = np.empty((len(z), len(masks[0]), len(taus)))
    with mp.get_context("fork").Pool(PROCS) as pool:     # forked: workers read _JOB
        for lo, arr in pool.imap_unordered(_chunk, bounds):
            out[lo:lo + len(arr)] = arr
    return out


def tables(sets: dict) -> dict:
    """Per set, log p(E) on GRID ([questions, variants, 801]) and at REF ('<set>_ref'); cached."""
    fp = fingerprint(sets)
    if os.path.exists(CACHE):
        c = np.load(CACHE)
        if (np.array_equal(c["fingerprint"], fp) and np.array_equal(c["grid"], GRID)
                and np.array_equal(c["ref"], REF)):
            return {k: c[k] for k in c.files}
    out = {"fingerprint": fp, "grid": GRID, "ref": REF}
    for name, (z, masks, _) in sets.items():
        t0 = time.time()
        t = logpe_table(z, masks, np.concatenate([GRID, REF]))
        out[name], out[name + "_ref"] = t[..., :len(GRID)], t[..., len(GRID):]
        print(f"  table {name}: {t.shape} in {time.time() - t0:.0f} s", flush=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    np.savez(CACHE, **out)
    return out


def s_of(logpe: np.ndarray, n: int = N) -> np.ndarray:
    """log(1 + n p(E)) from log p(E)."""
    return np.log1p(n * np.exp(logpe))


# ------------------------------------------------------------------------- fits and resampling
def optimum(curves: np.ndarray, log_grid: np.ndarray = LOG_GRID, maximize: bool = False) -> tuple:
    """Continuous optimum of each row over an even grid in log tau: the best grid point refined by
    the vertex of the parabola through it and its neighbours. Returns tau, value, at_grid_edge."""
    c = np.atleast_2d(np.asarray(curves, float))
    s = -c if maximize else c
    j = s.argmin(1)
    edge = (j == 0) | (j == c.shape[1] - 1)
    jj = np.clip(j, 1, c.shape[1] - 2)
    r = np.arange(len(c))
    f0, f1, f2 = s[r, jj - 1], s[r, jj], s[r, jj + 1]
    den = f0 - 2 * f1 + f2
    safe = np.where(den > 0, den, 1.0)
    h = log_grid[1] - log_grid[0]
    x = np.where(den > 0, 0.5 * h * (f0 - f2) / safe, 0.0)
    v = np.where(den > 0, f1 - (f2 - f0) ** 2 / (8 * safe), f1)
    theta = np.where(edge, log_grid[j], log_grid[jj] + x)
    val = np.where(edge, s[r, j], v)
    return np.exp(theta), (-val if maximize else val), edge


def at(curves: np.ndarray, taus: np.ndarray, log_grid: np.ndarray = LOG_GRID) -> np.ndarray:
    """Row d of curves (over log_grid) linearly interpolated at taus[d]."""
    c = np.atleast_2d(curves)
    h = log_grid[1] - log_grid[0]
    x = (np.log(np.broadcast_to(taus, (len(c),))) - log_grid[0]) / h
    i = np.clip(np.floor(x).astype(int), 0, len(log_grid) - 2)
    f = np.clip(x - i, 0.0, 1.0)
    r = np.arange(len(c))
    return c[r, i] * (1 - f) + c[r, i + 1] * f


def golden(f, lo: float, hi: float, iters: int = 40) -> float:
    """Golden-section minimum of f on [lo, hi]."""
    g = (np.sqrt(5.0) - 1) / 2
    a, b = lo, hi
    c, d = b - g * (b - a), a + g * (b - a)
    fc, fd = f(c), f(d)
    for _ in range(iters):
        if fc < fd:
            b, d, fd = d, c, fc
            c = b - g * (b - a)
            fc = f(c)
        else:
            a, c, fc = c, d, fd
            d = a + g * (b - a)
            fd = f(d)
    return (a + b) / 2


def draw_weights(vids: list, draws: int, seed: int = 0, size: int | None = None,
                 replace: bool = True) -> np.ndarray:
    """[draws, questions]: how often each question's video is drawn (videos as the unit)."""
    u, inv = np.unique(np.asarray(vids), return_inverse=True)
    rng = np.random.default_rng(seed)
    k = len(u) if size is None else size
    w = np.empty((draws, len(vids)))
    for d in range(draws):
        pick = rng.integers(0, len(u), k) if replace else rng.choice(len(u), k, replace=False)
        w[d] = np.bincount(pick, minlength=len(u))[inv]
    return w


def wmean(w: np.ndarray, table: np.ndarray) -> np.ndarray:
    """Weighted mean over questions of a [questions, ...] table for every weight row."""
    return (w @ table.reshape(len(table), -1) / w.sum(1, keepdims=True)).reshape(
        (len(w),) + table.shape[1:])


def summary(x: np.ndarray, nd: int = 4) -> dict:
    x = np.asarray(x, float)
    return {"mean": round(float(x.mean()), nd), "median": round(float(np.median(x)), nd),
            "lo": round(float(np.percentile(x, 2.5)), nd), "hi": round(float(np.percentile(x, 97.5)), nd)}


def s_lines() -> dict:
    """Each benchmark's 64-frame S line (accuracy = a + b S) as s_law.py last fitted it."""
    with open(S_LAW, encoding="utf-8") as fh:
        r = json.load(fh)
    out = {}
    for b in ("lvbench", "cgbench"):
        f = r[b]["fit"]
        s_m2 = [v["S"] for v in r[b]["variants"] if v["arm"] == M2][0]
        out[b] = {"intercept": f["intercept"], "slope": f["slope"], "n": f["n"], "r2": f["r2"],
                  "S_at_tau_s_law": s_m2}
    return out


# --------------------------------------------------------------------------------------- items
def bench_curves(tab: dict, lines: dict) -> dict:
    """Mean NLL and S (N = 64) of LVBench and CG-Bench over the grid, their values at 3.58, and
    each benchmark's own (in-sample) optima, for scoring fitted temperatures."""
    out = {}
    for b in ("lvbench", "cgbench"):
        lp, ref = tab[b][:, 0, :], tab[b + "_ref"][:, 0, I_TAU]
        nll_c, s_c = (-lp).mean(0), s_of(lp).mean(0)
        nll_ref, s_ref = float((-ref).mean()), float(s_of(ref).mean())
        assert abs(s_ref - lines[b]["S_at_tau_s_law"]) < 1e-6, (b, s_ref, lines[b]["S_at_tau_s_law"])
        t_nll, v_nll, _ = optimum(nll_c)
        t_s, v_s, _ = optimum(s_c, maximize=True)
        out[b] = {"nll": nll_c, "S": s_c, "nll_ref": nll_ref, "S_ref": s_ref,
                  "own_nll_opt": {"tau": float(t_nll[0]), "nll": float(v_nll[0]),
                                  "regret_of_3.58": nll_ref - float(v_nll[0])},
                  "own_S_opt": {"tau": float(t_s[0]), "S": float(v_s[0]),
                                "regret_of_3.58": float(v_s[0]) - s_ref,
                                "points": lines[b]["slope"] * (float(v_s[0]) - s_ref)}}
    return out


def score(taus: np.ndarray, curves: dict, lines: dict) -> dict:
    """NLL and S of LVBench and CG-Bench at fitted temperatures, relative to 3.58."""
    out = {}
    for b, c in curves.items():
        lt = np.log(taus)
        nll, s = np.interp(lt, LOG_GRID, c["nll"]), np.interp(lt, LOG_GRID, c["S"])
        out[b] = {"nll": summary(nll), "nll_minus_at_3.58": summary(nll - c["nll_ref"]),
                  "S": summary(s), "S_minus_at_3.58": summary(s - c["S_ref"]),
                  "points_through_S_line": summary(lines[b]["slope"] * (s - c["S_ref"]), 3)}
    return out


def w4a(tab: dict, sets: dict, curves: dict, lines: dict) -> dict:
    z, masks, vids = sets["cal"]
    loss = -tab["cal"][:, 0, :]
    t_full, v_full, _ = optimum(loss.mean(0))
    exact = np.exp(golden(lambda th: float(np.mean([cc.nll(c, m[0], np.exp(th))
                                                    for c, m in zip(z, masks)])),
                          np.log(t_full[0]) - 0.05, np.log(t_full[0]) + 0.05))
    w = draw_weights(vids, 2000, seed=0)
    taus, _, edge = optimum(wmean(w, loss))
    return {"draws": 2000, "videos": len(set(vids)), "questions": len(vids),
            "tau_full_split": round(float(t_full[0]), 4), "tau_full_split_exact": round(float(exact), 4),
            "nll_full_split": round(float(v_full[0]), 4),
            "nll_at_3.58": round(float((-tab["cal_ref"][:, 0, I_TAU]).mean()), 4),
            "tau": summary(taus), "tau_cosine": summary(taus / 100, 5),
            "share_within_10pct_of_3.58": round(float(np.mean(np.abs(np.log(taus / TAU)) <= np.log(1.1))), 4),
            "share_at_grid_edge": float(edge.mean()), "scored": score(taus, curves, lines)}


def w4b(tab: dict, sets: dict, curves: dict, lines: dict) -> list:
    _, _, vids = sets["cal"]
    loss = -tab["cal"][:, 0, :]
    rows = []
    for k in SIZES:
        replace = k >= len(set(vids))
        w = draw_weights(vids, 500, seed=0, size=k, replace=replace)
        taus, _, edge = optimum(wmean(w, loss))
        rows.append({"videos": k, "draws": 500,
                     "sampling": "with replacement" if replace else "without replacement",
                     "questions_per_draw": summary(w.sum(1), 1), "tau": summary(taus),
                     "share_at_grid_edge": float(edge.mean()), "scored": score(taus, curves, lines)})
    return rows


def w4c(tab: dict, sets: dict, curves: dict, lines: dict) -> list:
    z, masks, vids = sets["cal"]
    w = draw_weights(vids, 2000, seed=0)
    rows = []
    for j, d in enumerate(DELTAS):
        loss = -tab["cal"][:, j, :]
        t_full, v_full, _ = optimum(loss.mean(0))
        taus, _, _ = optimum(wmean(w, loss))
        paper = tau_fit(z, [m[j] for m in masks], vids)
        sc = score(t_full, curves, lines)
        rows.append({"delta_s": d, "evidence_seconds_mean": round(float(np.mean([m[j].sum() for m in masks])), 2),
                     "tau_continuous": round(float(t_full[0]), 4), "tau_boot": summary(taus),
                     "nll_at_fit": round(float(v_full[0]), 4),
                     "tau_paper_procedure": paper["tau"], "fold_tau": paper["fold_tau"],
                     "heldout_nll": paper["heldout_nll"],
                     "lvbench_points_through_S_line": sc["lvbench"]["points_through_S_line"]["mean"],
                     "cgbench_points_through_S_line": sc["cgbench"]["points_through_S_line"]["mean"],
                     "lvbench_S_minus_at_3.58": sc["lvbench"]["S_minus_at_3.58"]["mean"],
                     "cgbench_S_minus_at_3.58": sc["cgbench"]["S_minus_at_3.58"]["mean"]})
    return rows


def accuracy(arm: str, b: str) -> float:
    rows = load(path(arm, b))
    assert len(rows) == SIZE[b], (arm, b, len(rows))
    return 100.0 * float(np.mean(list(rows.values())))


def oracle_pooled(best: dict, draws: int = 5000, seed: int = 0) -> tuple:
    """Pooled gain if each benchmark used its own best temperature: paired, videos resampled within
    each benchmark (nsp5_pooled.pooled's bootstrap with a different arm per benchmark)."""
    strata, n, tot = [], 0, 0.0
    for b in BENCH:
        a, r = load(path(ARMS[best[b]], b)), load(path(M2, b))
        ks = sorted(set(a) & set(r))
        d = np.array([a[k] - r[k] for k in ks])
        _, inv = np.unique([video_of(k) for k in ks], return_inverse=True)
        strata.append((np.bincount(inv, weights=d), np.bincount(inv).astype(float)))
        n, tot = n + len(ks), tot + d.sum()
    rng = np.random.default_rng(seed)
    means = []
    for _ in range(draws):
        s = c = 0.0
        for sums, cnts in strata:
            idx = rng.integers(0, len(sums), len(sums))
            s, c = s + sums[idx].sum(), c + cnts[idx].sum()
        means.append(s / c)
    return 100.0 * tot / n, tuple(100.0 * np.percentile(means, [2.5, 97.5]))


def w4d() -> dict:
    per, best = {}, {}
    for b in BENCH:
        acc = {t: accuracy(a, b) for t, a in ARMS.items()}
        best[b] = max(acc, key=acc.get)
        rank = 1 + sum(v > acc[TAU] for v in acc.values())
        entry = {"acc": {str(t): round(v, 2) for t, v in acc.items()}, "best_tau": best[b],
                 "rank_of_3.58": rank}
        if best[b] != TAU:
            _, diff, (lo, hi) = pooled(ARMS[best[b]], M2, benches=(b,))
            entry.update(best_minus_3_58=round(float(diff), 2), ci=[round(float(lo), 2), round(float(hi), 2)])
        else:
            entry.update(best_minus_3_58=0.0, ci=None)
        per[b] = entry
    pooled_acc = {}
    for t, a in ARMS.items():
        pooled_acc[t] = sum(accuracy(a, b) * SIZE[b] for b in BENCH) / sum(SIZE.values())
    vs = {}
    for t, a in ARMS.items():
        if t != TAU:
            _, diff, (lo, hi) = pooled(a, M2)
            vs[str(t)] = {"minus_3.58": round(float(diff), 2), "ci": [round(float(lo), 2), round(float(hi), 2)]}
    o_diff, (o_lo, o_hi) = oracle_pooled(best)
    return {"per_benchmark": per,
            "pooled": {"acc": {str(t): round(v, 2) for t, v in pooled_acc.items()},
                       "best_tau": max(pooled_acc, key=pooled_acc.get), "vs_3.58": vs},
            "own_best_per_benchmark_pooled": {"minus_3.58": round(float(o_diff), 3),
                                              "ci": [round(float(o_lo), 3), round(float(o_hi), 3)],
                                              "best_tau": best}}


# ---------------------------------------------------------------------------------------- main
def report(res: dict) -> None:
    a = res["w4a"]
    print(f"\nW4a  split fit (continuous): tau {a['tau_full_split']} (golden {a['tau_full_split_exact']}); "
          f"bootstrap median {a['tau']['median']} [{a['tau']['lo']}, {a['tau']['hi']}], "
          f"{100 * a['share_within_10pct_of_3.58']:.0f}% within 10% of 3.58")
    for b in ("lvbench", "cgbench"):
        p = a["scored"][b]["points_through_S_line"]
        print(f"     {b}: points through the S line over the draws {p['median']} [{p['lo']}, {p['hi']}]")
    print("W4b  learning curve (tau median [95%]; LVBench / CG-Bench: NLL - NLL(3.58), points via S line)")
    for r in res["w4b"]:
        s = r["scored"]
        how = "w/o repl." if r["sampling"].startswith("without") else "with repl."
        print(f"     k={r['videos']:3d} ({how}) tau {r['tau']['median']:.2f} "
              f"[{r['tau']['lo']:.2f}, {r['tau']['hi']:.2f}] edge {r['share_at_grid_edge']:.3f} | "
              + " | ".join(f"{b[:3]} dNLL {s[b]['nll_minus_at_3.58']['mean']:+.3f} "
                           f"[{s[b]['nll_minus_at_3.58']['lo']:+.3f}, {s[b]['nll_minus_at_3.58']['hi']:+.3f}] "
                           f"pts {s[b]['points_through_S_line']['mean']:+.2f} "
                           f"[{s[b]['points_through_S_line']['lo']:+.2f}, {s[b]['points_through_S_line']['hi']:+.2f}]"
                           for b in ("lvbench", "cgbench")))
    print("W4c  clue intervals moved by d s at each end (tau continuous [boot 95%], paper procedure)")
    for r in res["w4c"]:
        print(f"     d={r['delta_s']:+3d}: evidence {r['evidence_seconds_mean']:6.1f} s, tau "
              f"{r['tau_continuous']:.3f} [{r['tau_boot']['lo']:.2f}, {r['tau_boot']['hi']:.2f}], paper "
              f"{r['tau_paper_procedure']:.3f}; pts LVB {r['lvbench_points_through_S_line']:+.2f} "
              f"CG {r['cgbench_points_through_S_line']:+.2f}")
    d = res["w4d"]
    print("W4d  accuracy at 3.58 against the best of seven")
    for b, e in d["per_benchmark"].items():
        print(f"     {b:9s} 3.58 {e['acc']['3.58']:5.2f} rank {e['rank_of_3.58']}; best tau {e['best_tau']} "
              f"{e['acc'][str(e['best_tau'])]:5.2f}: {e['best_minus_3_58']:+.2f} {e['ci']}")
    print(f"     pooled best tau {d['pooled']['best_tau']}; vs 3.58: {d['pooled']['vs_3.58']}")
    print(f"     each benchmark at its own best: {d['own_best_per_benchmark_pooled']}")


def main() -> None:
    t0 = time.time()
    sets = load_sets()
    tab = tables(sets)
    lines = s_lines()
    curves = bench_curves(tab, lines)
    print(f"tables ready in {time.time() - t0:.0f} s", flush=True)
    res = {"constants": {"tau_ref": TAU, "N": N, "grid": [0.1, 100.0, len(GRID)], "deltas": DELTAS,
                         "sizes": SIZES, "s_lines": lines},
           "reference": {b: {k: v for k, v in c.items() if k not in ("nll", "S")} for b, c in curves.items()},
           "w4a": w4a(tab, sets, curves, lines), "w4b": w4b(tab, sets, curves, lines),
           "w4c": w4c(tab, sets, curves, lines), "w4d": w4d()}
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    report(res)
    print(f"\nwrote {os.path.join(TL, OUT)} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
