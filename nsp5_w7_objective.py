"""NS-P5 Study W, item W7: the objective (CPU only, no answer model).

Pre-registered in METHODS-LOG 2026-11-12 ("Reviewer-requested experiments, wave 1", Study W): "W7 the
objective: on the calibration split and the LVBench and CG-Bench annotations, the temperature that
maximizes S at N = 16, 32, 64 and 128, and S of water-filling against proportional placement. A GPU
follow-up is registered separately if one predicts a gain of 0.5 point or more."

S is the paper's (s_law.py): the mean over questions of log(1 + n_E), n_E the frames on the annotated
evidence seconds. Proportional placement gives second t the share n_t = N p_t, so n_E = N p(E).
Water-filling maximizes sum_t p_t log(1 + n_t) under sum_t n_t = N, n_t >= 0: n_t = max(0, p_t/lam - 1).
Both allocations are scored by the same log coverage of the evidence, log(1 + sum_{t in E} n_t):
  continuous    the allocations as real numbers (for proportional placement this is S itself);
  whole frames  proportional placement by the method's own rule, frame i at F^-1((i - 1/2) / N),
                counted in the second it falls in (eval_lvbench_foveate.curve_times, kelly);
                water-filling by greedy marginal gains p_t log((k + 2) / (k + 1)), the exact
                maximizer of sum_t p_t log(1 + n_t) over whole frames (it only ever uses the N most
                probable seconds).
p = softmax(z / tau) of the 8B per-option maximum curves over the curve's full length, as s_law.py;
evidence seconds as calibration_checks.py.

  W7a  for N = 16, 32, 64, 128 and each annotation set (calibration split, LVBench, CG-Bench), the
       continuous maximizer of S over tau (nsp5_w4_tau's 801-point log grid, parabola-refined;
       golden-section check at N = 64), with a video bootstrap (5,000 draws, seed 0), and what the
       calibration split's maximizer (transfer) and each benchmark's own (in-sample, an upper bound)
       predict through the benchmark's 64-frame S line (logs/nsp5/s_law.json).
  W7b  water-filling against proportional placement at tau 3.58 for each N, paired, videos
       resampled (5,000 draws, seed 0); water-filling at the temperature that maximizes its own S
       on the calibration split (72-point grid, 5% steps); points through the 64-frame S line.

Points at N other than 64 use the 64-frame line and are marked as an extrapolation. The transfer
intervals resample the calibration split (seed 1) and the benchmark (seed 0) independently.

    cd research/TimeLens && .venv/bin/python nsp5_w7_objective.py | tee logs/nsp5/analysis/w7_objective.log

Writes logs/nsp5/analysis/w7_objective.json and caches logs/nsp5/analysis/w7_alloc_tables.npz.
"""
from __future__ import annotations

import json
import multiprocessing as mp
import os
import sys
import time

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "8")

import numpy as np  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import nsp5_w4_tau as w4  # noqa: E402  (calibration_checks, the sets, the log p(E) tables)
from nsp5_pooled import load, path, pooled  # noqa: E402

OUT = os.path.join(w4.OUT_DIR, "w7_objective.json")
CACHE = os.path.join(w4.OUT_DIR, "w7_alloc_tables.npz")
NS = (16, 32, 64, 128)
TAU = w4.TAU
COARSE = TAU * np.exp(0.05 * np.arange(-36, 36))        # 0.59 .. 20.4, 3.58 exactly at J_TAU
LOG_COARSE = np.log(COARSE)
J_TAU = 36
KINDS = ("prop", "wf", "prop_int", "wf_int")           # continuous (prop = S), whole frames
GAIN = np.log(np.arange(2, max(NS) + 2) / np.arange(1, max(NS) + 1))   # log((k + 2) / (k + 1))
DRAWS = 5000
TRIGGER = 0.5                                           # points, through the 64-frame S line
SETS = ("cal", "lvbench", "cgbench")
BENCHES = ("lvbench", "cgbench")

_JOB: dict = {}


# --------------------------------------------------------------------------------- allocations
def alloc_scores(z: np.ndarray, m: np.ndarray, tau: float) -> np.ndarray:
    """[len(NS), 2, 4] at one temperature: log(1 + frames on the evidence) and the objective
    sum_t p_t log(1 + n_t), for proportional, water-filling, and both in whole frames."""
    p = np.exp((z - z.max()) / tau)
    p /= p.sum()
    T = len(p)
    order = np.argsort(-p, kind="stable")
    s = p[order]
    c = np.cumsum(s)
    rank = np.arange(1, T + 1)
    cdf = np.cumsum(p)
    out = np.empty((len(NS), 2, len(KINDS)))
    for i, n in enumerate(NS):
        k = int(np.flatnonzero(s * (n + rank) > c)[-1]) + 1     # active seconds (a prefix)
        wf = np.maximum(0.0, p / (c[k - 1] / (n + k)) - 1.0)
        q = np.minimum(np.searchsorted(cdf, (np.arange(n) + 0.5) / n), T - 1)
        prop_int = np.bincount(q, minlength=T).astype(float)
        top = order[:n]
        g = p[top][:, None] * GAIN[None, :n]
        sel = np.argpartition(-g.ravel(), n - 1)[:n]
        wf_int = np.zeros(T)
        np.add.at(wf_int, top[sel // n], 1.0)
        for j, a in enumerate((n * p, wf, prop_int, wf_int)):
            out[i, 0, j] = np.log1p(a[m].sum())
            out[i, 1, j] = float(p @ np.log1p(a))
    return out


def _chunk(bounds: tuple) -> tuple:
    lo, hi = bounds
    z, m = _JOB["z"], _JOB["m"]
    out = np.empty((hi - lo, len(COARSE), len(NS), 2, len(KINDS)))
    for i in range(lo, hi):
        for j, t in enumerate(COARSE):
            out[i - lo, j] = alloc_scores(z[i], m[i], t)
    return lo, out


def alloc_tables(sets: dict) -> dict:
    """Per set [questions, COARSE, NS, (coverage, objective), KINDS]; cached."""
    fp = w4.fingerprint(sets)
    if os.path.exists(CACHE):
        c = np.load(CACHE)
        if np.array_equal(c["fingerprint"], fp) and np.array_equal(c["coarse"], COARSE):
            return {k: c[k] for k in SETS}
    out = {}
    for name in SETS:
        t0 = time.time()
        z, masks, _ = sets[name]
        _JOB.update(z=z, m=[r[0] for r in masks])
        bounds = [(i, min(len(z), i + 8)) for i in range(0, len(z), 8)]
        arr = np.empty((len(z), len(COARSE), len(NS), 2, len(KINDS)))
        with mp.get_context("fork").Pool(w4.PROCS) as pool:
            for lo, a in pool.imap_unordered(_chunk, bounds):
                arr[lo:lo + len(a)] = a
        out[name] = arr
        print(f"  allocations {name}: {arr.shape} in {time.time() - t0:.0f} s", flush=True)
    np.savez(CACHE, fingerprint=fp, coarse=COARSE, **out)
    return out


# --------------------------------------------------------------------------------------- W7a
def exact_s_opt(sets: dict, name: str, n: int, tau0: float) -> float:
    """Golden-section maximizer of the exact S (no grid) near tau0."""
    z, masks, _ = sets[name]

    def neg_s(th: float) -> float:
        t = np.array([np.exp(th)])
        return -float(np.mean([np.log1p(n * np.exp(w4.lse(c[m[0]], t)[0] - w4.lse(c, t)[0]))
                               for c, m in zip(z, masks)]))
    return float(np.exp(w4.golden(neg_s, np.log(tau0) - 0.03, np.log(tau0) + 0.03, iters=30)))


def ci(x: np.ndarray, nd: int = 4) -> list:
    return [round(float(v), nd) for v in np.percentile(x, [2.5, 97.5])]


def w7a(tab: dict, sets: dict, lines: dict) -> dict:
    w = {s: w4.draw_weights(sets[s][2], DRAWS, seed=0) for s in SETS}
    w_cal1 = w4.draw_weights(sets["cal"][2], DRAWS, seed=1)     # independent of the benchmarks'
    out = {}
    for n in NS:
        tbl = {s: w4.s_of(tab[s][:, 0, :], n) for s in SETS}
        ref = {s: w4.s_of(tab[s + "_ref"][:, 0, w4.I_TAU], n) for s in SETS}
        res = {}
        for s in SETS:
            t, v, edge = w4.optimum(tbl[s].mean(0), maximize=True)
            tb, vb, _ = w4.optimum(w4.wmean(w[s], tbl[s]), maximize=True)
            gain_b = vb - w[s] @ ref[s] / w[s].sum(1)
            res[s] = {"tau": round(float(t[0]), 4), "tau_ci": ci(tb), "tau_cosine": round(float(t[0]) / 100, 5),
                      "at_grid_edge": bool(edge[0]), "S_at_opt": round(float(v[0]), 4),
                      "S_at_3.58": round(float(ref[s].mean()), 4),
                      "S_gain_in_sample": round(float(v[0] - ref[s].mean()), 4), "S_gain_ci": ci(gain_b)}
            if n == 64:
                res[s]["tau_exact_golden"] = round(exact_s_opt(sets, s, n, float(t[0])), 4)
            if s in BENCHES:
                b = lines[s]["slope"]
                res[s]["points_in_sample"] = round(b * res[s]["S_gain_in_sample"], 3)
                res[s]["points_in_sample_ci"] = [round(b * x, 3) for x in res[s]["S_gain_ci"]]
        t_cal = res["cal"]["tau"]
        tb_cal, _, _ = w4.optimum(w4.wmean(w_cal1, tbl["cal"]), maximize=True)
        for s in BENCHES:
            gain = float(np.interp(np.log(t_cal), w4.LOG_GRID, tbl[s].mean(0)) - ref[s].mean())
            gain_b = (w4.at(w4.wmean(w[s], tbl[s]), tb_cal) - w[s] @ ref[s] / w[s].sum(1))
            b = lines[s]["slope"]
            res[s].update({"S_gain_at_cal_tau": round(gain, 4), "S_gain_at_cal_tau_ci": ci(gain_b),
                           "points_at_cal_tau": round(b * gain, 3),
                           "points_at_cal_tau_ci": [round(b * x, 3) for x in ci(gain_b)]})
        res["line_frames"] = 64
        res["extrapolated"] = n != 64
        out[str(n)] = res
    out["nll_optimum"] = {s: round(float(w4.optimum((-tab[s][:, 0, :]).mean(0))[0][0]), 4) for s in SETS}
    return out


# --------------------------------------------------------------------------------------- W7b
def w7b(alloc: dict, sets: dict, lines: dict) -> dict:
    w = {s: w4.draw_weights(sets[s][2], DRAWS, seed=0) for s in SETS}
    w_cal1 = w4.draw_weights(sets["cal"][2], DRAWS, seed=1)
    out = {}
    for i, n in enumerate(NS):
        res = {}
        for s in SETS:
            a = alloc[s][:, J_TAU, i]                          # [questions, 2, KINDS]
            cov = {k: round(float(a[:, 0, j].mean()), 4) for j, k in enumerate(KINDS)}
            obj = {k: round(float(a[:, 1, j].mean()), 4) for j, k in enumerate(KINDS)}
            zero = {k: round(float(np.mean(a[:, 0, j] == 0)), 4) for j, k in enumerate(KINDS[2:], 2)}
            res[s] = {"log_coverage": cov, "objective": obj, "share_no_frame_on_evidence": zero}
            for lab, x, y in (("continuous", 1, 0), ("whole_frames", 3, 2)):
                d = a[:, 0, x] - a[:, 0, y]
                db = w[s] @ d / w[s].sum(1)
                res[s][f"wf_minus_prop_{lab}"] = {"S": round(float(d.mean()), 4), "ci": ci(db)}
                if s in BENCHES:
                    b = lines[s]["slope"]
                    res[s][f"wf_minus_prop_{lab}"].update(points=round(b * float(d.mean()), 3),
                                                          points_ci=[round(b * v, 3) for v in ci(db)])
        res["own_tau"] = wf_own_tau(alloc, w, w_cal1, i, lines)
        res["extrapolated"] = n != 64
        out[str(n)] = res
    return out


def wf_own_tau(alloc: dict, w: dict, w_cal1: np.ndarray, i: int, lines: dict) -> dict:
    """Water-filling at the temperature that maximizes its own (continuous) S, against M2 as run
    (proportional at 3.58): the calibration split's maximizer transferred, and each benchmark's own."""
    curve = {s: alloc[s][:, :, i, 0, 1] for s in SETS}      # water-filling coverage [q, COARSE]
    m2 = {s: alloc[s][:, J_TAU, i, 0, 0] for s in SETS}     # proportional at 3.58 = S
    t_cal, _, _ = w4.optimum(curve["cal"].mean(0), LOG_COARSE, maximize=True)
    tb_cal, _, _ = w4.optimum(w4.wmean(w_cal1, curve["cal"]), LOG_COARSE, maximize=True)
    res = {"cal_tau": round(float(t_cal[0]), 4), "cal_tau_ci": ci(tb_cal)}
    for s in BENCHES:
        own, v_own, _ = w4.optimum(curve[s].mean(0), LOG_COARSE, maximize=True)
        gain = float(np.interp(np.log(t_cal[0]), LOG_COARSE, curve[s].mean(0)) - m2[s].mean())
        gain_b = (w4.at(w4.wmean(w[s], curve[s]), tb_cal, LOG_COARSE) - w[s] @ m2[s] / w[s].sum(1))
        b = lines[s]["slope"]
        res[s] = {"S_gain_at_cal_tau": round(gain, 4), "ci": ci(gain_b),
                  "points": round(b * gain, 3), "points_ci": [round(b * x, 3) for x in ci(gain_b)],
                  "own_tau": round(float(own[0]), 4),
                  "points_at_own_tau": round(b * float(v_own[0] - m2[s].mean()), 3)}
    return res


# ------------------------------------------------------------------------------------ context
def realized(alloc: dict) -> dict:
    """The whole-frame proportional coverage at 3.58, N = 64, against the in_span logged by M2's
    runs (+-0.5 s tolerance, curve cut at the duration), as a check of the frame counting."""
    out = {}
    for b, f in (("lvbench", "logs/nsp5/q8_kelly_eps0.jsonl"),
                 ("cgbench", "logs/nsp5/x1_cgbench_q8_kelly_eps0.jsonl")):
        with open(f, encoding="utf-8") as fh:
            n_e = np.array([json.loads(x).get("in_span") or 0 for x in fh if x.strip()], float)
        out[b] = {"logged_C": round(float(np.mean(np.log1p(n_e))), 4),
                  "prop_int_here": round(float(alloc[b][:, J_TAU, NS.index(64), 0, 2].mean()), 4),
                  "S_here": round(float(alloc[b][:, J_TAU, NS.index(64), 0, 0].mean()), 4)}
    return out


def observed(tab: dict, lines: dict) -> dict:
    """The seven run temperatures: S-line prediction against observed accuracy, minus 3.58."""
    out = {}
    for b in BENCHES:
        ref = w4.s_of(tab[b + "_ref"][:, 0, :]).mean(0)          # S at each run temperature
        rows = {}
        for k, (t, arm) in enumerate(sorted(w4.ARMS.items())):
            if t == TAU:
                continue
            _, diff, (lo, hi) = pooled(arm, w4.M2, benches=(b,))
            rows[str(t)] = {"S": round(float(ref[k]), 4),
                            "predicted": round(lines[b]["slope"] * float(ref[k] - ref[w4.I_TAU]), 2),
                            "observed": round(float(diff), 2), "ci": [round(float(lo), 2), round(float(hi), 2)]}
        out[b] = rows
    return out


def triggers(a: dict, bres: dict) -> list:
    """Every 64-frame prediction, flagged if it reaches TRIGGER points."""
    rows = []
    for s in BENCHES:
        r = a["64"][s]
        rows.append(("S-optimal tau of the calibration split", s, r["points_at_cal_tau"], r["points_at_cal_tau_ci"]))
        rows.append(("S-optimal tau of the benchmark itself (in-sample)", s, r["points_in_sample"],
                     r["points_in_sample_ci"]))
        for lab in ("continuous", "whole_frames"):
            e = bres["64"][s][f"wf_minus_prop_{lab}"]
            rows.append((f"water-filling minus proportional at 3.58 ({lab})", s, e["points"], e["points_ci"]))
        e = bres["64"]["own_tau"][s]
        rows.append(("water-filling at the calibration split's S-optimal tau minus M2", s, e["points"],
                     e["points_ci"]))
    return [{"what": w_, "bench": s, "points": p, "ci": c, "reaches_0.5": bool(p >= TRIGGER)}
            for w_, s, p, c in rows]


def report(res: dict) -> None:
    a, b = res["w7a"], res["w7b"]
    print("\nW7a  tau maximizing S (stored scale; x0.01 on cosines) [video bootstrap 95%]")
    for n in NS:
        r = a[str(n)]
        print(f"  N={n:3d}: " + "; ".join(f"{s} {r[s]['tau']:.2f} [{r[s]['tau_ci'][0]:.2f}, {r[s]['tau_ci'][1]:.2f}]"
                                         f" dS {r[s]['S_gain_in_sample']:+.4f}" for s in SETS))
        print("         transfer of the split's tau: " + "; ".join(
            f"{s} dS {r[s]['S_gain_at_cal_tau']:+.4f} -> {r[s]['points_at_cal_tau']:+.2f} pts "
            f"{r[s]['points_at_cal_tau_ci']}" for s in BENCHES) + ("  (64-frame line)" if n != 64 else ""))
    print(f"  NLL optimum (N -> infinity): {a['nll_optimum']}")
    print("W7b  water-filling minus proportional at 3.58 (S; points through the 64-frame line)")
    for n in NS:
        r = b[str(n)]
        for s in SETS:
            c, wfr = r[s]["wf_minus_prop_continuous"], r[s]["wf_minus_prop_whole_frames"]
            pts = (f" | pts {c['points']:+.2f} / {wfr['points']:+.2f}" if s in BENCHES else "")
            print(f"  N={n:3d} {s:8s} cont {c['S']:+.4f} {c['ci']}  whole {wfr['S']:+.4f} {wfr['ci']}{pts}")
        o = r["own_tau"]
        print(f"         water-filling at the split's own tau {o['cal_tau']:.2f}: "
              + "; ".join(f"{s} {o[s]['points']:+.2f} pts {o[s]['points_ci']}" for s in BENCHES))
    print("Context: run temperatures, S-line prediction vs observed (minus 3.58)")
    for bn, rows in res["observed"].items():
        print(f"  {bn}: " + "; ".join(f"{t}: {e['predicted']:+.2f} vs {e['observed']:+.2f} {e['ci']}"
                                      for t, e in rows.items()))
    print(f"Check: {res['realized']}")
    hit = [t for t in res["triggers"] if t["reaches_0.5"]]
    print(f"GPU follow-up trigger (>= {TRIGGER} points at N = 64): {len(hit)} of {len(res['triggers'])}")
    for t in res["triggers"]:
        print(f"  {'*' if t['reaches_0.5'] else ' '} {t['bench']:8s} {t['points']:+.2f} {t['ci']}  {t['what']}")


def main() -> None:
    t0 = time.time()
    sets = w4.load_sets()
    tab = w4.tables(sets)
    lines = w4.s_lines()
    alloc = alloc_tables(sets)
    print(f"tables ready in {time.time() - t0:.0f} s", flush=True)
    for s in SETS:                                            # the coarse grid reproduces S exactly
        exact = w4.s_of(tab[s + "_ref"][:, 0, w4.I_TAU]).mean()
        assert abs(alloc[s][:, J_TAU, NS.index(64), 0, 0].mean() - exact) < 1e-9, s
    res = {"constants": {"tau_ref": TAU, "N": NS, "coarse_grid": [float(COARSE[0]), float(COARSE[-1]), len(COARSE)],
                         "draws": DRAWS, "s_lines": lines,
                         "rounding": "proportional: frame i at F^-1((i - 1/2)/N), counted per second; "
                                     "water-filling: greedy marginal gains (exact integer optimum)"},
           "w7a": w7a(tab, sets, lines), "w7b": w7b(alloc, sets, lines),
           "observed": observed(tab, lines), "realized": realized(alloc)}
    res["triggers"] = triggers(res["w7a"], res["w7b"])
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    report(res)
    print(f"\nwrote {os.path.join(w4.TL, OUT)} in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
