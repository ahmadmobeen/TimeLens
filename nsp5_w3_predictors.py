"""NS-P5 Study W3 (METHODS-LOG 2026-11-12, wave 1, Study W): prediction before the run.

Registered: competing predictors (mean N p(E), the chance of at least one frame on the evidence,
held-out NLL, S) scored on the eight registered placements; LVBench's line used to predict CG-Bench
and the reverse. Descriptive (no reading registered).

Each predictor is a mean over questions of a function of p(E), the probability a placement puts on
the annotated evidence seconds (tools/s_law.py's probability and evidence, N = 64):
  S     log(1 + N p(E))            the registered predictor
  Npe   N p(E)                     expected frames on the evidence
  hit   1 - (1 - p(E))^N           chance of at least one frame on the evidence (N independent draws
                                   from p, the standard reading of "the chance")
  nll   -log p(E), floored 1e-300  held-out negative log-likelihood of the evidence

Part 1, the eight registered placements (P-S1's five, P-S2's three; CG-Bench): each predictor's line
is fitted on the same twelve variants P-S1's line was fitted on (2026-11-10), and scored by how many
of the eight observed accuracies fall inside its 95% prediction interval (t, 10 df, as registered),
its mean absolute error and its largest error. Intervals for the mean absolute error, and for each
predictor's MAE minus S's, from the video bootstrap (the line refit in every draw).

Part 2, transfer: each benchmark's line (the paper's S-line placements: s_law.VARIANTS, on CG-Bench
without the P-S1 arms) predicts every placement with a probability on the other benchmark (s_law's
twenty variants, the three P-S2 arms and C2's two redone hedged scorers), with the line's intercept
and slope unchanged. Scored by mean absolute error, mean error (the level shift between the
benchmarks), the SD of the errors, R^2 of the predictions and their correlation with the observed
accuracies; MAE with a bootstrap interval (both benchmarks' videos resampled). The realized log
coverage C (Fig. 3's line, which needs a finished run) is added as a reference.

    env -u VIRTUAL_ENV OMP_NUM_THREADS=8 .venv/bin/python -u nsp5_w3_predictors.py

writes logs/nsp5/analysis/w3_predictors.json.
"""
from __future__ import annotations

import numpy as np

import nsp5_w_data as W

PREDICTORS = ("S", "Npe", "hit", "nll")


def bench_data(bench: str) -> dict:
    d = W.s_arrays(bench)
    boot = W.boot_for(bench, d["vid"])
    pr = W.predictors(d["pe"])
    done = ~np.isnan(d["acc"]).any(1)
    d["done"] = done
    d["acc0"] = np.where(done, 100 * np.nanmean(d["acc"], 1), np.nan)
    d["accb"] = 100 * boot.means(np.where(done[:, None], d["acc"], 0.0))
    d["x0"] = {k: v.mean(1) for k, v in pr.items()}
    d["xb"] = {k: boot.means(v) for k, v in pr.items()}
    return d


def registered_eight(cg: dict) -> dict:
    plist = cg["plist"]
    base = np.array([p["base12"] for p in plist])
    order = list(W.s_law.P_S1) + list(W.s_law.P_S2)
    reg = np.array([next(i for i, p in enumerate(plist) if p["arm"] == a and p["registered"]) for a in order])
    obs = cg["acc0"][reg]
    out, mae_b = {}, {}
    for k in PREDICTORS:
        x0, xb = cg["x0"][k], cg["xb"][k]
        l0 = W.fit_lines(x0[base], cg["acc0"][base])
        pred, lo, hi = W.prediction_interval(l0, x0[reg])
        err = obs - pred
        lb = W.fit_lines(xb[:, base], cg["accb"][:, base])
        errb = cg["accb"][:, reg] - (lb["intercept"][:, None] + lb["slope"][:, None] * xb[:, reg])
        mae_b[k] = np.abs(errb).mean(1)
        out[k] = {"line": {"intercept": float(l0["intercept"]), "slope": float(l0["slope"]),
                           "r2": float(l0["r2"]), "resid_sd": float(l0["resid_sd"])},
                  "inside": int(((obs >= lo) & (obs <= hi)).sum()),
                  "mae": float(np.abs(err).mean()), "mae_ci": W.ci(mae_b[k]),
                  "max_abs_error": float(np.abs(err).max()),
                  "arms": {a: {"x": float(x0[i]), "predicted": float(p), "pi": [float(a_), float(b_)],
                               "observed": float(o), "error": float(e)}
                           for a, i, p, a_, b_, o, e in zip(order, reg, pred, lo, hi, obs, err)}}
    for k in PREDICTORS:
        out[k]["mae_minus_S"] = {"value": out[k]["mae"] - out["S"]["mae"], "ci": W.ci(mae_b[k] - mae_b["S"])}
    return out


def transfer_scores(err: np.ndarray, obs: np.ndarray, pred: np.ndarray) -> dict:
    return {"n": int(len(err)), "mae": float(np.abs(err).mean()), "mean_error": float(err.mean()),
            "sd_error": float(err.std(ddof=1)),
            "r2_fixed_line": float(1 - (err ** 2).sum() / ((obs - obs.mean()) ** 2).sum()),
            "correlation": float(np.corrcoef(pred, obs)[0, 1])}


def transfer(data: dict) -> dict:
    out = {}
    for src, tgt in (("lvbench", "cgbench"), ("cgbench", "lvbench")):
        s, t = data[src], data[tgt]
        line = np.array([p["line"] for p in s["plist"]])
        target = t["done"]
        reg = np.array([p["registered"] for p in t["plist"]])
        res = {}
        for k in PREDICTORS:
            l0 = W.fit_lines(s["x0"][k][line], s["acc0"][line])
            pred = l0["intercept"] + l0["slope"] * t["x0"][k][target]
            obs = t["acc0"][target]
            r = transfer_scores(obs - pred, obs, pred)
            lb = W.fit_lines(s["xb"][k][:, line], s["accb"][:, line])
            errb = t["accb"][:, target] - (lb["intercept"][:, None] + lb["slope"][:, None] * t["xb"][k][:, target])
            r["mae_ci"] = W.ci(np.abs(errb).mean(1))
            r["source_line"] = {"intercept": float(l0["intercept"]), "slope": float(l0["slope"]),
                                "n": int(l0["n"])}
            if reg.any():
                m = reg[target]
                r["registered_eight"] = transfer_scores((obs - pred)[m], obs[m], pred[m])
            res[k] = r
        res["C_reference"] = c_transfer(src, tgt)
        out[f"{src}_to_{tgt}"] = res
    return out


def c_transfer(src: str, tgt: str) -> dict:
    """The realized-coverage line of Fig. 3 on one benchmark predicting the other's Fig. 3 runs."""
    lines = {}
    for b in (src, tgt):
        keys = W.bench_keys(b)
        runs, Y, Nn = W.fig3_arrays(b, keys)
        lines[b] = (runs, 100 * np.nanmean(Y, 1), np.nanmean(np.log1p(Nn), 1))
    runs, acc, c = lines[src]
    fit = np.array([not r["registered"] for r in runs])
    l0 = W.fit_lines(c[fit], acc[fit])
    _, obs, ct = lines[tgt]
    pred = l0["intercept"] + l0["slope"] * ct
    r = transfer_scores(obs - pred, obs, pred)
    r["source_line"] = {"intercept": float(l0["intercept"]), "slope": float(l0["slope"]), "n": int(l0["n"])}
    return r


def main() -> None:
    data = {b: bench_data(b) for b in W.BENCHES}
    res = {"draws": W.DRAWS, "seed": W.SEED, "registered_eight": registered_eight(data["cgbench"]),
           "transfer": transfer(data)}
    print("== the eight registered placements (lines on the twelve CG-Bench variants)")
    for k, e in res["registered_eight"].items():
        print(f"  {k:4s} R^2(12) {e['line']['r2']:.3f}  inside {e['inside']}/8  MAE {e['mae']:.2f} "
              f"[{e['mae_ci'][0]:.2f}, {e['mae_ci'][1]:.2f}]  max {e['max_abs_error']:.2f}  MAE minus S "
              f"{e['mae_minus_S']['value']:+.2f} [{e['mae_minus_S']['ci'][0]:+.2f}, {e['mae_minus_S']['ci'][1]:+.2f}]")
    for direction, r in res["transfer"].items():
        print(f"== {direction}")
        for k, e in r.items():
            extra = ""
            if "registered_eight" in e:
                extra = f"  on the eight: MAE {e['registered_eight']['mae']:.2f} bias {e['registered_eight']['mean_error']:+.2f}"
            ci = f" [{e['mae_ci'][0]:.2f}, {e['mae_ci'][1]:.2f}]" if "mae_ci" in e else ""
            print(f"  {k:11s} n {e['n']:2d} MAE {e['mae']:.2f}{ci} bias {e['mean_error']:+.2f} SD {e['sd_error']:.2f} "
                  f"R^2 {e['r2_fixed_line']:+.2f} r {e['correlation']:.3f}{extra}")
    print("wrote", W.write_json("w3_predictors.json", res))


if __name__ == "__main__":
    main()
