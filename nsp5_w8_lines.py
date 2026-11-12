"""NS-P5 Study W8, part (METHODS-LOG 2026-11-12, wave 1, Study W): the Fig. 3 (C) and S lines refit
inside each video-bootstrap draw (videos resampled within each benchmark, 5,000 draws, seed 0).

Each draw recomputes every run's accuracy, log coverage C and S from the drawn questions and refits
the line, so the intervals carry the questions all runs share:

  C line   Fig. 3: accuracy on C = mean log(1 + n_E) over the runs of tools/s_law.realized (CG-Bench:
           without the eight registered arms). Also the R^2 of the two simpler counts the supplement
           compares (mean frames on the evidence; share with at least one) and C's lead over each, and
           the line fitted without the three published selectors predicting them (the "within 1.2
           points" claim).
  S line   accuracy on S = mean log(1 + 64 p(E)) over s_law.VARIANTS (CG-Bench: without the five
           P-S1 arms), and the likelihood line beside it.
  P-S      the twelve-variant CG-Bench line that P-S1 and P-S2 were registered from. Honest
           prediction intervals, three ways: (i) in each draw the t interval of the refit line
           (10 df, as registered) is recomputed and the drawn accuracy checked against it (the share
           of draws inside, per arm and for all eight); (ii) the arm's residual from the refit line
           with its interval; (iii) a prospective interval that uses no registered outcome: the
           prediction plus the 2.5 and 97.5 percentiles of the twelve variants' leave-one-out errors
           pooled over the draws.

    env -u VIRTUAL_ENV OMP_NUM_THREADS=8 .venv/bin/python -u nsp5_w8_lines.py

writes logs/nsp5/analysis/w8_lines.json.
"""
from __future__ import annotations

import numpy as np

import nsp5_w_data as W

STATS = ("slope", "intercept", "r2", "resid_sd")


def summarize(x0: np.ndarray, y0: np.ndarray, xb: np.ndarray, yb: np.ndarray) -> tuple[dict, dict]:
    """The line on the original runs and its bootstrap intervals (refit in every draw)."""
    l0, lb = W.fit_lines(x0, y0), W.fit_lines(xb, yb)
    out = {k: {"value": float(l0[k]), "ci": W.ci(lb[k])} for k in STATS}
    out["n"] = int(l0["n"])
    return out, lb


def c_line(bench: str) -> dict:
    keys = W.bench_keys(bench)
    boot = W.boot_for(bench, W.videos(keys))
    runs, Y, Nn = W.fig3_arrays(bench, keys)
    acc0, accb = 100 * np.nanmean(Y, 1), 100 * boot.means(Y)
    feats = {"C": np.log1p(Nn), "frames": Nn, "hit": (Nn >= 1).astype(float)}
    feats = {k: np.where(np.isnan(Y), np.nan, v) for k, v in feats.items()}
    x0 = {k: np.nanmean(v, 1) for k, v in feats.items()}
    xb = {k: boot.means(v) for k, v in feats.items()}
    fit = np.array([not r["registered"] for r in runs])
    out = {"runs": [{"name": r["name"], "arm": r["arm"], "C": float(x0["C"][i]), "acc": float(acc0[i]),
                     "registered": r["registered"], "published": r["published"]} for i, r in enumerate(runs)]}
    lines = {}
    for k in feats:
        out[f"line_{k}"], lines[k] = summarize(x0[k][fit], acc0[fit], xb[k][:, fit], accb[:, fit])
    out["r2_C_minus_frames"] = {"value": out["line_C"]["r2"]["value"] - out["line_frames"]["r2"]["value"],
                                "ci": W.ci(lines["C"]["r2"] - lines["frames"]["r2"])}
    out["r2_C_minus_hit"] = {"value": out["line_C"]["r2"]["value"] - out["line_hit"]["r2"]["value"],
                             "ci": W.ci(lines["C"]["r2"] - lines["hit"]["r2"])}
    # the line without the published selectors, predicting them
    pub = np.array([r["published"] for r in runs])
    own = fit & ~pub
    l0 = W.fit_lines(x0["C"][own], acc0[own])
    lb = W.fit_lines(xb["C"][:, own], accb[:, own])
    out["published_predicted"] = {}
    for i in np.flatnonzero(pub & fit):
        pred0 = l0["intercept"] + l0["slope"] * x0["C"][i]
        errb = accb[:, i] - (lb["intercept"] + lb["slope"] * xb["C"][:, i])
        out["published_predicted"][runs[i]["arm"]] = {"predicted": float(pred0), "observed": float(acc0[i]),
                                                      "error": float(acc0[i] - pred0), "error_ci": W.ci(errb)}
    errs = np.abs(np.stack([accb[:, i] - (lb["intercept"] + lb["slope"] * xb["C"][:, i])
                            for i in np.flatnonzero(pub & fit)], 1))
    out["published_max_abs_error_ci"] = W.ci(errs.max(1))
    # The committed figure (4ace5c1) drew logs/nsp5/s_law.json of 03:51, written before CG-Bench's
    # C2 BLIP hedged run (x1_cgbench_cal_kelly_blip_nr, 05:36) was complete: that line, for reference
    late = np.array([r["stem"] in SNAPSHOT_MISSING for r in runs])
    if late.any():
        snap = fit & ~late
        out["line_C_committed_figure"], _ = summarize(x0["C"][snap], acc0[snap], xb["C"][:, snap],
                                                      accb[:, snap])
    return out


SNAPSHOT_MISSING = {"x1_cgbench_cal_kelly_blip_nr"}


def s_line(bench: str) -> dict:
    d = W.s_arrays(bench)
    boot = W.boot_for(bench, d["vid"])
    pr = W.predictors(d["pe"])
    plist = d["plist"]
    done = ~np.isnan(d["acc"]).any(1)
    acc0 = np.where(done, 100 * np.nanmean(d["acc"], 1), np.nan)
    accb = 100 * boot.means(np.where(done[:, None], d["acc"], 0.0))
    s0, sb = pr["S"].mean(1), boot.means(pr["S"])
    n0, nb = pr["nll"].mean(1), boot.means(pr["nll"])
    line = np.array([p["line"] for p in plist])
    out = {"placements": [{"name": p["name"], "arm": p["arm"], "S": float(s0[i]), "nll": float(n0[i]),
                           "acc": None if np.isnan(acc0[i]) else float(acc0[i]),
                           "line": p["line"], "registered": p["registered"]} for i, p in enumerate(plist)]}
    out["line_S"], _ = summarize(s0[line], acc0[line], sb[:, line], accb[:, line])
    out["line_nll"], _ = summarize(n0[line], acc0[line], nb[:, line], accb[:, line])
    if bench == "cgbench":
        out["registered"] = registered(plist, s0, sb, acc0, accb)
    return out


def registered(plist: list, s0: np.ndarray, sb: np.ndarray, acc0: np.ndarray, accb: np.ndarray) -> dict:
    """Honest prediction intervals for P-S1 and P-S2 from the twelve-variant line, refit per draw."""
    base = np.array([p["base12"] for p in plist])
    assert base.sum() == 12, base.sum()
    order = list(W.s_law.P_S1) + list(W.s_law.P_S2)
    idx = [next(i for i, p in enumerate(plist) if p["arm"] == a and p["registered"]) for a in order]
    out, lb = summarize(s0[base], acc0[base], sb[:, base], accb[:, base])
    l0 = W.fit_lines(s0[base], acc0[base])
    t = W.t_quantile(0.975, 10)
    lev = lambda x, L: 1 + 1 / 12 + (x - L["xbar"]) ** 2 / L["sxx"]
    loo = W.loo_residuals(sb[:, base], accb[:, base]).ravel()
    q_lo, q_hi = np.percentile(loo, [2.5, 97.5])
    loo0 = W.loo_residuals(s0[base], acc0[base])
    j_lo, j_hi = np.percentile(loo0, [2.5, 97.5])
    arms, inside_b = {}, []
    for a, i in zip(order, idx):
        reg = W.s_law.P_S1.get(a) or W.s_law.P_S2[a][2]
        pred0 = l0["intercept"] + l0["slope"] * s0[i]
        predb = lb["intercept"] + lb["slope"] * sb[:, i]
        half_b = t * lb["resid_sd"] * np.sqrt(lev(sb[:, i], lb))
        ins = np.abs(accb[:, i] - predb) <= half_b
        inside_b.append(ins)
        arms[a] = {"label": W.s_law.LABELS[a], "S": float(s0[i]), "observed": float(acc0[i]),
                   "registered_prediction": reg[0], "registered_pi": [reg[1], reg[2]],
                   "inside_registered": bool(reg[1] <= acc0[i] <= reg[2]),
                   "refit_prediction": float(pred0),
                   "share_of_draws_inside_refit_pi": float(ins.mean()),
                   "residual": float(acc0[i] - pred0), "residual_ci": W.ci(accb[:, i] - predb),
                   "pi_loo_pooled": [float(pred0 + q_lo), float(pred0 + q_hi)],
                   "inside_pi_loo_pooled": bool(pred0 + q_lo <= acc0[i] <= pred0 + q_hi),
                   "pi_bootstrap_se": pi_boot_se(l0, lb, s0[i], t)}
        arms[a]["inside_pi_bootstrap_se"] = bool(arms[a]["pi_bootstrap_se"][0] <= acc0[i]
                                                 <= arms[a]["pi_bootstrap_se"][1])
    count = np.stack(inside_b, 1).sum(1)
    out["arms"] = arms
    out["registered_line_check"] = {"intercept": float(l0["intercept"]), "slope": float(l0["slope"]),
                                    "registered": list(W.s_law.P_S1_LINE)}
    out["inside_count_over_draws"] = {"mean": float(count.mean()), "median": float(np.median(count)),
                                      "ci": W.ci(count), "share_all_8": float(np.mean(count == 8)),
                                      "share_7_or_8": float(np.mean(count >= 7))}
    out["loo_pooled_error_quantiles"] = [float(q_lo), float(q_hi)]
    out["loo_original_sample_quantiles"] = [float(j_lo), float(j_hi)]
    out["inside_counts"] = {k: int(sum(v[k] for v in arms.values()))
                            for k in ("inside_registered", "inside_pi_loo_pooled", "inside_pi_bootstrap_se")}
    return out


def pi_boot_se(l0: dict, lb: dict, x: float, t: float) -> list:
    """Prediction +- t sqrt(residual variance + bootstrap variance of the line at x): the classical
    interval with the independence-based parameter term replaced by the video bootstrap's."""
    pred0 = l0["intercept"] + l0["slope"] * x
    var_line = np.var(lb["intercept"] + lb["slope"] * x, ddof=1)
    half = t * np.sqrt(l0["resid_sd"] ** 2 + var_line)
    return [float(pred0 - half), float(pred0 + half)]


def main() -> None:
    res = {"draws": W.DRAWS, "seed": W.SEED}
    for bench in W.BENCHES:
        res[bench] = {"C": c_line(bench), "S": s_line(bench)}
        c, s = res[bench]["C"], res[bench]["S"]
        fmt = lambda e: f"{e['value']:.3f} [{e['ci'][0]:.3f}, {e['ci'][1]:.3f}]"
        print(f"== {bench}: C line on {c['line_C']['n']} runs: R^2 {fmt(c['line_C']['r2'])}, slope "
              f"{fmt(c['line_C']['slope'])}; R^2 frames {fmt(c['line_frames']['r2'])}, hit "
              f"{fmt(c['line_hit']['r2'])}; C minus frames {fmt(c['r2_C_minus_frames'])}, minus hit "
              f"{fmt(c['r2_C_minus_hit'])}")
        for a, e in c["published_predicted"].items():
            print(f"   {a:14s} predicted {e['predicted']:.2f} observed {e['observed']:.2f} error "
                  f"{e['error']:+.2f} [{e['error_ci'][0]:+.2f}, {e['error_ci'][1]:+.2f}]")
        print(f"   S line on {s['line_S']['n']} variants: R^2 {fmt(s['line_S']['r2'])}, slope "
              f"{fmt(s['line_S']['slope'])}; NLL line R^2 {fmt(s['line_nll']['r2'])}")
    reg = res["cgbench"]["S"]["registered"]
    print("P-S line (12 variants):", reg["registered_line_check"], "R^2", reg["r2"], "slope", reg["slope"])
    for a, e in reg["arms"].items():
        print(f"   {e['label']:28s} obs {e['observed']:.2f} reg PI {e['registered_pi']} draws inside "
              f"{e['share_of_draws_inside_refit_pi']:.3f} residual {e['residual']:+.2f} "
              f"[{e['residual_ci'][0]:+.2f}, {e['residual_ci'][1]:+.2f}] LOO-PI "
              f"[{e['pi_loo_pooled'][0]:.1f}, {e['pi_loo_pooled'][1]:.1f}] bootSE-PI "
              f"[{e['pi_bootstrap_se'][0]:.1f}, {e['pi_bootstrap_se'][1]:.1f}]")
    print("   inside count over draws:", reg["inside_count_over_draws"], reg["inside_counts"])
    print("wrote", W.write_json("w8_lines.json", res))


if __name__ == "__main__":
    main()
