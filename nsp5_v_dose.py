"""NS-P5 Study V (METHODS-LOG 2026-11-12, wave 1, registered in da7940b): a controlled dose of
frames on the evidence.

Arms: `--arm dose --dose_k k --refill nearest`, exactly k frames spread evenly over the evidence
seconds and the other 64 - k spread evenly over the video at least 30 s from the evidence. LVBench
k in {0, 1, 2, 4, 8, 16, 32} (v_dose<k>), CG-Bench k in {0, 1, 4, 16} (x1_cgbench_v_dose<k>).
Population: the questions whose logged frames on the evidence equal k in every arm; the count left
out is reported.

- V1 (primary): log(1 + k) against k and 1[k >= 1] by video-blocked cross-validated log loss of the
  logit with question fixed effects, on each benchmark. nsp5_w1_law's machinery, unchanged: the
  conditional logit, five folds by video, the held-out conditional NLL per observation.
- V2: accuracy at k = 4 minus k = 1, and at 16 minus 4, paired over the population, with 95%
  intervals from the video bootstrap (5,000 draws, seed 0).
- Readings (fixed in the registration): V1 (log wins on each benchmark) and V2 (both contrasts
  above zero) hold: the dose-response is log-like, and the paper may keep "follows the log". A
  threshold fits best: accuracy depends mainly on whether a frame lands on the evidence. Otherwise
  reported per benchmark.

    env -u VIRTUAL_ENV OMP_NUM_THREADS=8 .venv/bin/python -u nsp5_v_dose.py

writes logs/nsp5/analysis/v_dose.json and prints the summary.
"""
from __future__ import annotations

import numpy as np

import nsp5_w1_law as L1
import nsp5_w_data as W

DOSES = {"lvbench": (0, 1, 2, 4, 8, 16, 32), "cgbench": (0, 1, 4, 16)}
FORMS = ("log", "n", "thr")
CONTRASTS = ((4, 1), (16, 4))


def stem(bench: str, k: int) -> str:
    return f"v_dose{k}" if bench == "lvbench" else f"{W.CG_PRE}v_dose{k}"


def build() -> tuple[dict, dict]:
    """Questions x doses for both benchmarks stacked (absent doses masked; questions outside the
    population fully masked), and the population counts."""
    width = max(len(v) for v in DOSES.values())
    parts, census = [], {}
    for bench, doses in DOSES.items():
        keys = W.bench_keys(bench)
        Y, N = np.zeros((len(keys), width)), np.zeros((len(keys), width))
        M = np.zeros((len(keys), width), bool)
        exact = np.ones(len(keys), bool)
        for t, k in enumerate(doses):
            arr = W.run_arrays(stem(bench, k), keys)
            if arr is None:
                raise SystemExit(f"missing run {stem(bench, k)}")
            y, n = arr
            ok = ~np.isnan(y)
            exact &= ok & (n == k)
            Y[ok, t], N[ok, t], M[ok, t] = y[ok], n[ok], True
        M &= exact[:, None]
        census[bench] = {"questions": len(keys), "population": int(exact.sum()),
                         "left_out": int((~exact).sum())}
        parts.append({"bench": np.array([bench] * len(keys)), "keys": keys, "vid": W.videos(keys),
                      "Y": Y, "N": N, "M": M, "exact": exact})
    data = {f: (sum((p[f] for p in parts), []) if f in ("keys", "vid")
                else np.concatenate([p[f] for p in parts])) for f in parts[0]}
    return data, census


def dose_contrasts(data: dict, bench: str, boots: dict) -> dict:
    """Accuracy at each dose and the registered contrasts, over the population, video bootstrap."""
    sel = (data["bench"] == bench) & data["exact"]
    wq = L1.video_counts(data, sel, boots)
    col = {k: j for j, k in enumerate(DOSES[bench])}
    Y = data["Y"][sel]
    out = {"accuracy": {}, "contrasts": {}}
    for k, j in col.items():
        out["accuracy"][str(k)] = [float(100 * Y[:, j].mean()), W.ci(100 * (wq @ Y[:, j]) / wq.sum(1))]
    for hi, lo in CONTRASTS:
        if hi in col and lo in col:
            d = Y[:, col[hi]] - Y[:, col[lo]]
            out["contrasts"][f"{hi} minus {lo}"] = [float(100 * d.mean()), W.ci(100 * (wq @ d) / wq.sum(1))]
    return out


def main() -> None:
    data, census = build()
    print("population:", census, flush=True)
    boots = W.make_boots({b: [v for v, bb in zip(data["vid"], data["bench"]) if bb == b]
                          for b in W.BENCHES})
    res, holds, threshold_best = {"census": census, "doses": DOSES}, {}, {}
    for bench in DOSES:
        sel = (data["bench"] == bench) & data["exact"]
        fit = L1.analyse(data, sel, np.ones_like(data["M"]), boots, bench, forms=FORMS)
        v2 = dose_contrasts(data, bench, boots)
        win = fit["cv"]["cond_nll"]["winner"]
        v2_ok = all(c[1][0] > 0 for c in v2["contrasts"].values())
        holds[bench] = bool(win == "log" and v2_ok)
        threshold_best[bench] = win == "thr"
        res[bench] = {"fit": fit, "v2": v2, "cv_winner": win, "v2_both_above_zero": bool(v2_ok)}
        print(f"== {bench}: CV winner {win}; per-observation held-out NLL "
              + ", ".join(f"{f} {fit['forms'][f]['cv_cond_nll_per_obs']:.4f}" for f in FORMS), flush=True)
        for k, (a, (lo, hi)) in v2["accuracy"].items():
            print(f"    k={k:>2}: {a:5.1f} [{lo:5.1f}, {hi:5.1f}]")
        for name, (d, (lo, hi)) in v2["contrasts"].items():
            print(f"    {name}: {d:+.1f} [{lo:+.1f}, {hi:+.1f}]")
    res["reading"] = ("log-like on both benchmarks" if all(holds.values()) else
                      "threshold fits best" if all(threshold_best.values()) else
                      f"per benchmark: {holds}")
    print("Reading:", res["reading"])
    print("wrote", W.write_json("v_dose.json", res))


if __name__ == "__main__":
    main()
