"""NS-P5 Study W2 (METHODS-LOG 2026-11-12, wave 1, Study W): interventions on the Fig. 3 line.

Registered: the LVBench oracles (Gaussian sigma 15, 60 and 240 s; off the evidence), the same-video
placebos, the 16 to 128-frame runs and the Q-Frame rule; residuals from the paper's line with
video-bootstrap intervals. Descriptive (no reading registered).

The line is the paper's Fig. 3 line, accuracy = a + b C with C = mean log(1 + n_E), fitted on the
runs of tools/s_law.realized (CG-Bench: without the eight registered arms), recomputed from the
current logs. Residual = observed accuracy minus the line at the run's own C. In each of 5,000 draws
(videos resampled within the benchmark, seed 0) every run's accuracy and C are recomputed, the line
is refitted and the residual taken again; the interval is the 2.5 and 97.5 percentiles. A run that
covers a subset of the questions (CG-Bench's placebos skip its 317 single-question videos) is
compared with the line fitted on the same questions. For context each row also carries the line's
classical 95% prediction interval (t, n - 2 df) and whether C lies inside the range the line was
fitted on.

The oracles' frames on the evidence are recomputed from their logged centre and duration
(nsp5_w_data.e3_in_span). The budget runs are Study G's and I3's (uniform, CaliFrame and GMM-EVA's
selection at 16, 32 and 128 frames, Qwen2.5-VL-7B, the shared token budget).

    env -u VIRTUAL_ENV OMP_NUM_THREADS=8 .venv/bin/python -u nsp5_w2_interventions.py

writes logs/nsp5/analysis/w2_interventions.json.
"""
from __future__ import annotations

import numpy as np

import nsp5_w_data as W

BUDGET_ARMS = {"uniform": "uniform", "q8_kelly_eps0": "CaliFrame", "f2c_topk": "GMM-EVA's selection"}
PLACEBOS = {"q8_kelly_eps0_placebo": "CaliFrame, placebo", "q8_kelly_eps010_placebo": "share 0.1, placebo",
            "q8_kelly_eps025_placebo": "share 0.25, placebo", "q8_kelly_placebo_nr": "share 0.5, placebo",
            "a2_kelly_placebo": "SigLIP share 0.5 (A2), placebo",
            "a3_kelly_placebo": "fused SigLIP+8B share 0.5 (A3), placebo"}


def interventions(bench: str) -> list[tuple[str, str, str]]:
    """(kind, file stem, label) of every registered intervention run on the benchmark."""
    pre = "" if bench == "lvbench" else W.CG_PRE
    out = []
    if bench == "lvbench":
        out += [("oracle", "e3_gauss_s15", "oracle, Gaussian sigma 15 s"),
                ("oracle", "e3_gauss_s60", "oracle, Gaussian sigma 60 s"),
                ("oracle", "e3_gauss_s240", "oracle, Gaussian sigma 240 s"),
                ("oracle", "e3_offpeak_s60", "off the evidence, sigma 60 s, 600 s away")]
    out += [("placebo", pre + s, lab) for s, lab in PLACEBOS.items()
            if W.file_rows(pre + s) is not None]
    out += [("budget", f"{pre}n{n}_{a}", f"{lab}, {n} frames") for a, lab in BUDGET_ARMS.items()
            for n in (16, 32, 128)]
    out += [("qframe", pre + "m_qframe128_q8", "Q-Frame's rule, 128 candidates"),
            ("qframe", pre + "m_qframe_all_q8", "Q-Frame's rule, every second")]
    return out


def analyse(bench: str) -> dict:
    keys = W.bench_keys(bench)
    boot = W.boot_for(bench, W.videos(keys))
    runs, Y, Nn = W.fig3_arrays(bench, keys)
    fit = np.array([not r["registered"] for r in runs])
    Yl, Cl = Y[fit], np.log1p(Nn[fit])
    out = {"line_runs": [r["arm"] for r, f in zip(runs, fit) if f], "rows": []}
    cache = {}
    for kind, stem, label in interventions(bench):
        arr = W.run_arrays(stem, keys)
        if arr is None or np.isnan(arr[0]).all():
            out["rows"].append({"kind": kind, "stem": stem, "label": label, "missing": True})
            continue
        y, n = arr
        present = ~np.isnan(y)
        sig = present.tobytes()
        if sig not in cache:      # the line on the questions this run covers, and its draws
            Ym, Cm = np.where(present, Yl, np.nan), np.where(present, Cl, np.nan)
            cache[sig] = (W.fit_lines(np.nanmean(Cm, 1), 100 * np.nanmean(Ym, 1)),
                          W.fit_lines(boot.means(Cm), 100 * boot.means(Ym)),
                          float(np.nanmin(np.nanmean(Cm, 1))), float(np.nanmax(np.nanmean(Cm, 1))))
        l0, lb, c_min, c_max = cache[sig]
        c = np.where(present, np.log1p(n), np.nan)
        acc0, c0 = 100 * np.nanmean(y), float(np.nanmean(c))
        accb, cb = 100 * boot.means(y[None])[:, 0], boot.means(c[None])[:, 0]
        pred, lo, hi = W.prediction_interval(l0, c0)
        resb = accb - (lb["intercept"] + lb["slope"] * cb)
        frames = W.file_rows(stem)
        n_frames = sorted({r["n_frames"] for r in frames.values()})
        out["rows"].append({
            "kind": kind, "stem": stem, "label": label, "questions": int(present.sum()),
            "n_frames": n_frames, "mean_frames_on_evidence": float(np.nanmean(n)),
            "share_with_none": float(np.nanmean(n[present] == 0)), "C": c0, "acc": float(acc0),
            "line_prediction": float(pred), "residual": float(acc0 - pred), "residual_ci": W.ci(resb),
            "line_pi95": [float(lo), float(hi)], "inside_line_pi": bool(lo <= acc0 <= hi),
            "C_inside_fitted_range": bool(c_min <= c0 <= c_max), "fitted_C_range": [c_min, c_max]})
    full = cache[np.ones(len(keys), bool).tobytes()] if np.ones(len(keys), bool).tobytes() in cache else None
    l0 = full[0] if full else W.fit_lines(np.nanmean(Cl, 1), 100 * np.nanmean(Yl, 1))
    out["line"] = {k: float(l0[k]) for k in ("intercept", "slope", "r2", "resid_sd")}
    out["line"]["n"] = int(l0["n"])
    return out


def main() -> None:
    res = {"draws": W.DRAWS, "seed": W.SEED, "e3_check": W.e3_check()}
    print("E3 recomputation check:", res["e3_check"])
    for bench in W.BENCHES:
        res[bench] = r = analyse(bench)
        L = r["line"]
        print(f"== {bench}: line acc = {L['intercept']:.2f} + {L['slope']:.2f} C on {L['n']} runs, "
              f"R^2 {L['r2']:.3f}, residual SD {L['resid_sd']:.2f}")
        for e in r["rows"]:
            if e.get("missing"):
                print(f"  {e['label']:42s} MISSING")
                continue
            flag = "" if e["C_inside_fitted_range"] else "  (C outside the fitted range)"
            print(f"  {e['label']:42s} n_E {e['mean_frames_on_evidence']:5.2f} C {e['C']:.3f} acc "
                  f"{e['acc']:5.2f} line {e['line_prediction']:5.2f} residual {e['residual']:+6.2f} "
                  f"[{e['residual_ci'][0]:+.2f}, {e['residual_ci'][1]:+.2f}] "
                  f"{'inside' if e['inside_line_pi'] else 'OUTSIDE'} PI{flag}")
    print("wrote", W.write_json("w2_interventions.json", res))


if __name__ == "__main__":
    main()
