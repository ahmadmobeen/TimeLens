"""NS-P1 D-lever verdict: score baseline vs rebalanced vs natural-control on charades-timelens.

Computes — through ONE identical code path for all three models — the pre-registered PASS-bar
quantities: the signed length-miscalibration curve (med_log2_ratio crossover), OLS β(L_pred~L_gt)
with cluster-bootstrap CI + RTM null, length-proportional (scale-fair) recall, the <2 s emission
rate, and per-bucket recall. Run from the repo ROOT with the main venv (needs `benchmarks`):

  env -u VIRTUAL_ENV uv run python research/TimeLens/score_dlever.py [<natural_shard_dir>]
"""
import json
import os
import sys

from benchmarks.datasets.public.charades_timelens import (
    CharadesTimeLensBenchmark,
    read_timelens_eval,
)
from benchmarks.length_strata import LengthStratifiedGrounding
from benchmarks.length_profile import (
    length_fair_recall,
    length_ratio_slope,
    rtm_null,
    signed_length_profile,
)

R = "/home/mobeen/codes/active_repos/NeedleScan/research/TimeLens"
# NOTE: the eval harness MERGES the 2 GPU shards into one charades-timelens.jsonl at the end.
import glob


def _latest(pattern):
    """Newest charades-timelens.jsonl under a glob of run dirs (handles the eval's merged output)."""
    hits = sorted(glob.glob(pattern))
    return [hits[-1]] if hits else []


# Baseline = the full-set (3363-pair) clean reproduction (085540); 073438 was a 300-pair subset.
# The three finetune arms auto-discover their newest eval run by glob (timestamps vary per run).
MODELS = {
    "baseline": [f"{R}/logs/TimeLens-7B_20260615_085540/charades-timelens.jsonl"],
    "natural(ctrl)": _latest(f"{R}/logs/dlever_nat/*/charades-timelens.jsonl"),
    "rebalanced": _latest(f"{R}/logs/dlever_reb/*/charades-timelens.jsonl"),
    "speedaug": _latest(f"{R}/logs/dlever_speedaug/*/charades-timelens.jsonl"),
}


def load_rows(paths):
    """read_timelens_eval per shard -> aligned (samples, preds) lists; concat; build the long table."""
    samples, preds = [], []
    for path in paths:
        s, p = read_timelens_eval(path, from_answers=True)  # (samples, preds), aligned & positional
        samples.extend(s)
        preds.extend(p)
    rows = LengthStratifiedGrounding(CharadesTimeLensBenchmark()).to_long_table(preds, samples)
    return rows


def emission_lt(rows, thresh=2.0):
    lps = [
        float(r["pred_end"]) - float(r["pred_start"])
        for r in rows
        if r.get("pred_start") is not None and float(r["pred_end"]) > float(r["pred_start"])
    ]
    if not lps:
        return float("nan"), 0
    return sum(1 for x in lps if x < thresh) / len(lps), len(lps)


def overall(rows):
    n = len(rows)
    return {
        "n": n,
        "mIoU": sum(float(r["iou"]) for r in rows) / n,
        "R@0.5": sum(1 for r in rows if float(r["iou"]) >= 0.5) / n,
        "R@0.7": sum(1 for r in rows if float(r["iou"]) >= 0.7) / n,
    }


def report(name, rows):
    gt_lt2 = sum(1 for r in rows if float(r["L_gt"]) < 2.0) / len(rows)
    em2, npred = emission_lt(rows, 2.0)
    ov = overall(rows)
    beta = length_ratio_slope(rows, n_boot=500, seed=0)
    rtm = rtm_null(rows, n_sim=200, seed=0)
    sp = signed_length_profile(rows)
    fair = {d["bin"]: d.get("recall") for d in length_fair_recall(rows, alpha=0.25)}
    print(f"\n========== {name}  (n={ov['n']}, GT<2s frac={gt_lt2:.3f}) ==========")
    print(f"  overall: mIoU={ov['mIoU']:.4f}  R@0.5={ov['R@0.5']:.4f}  R@0.7={ov['R@0.7']:.4f}")
    print(f"  <2s emission rate = {em2:.4f}  (of {npred} preds; GT reference {gt_lt2:.3f})")
    print(f"  beta(L_pred~L_gt) = {beta.get('beta'):.3f}  CI[{beta.get('ci_lo'):.3f},{beta.get('ci_hi'):.3f}]  compression={beta.get('compression')}")
    print(f"  RTM: beta_obs={rtm.get('beta_obs'):.3f}  null_lo={rtm.get('beta_null_lo'):.3f}  genuine_compression={rtm.get('genuine_compression')}")
    print(f"  {'bin':>9} {'n':>5} {'med_log2':>9} {'|dctr|':>7} {'mIoU':>6} {'R@0.5':>6} {'R@0.7':>6} {'fair@.25':>8}")
    for d in sp:
        b = d["bin"]
        if d["n"] == 0:
            continue
        fb = fair.get(b)
        print(f"  {b:>9} {int(d['n']):>5} {d.get('med_log2_ratio',float('nan')):>9.3f} "
              f"{d.get('med_abs_center_err',float('nan')):>7.2f} {d.get('mIoU',float('nan')):>6.3f} "
              f"{d.get('R@1_IoU0.5',float('nan')):>6.3f} {d.get('R@1_IoU0.7',float('nan')):>6.3f} "
              f"{(fb if fb is not None else float('nan')):>8.3f}")
    return {"name": name, **ov, "emission_lt2": em2, "gt_lt2": gt_lt2, "beta": beta.get("beta")}


def main():
    if len(sys.argv) > 1:
        import glob
        MODELS["natural"] = sorted(glob.glob(f"{sys.argv[1]}/charades-timelens.jsonl"))
    results = {}
    for name, paths in MODELS.items():
        existing = [p for p in paths if os.path.exists(p)]
        if not existing:
            print(f"[skip] {name}: no prediction files found ({paths})")
            continue
        results[name] = report(name, load_rows(existing))
    print("\n========== PASS-BAR SUMMARY ==========")
    for name, r in results.items():
        print(f"  {name:>11}: <2s_emission={r['emission_lt2']:.3f}  beta={r['beta']:.3f}  "
              f"mIoU={r['mIoU']:.3f}  R@0.5={r['R@0.5']:.3f}")


if __name__ == "__main__":
    main()
