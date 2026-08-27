"""NS-P1 Exp 1 — training-free decode-time length recalibration of the length-compression prior.

The model emits spans whose LENGTH is compressed toward ~3-5s (beta(L_pred~L_gt)=0.60) and rarely
<2s. A decode-time wrapper rescales the emitted span's LENGTH while KEEPING ITS CENTER (the center
error is a roughly constant ~1s absolute floor, smaller than the length error for mid/long moments,
so the center is the more trustworthy quantity). No weight update -> immune to the training support
gap that sank the finetune (PAPER.md §6.4).

Two recalibration targets, which pull OPPOSITE ways and are both reported honestly:
  * QUANTILE matching  : map L_pred through its calibration CDF onto the GT-length CDF. Restores the
                         marginal length distribution BY CONSTRUCTION -> beta->1, <2s emission->GT rate.
                         The "flatten the curve" objective. Helps IoU only insofar as L_pred RANK tracks
                         L_gt rank (reported as the Spearman ceiling).
  * LINEAR (MMSE)      : emit E[L_gt|L_pred] = a*L_pred+b (OLS). Minimizes length MSE; if L_pred is weakly
                         informative this SHRINKS spread (beta worse). Reported for contrast.

Honesty: fit the map on a CALIBRATION split, apply to a DISJOINT test split (5-fold by video_id, so a
video's pairs never span fit/test). Then fit on ALL Charades and apply to ActivityNet (cross-dataset
transfer) to show the recalibration is a property of the MODEL, not a per-dataset overfit.

CPU-only; runs on existing baseline predictions. From repo ROOT with the main venv:
  env -u VIRTUAL_ENV uv run python research/TimeLens/decode_recalibrate.py
"""
import math
import os

import numpy as np

from benchmarks.datasets.public.charades_timelens import (
    CharadesTimeLensBenchmark,
    read_timelens_eval,
)
from benchmarks.length_profile import (
    length_fair_recall,
    length_ratio_slope,
    signed_length_profile,
)
from benchmarks.length_strata import LengthStratifiedGrounding

R = "/home/mobeen/codes/active_repos/NeedleScan/research/TimeLens"
CHARADES = f"{R}/logs/TimeLens-7B_20260615_085540/charades-timelens.jsonl"
ANET = f"{R}/logs/anet/TimeLens-7B_20260617_070017/activitynet-omniembed.jsonl"
MIN_LEN = 0.1  # floor for a recalibrated length (seconds)


def _iou(a, b):
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def to_records(path):
    """One record per predicted pair: gt span, top-1 pred span, duration, video_id."""
    samples, preds = read_timelens_eval(path, from_answers=True)
    recs = []
    for s, p in zip(samples, preds):
        gt = s.target_spans[0]
        top1 = p.pred_spans[0] if p.pred_spans else None
        recs.append({
            "video_id": s.video_id,
            "duration": s.duration,
            "gt": (float(gt[0]), float(gt[1])),
            "pred": (float(top1[0]), float(top1[1])) if top1 else None,
        })
    return recs


def fit_quantile(cal_lpred, cal_lgt):
    """Return f(L_pred)->L_recal that maps the predicted-length CDF onto the GT-length CDF."""
    xp = np.sort(np.asarray(cal_lpred, float))
    gq = np.sort(np.asarray(cal_lgt, float))
    n = len(xp)
    u_grid = (np.arange(n) + 0.5) / n  # plotting-position quantiles

    def f(lp):
        u = np.interp(lp, xp, u_grid, left=u_grid[0], right=u_grid[-1])
        return float(np.interp(u, u_grid, gq))
    return f


def fit_linear(cal_lpred, cal_lgt):
    a, b = np.polyfit(np.asarray(cal_lpred, float), np.asarray(cal_lgt, float), 1)

    def f(lp):
        return a * lp + b
    return f


def recalibrate(recs, f):
    """Center-preserving length rescale; clamp to [0, duration]; missing preds pass through."""
    out = []
    for r in recs:
        nr = dict(r)
        if r["pred"] is not None:
            s, e = r["pred"]
            c = (s + e) / 2.0
            new_len = max(MIN_LEN, f(e - s))
            ns, ne = c - new_len / 2.0, c + new_len / 2.0
            dur = r["duration"] if r["duration"] else None
            if ns < 0:
                ns, ne = 0.0, new_len
            if dur and ne > dur:
                ne, ns = dur, max(0.0, dur - new_len)
            nr["pred"] = (ns, ne)
        out.append(nr)
    return out


def kfold_recalibrate(recs, fit_fn, k=5, seed=0):
    """5-fold by video_id: fit on the other folds, apply to the held-out fold; union the results."""
    rng = np.random.default_rng(seed)
    vids = sorted({r["video_id"] for r in recs})
    rng.shuffle(vids)
    fold_of = {v: i % k for i, v in enumerate(vids)}
    out = []
    for fold in range(k):
        cal = [r for r in recs if fold_of[r["video_id"]] != fold and r["pred"]]
        test = [r for r in recs if fold_of[r["video_id"]] == fold]
        f = fit_fn([r["pred"][1] - r["pred"][0] for r in cal],
                   [r["gt"][1] - r["gt"][0] for r in cal])
        out.extend(recalibrate(test, f))
    return out


def to_rows(recs):
    """Long-table rows for the shared scorers (signed_length_profile / length_ratio_slope / fair)."""
    rows = []
    for r in recs:
        gs, ge = r["gt"]
        ps = pe = None
        iou = 0.0
        if r["pred"] is not None:
            ps, pe = r["pred"]
            iou = _iou(r["pred"], r["gt"])
        rows.append({
            "video_id": r["video_id"], "L_gt": ge - gs,
            "gt_start": gs, "gt_end": ge,
            "pred_start": ps, "pred_end": pe, "iou": iou,
        })
    return rows


def emission_lt2(recs):
    lps = [r["pred"][1] - r["pred"][0] for r in recs if r["pred"] and r["pred"][1] > r["pred"][0]]
    return (sum(1 for x in lps if x < 2.0) / len(lps)) if lps else float("nan")


def length_rank_corr(recs):
    pairs = [(r["pred"][1] - r["pred"][0], r["gt"][1] - r["gt"][0]) for r in recs if r["pred"]]
    a = np.asarray(pairs, float)
    if len(a) < 3:
        return float("nan"), float("nan")
    pear = float(np.corrcoef(a[:, 0], a[:, 1])[0, 1])
    rp = np.argsort(np.argsort(a[:, 0])); rg = np.argsort(np.argsort(a[:, 1]))
    spear = float(np.corrcoef(rp, rg)[0, 1])
    return pear, spear


def summarize(name, recs):
    rows = to_rows(recs)
    n = len(rows)
    miou = sum(r["iou"] for r in rows) / n
    r05 = sum(1 for r in rows if r["iou"] >= 0.5) / n
    r07 = sum(1 for r in rows if r["iou"] >= 0.7) / n
    beta = length_ratio_slope(rows, n_boot=300, seed=0).get("beta")
    em2 = emission_lt2(recs)
    sp = signed_length_profile(rows)
    fair = {d["bin"]: d.get("recall") for d in length_fair_recall(rows, alpha=0.25)}
    print(f"\n===== {name}  (n={n}) =====")
    print(f"  mIoU={miou:.4f}  R@0.5={r05:.4f}  R@0.7={r07:.4f}  beta={beta:.3f}  <2s_emit={em2:.4f}")
    print(f"  {'bin':>9} {'n':>5} {'med_log2':>9} {'mIoU':>6} {'R@0.5':>6} {'R@0.7':>6} {'fair@.25':>8}")
    for d in sp:
        if d["n"] == 0:
            continue
        b = d["bin"]; fb = fair.get(b)
        print(f"  {b:>9} {int(d['n']):>5} {d.get('med_log2_ratio',float('nan')):>9.3f} "
              f"{d.get('mIoU',float('nan')):>6.3f} {d.get('R@1_IoU0.5',float('nan')):>6.3f} "
              f"{d.get('R@1_IoU0.7',float('nan')):>6.3f} {(fb if fb is not None else float('nan')):>8.3f}")
    return {"name": name, "mIoU": miou, "R@0.5": r05, "R@0.7": r07, "beta": beta, "emit2": em2}


def main():
    ch = to_records(CHARADES)
    pear, spear = length_rank_corr(ch)
    print(f"[Charades] predicted-vs-true LENGTH correlation: Pearson={pear:.3f}  Spearman={spear:.3f}")
    print("  (this is the CEILING for any length-only decode fix: if ~0, length carries no signal)")
    res = [summarize("Charades baseline", ch)]
    res.append(summarize("Charades + QUANTILE recal (5-fold)", kfold_recalibrate(ch, fit_quantile)))
    res.append(summarize("Charades + LINEAR/MMSE recal (5-fold)", kfold_recalibrate(ch, fit_linear)))

    # cross-dataset transfer: fit on ALL Charades, apply to ActivityNet
    if os.path.exists(ANET):
        an = to_records(ANET)
        pear_a, spear_a = length_rank_corr(an)
        print(f"\n[ActivityNet] length corr: Pearson={pear_a:.3f}  Spearman={spear_a:.3f}")
        summarize("ANet baseline", an)
        fq = fit_quantile([r["pred"][1] - r["pred"][0] for r in ch if r["pred"]],
                          [r["gt"][1] - r["gt"][0] for r in ch if r["pred"]])
        summarize("ANet + Charades-fit QUANTILE recal (naive transfer)", recalibrate(an, fq))
        # fair generality test: the decode wrapper as a METHOD on a 2nd dataset (self-calibrated,
        # disjoint folds) — isolates "does quantile recal work cross-dataset" from the domain
        # length-distribution mismatch that sinks the naive Charades->ANet transfer above.
        summarize("ANet + ANet-self QUANTILE recal (5-fold)", kfold_recalibrate(an, fit_quantile))

    print("\n===== SUMMARY =====")
    for r in res:
        print(f"  {r['name']:>40}: mIoU={r['mIoU']:.3f} R@0.5={r['R@0.5']:.3f} "
              f"R@0.7={r['R@0.7']:.3f} beta={r['beta']:.3f} <2s={r['emit2']:.3f}")


if __name__ == "__main__":
    main()
