"""NS-P1 B — length-ADAPTIVE routing of the training-free fixes (the method leg).

Applying decode recalibration or denser sampling to EVERY query is a redistribution: short improves
but mid/long pay, so overall mIoU drops (PAPER §6.4.1). The fix: a router gated on the model's OWN
first-pass predicted length — apply the intervention only when L_pred < tau, else keep the baseline
answer. The model over-predicts short moments only to ~3 s, so true-longs (predicted >~7 s) are left
untouched; the goal is NET-POSITIVE (short-bucket recall up, overall mIoU >= baseline).

Fully CPU / from existing predictions (no GPU): baseline fps=2, denser fps=8 (Exp 3, full set), and the
quantile recalibration map are all already computed. We sweep tau for two arms — route->recal and
route->dense — and report the short-gain vs overall-mIoU trade-off.

  env -u VIRTUAL_ENV uv run python research/TimeLens/router_eval.py
"""
import glob

import numpy as np

from benchmarks.datasets.public.charades_timelens import read_timelens_eval

R = "/home/mobeen/codes/active_repos/NeedleScan/research/TimeLens"
BASE = f"{R}/logs/TimeLens-7B_20260615_085540/charades-timelens.jsonl"
DENSE = sorted(glob.glob(f"{R}/logs/denser_fps8/*/charades-timelens.jsonl"))[-1]
TAUS = [0.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0, 1e9]  # 0 = never route (=baseline); 1e9 = always


def iou(a, b):
    if a is None:
        return 0.0
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = (a[1] - a[0]) + (b[1] - b[0]) - inter
    return inter / union if union > 0 else 0.0


def load(path):
    """key (video_id, query) -> {gt, dur, pred(span or None)}."""
    s, p = read_timelens_eval(path, from_answers=True)
    out = {}
    for samp, pred in zip(s, p):
        gt = samp.target_spans[0]
        top = pred.pred_spans[0] if pred.pred_spans else None
        out[(samp.video_id, samp.query)] = {
            "gt": (float(gt[0]), float(gt[1])), "dur": samp.duration,
            "pred": (float(top[0]), float(top[1])) if top else None}
    return out


def fit_quantile(cal_lpred, cal_lgt):
    xp = np.sort(np.asarray(cal_lpred, float)); gq = np.sort(np.asarray(cal_lgt, float))
    n = len(xp); u = (np.arange(n) + 0.5) / n
    return lambda lp: float(np.interp(np.interp(lp, xp, u, left=u[0], right=u[-1]), u, gq))


def recal_preds(base):
    """Center-preserving quantile recalibration of the baseline pred lengths, 5-fold by video."""
    keys = list(base); vids = sorted({k[0] for k in keys})
    rng = np.random.default_rng(0); rng.shuffle(vids)
    fold = {v: i % 5 for i, v in enumerate(vids)}
    out = {}
    for f in range(5):
        cal = [base[k] for k in keys if fold[k[0]] != f and base[k]["pred"]]
        fq = fit_quantile([r["pred"][1] - r["pred"][0] for r in cal],
                          [r["gt"][1] - r["gt"][0] for r in cal])
        for k in keys:
            if fold[k[0]] != f:
                continue
            r = base[k]; out[k] = None
            if r["pred"]:
                s, e = r["pred"]; c = (s + e) / 2.0; L = max(0.1, fq(e - s))
                ns, ne = c - L / 2.0, c + L / 2.0
                d = r["dur"]
                if ns < 0:
                    ns, ne = 0.0, L
                if d and ne > d:
                    ne, ns = d, max(0.0, d - L)
                out[k] = (ns, ne)
    return out


def metrics(routed, base):
    keys = list(base); n = len(keys)
    ious = [iou(routed.get(k), base[k]["gt"]) for k in keys]
    miou = sum(ious) / n
    r05 = sum(x >= 0.5 for x in ious) / n
    r07 = sum(x >= 0.7 for x in ious) / n
    s0 = [k for k in keys if (base[k]["gt"][1] - base[k]["gt"][0]) < 2.0]
    s0r05 = sum(iou(routed.get(k), base[k]["gt"]) >= 0.5 for k in s0) / len(s0)
    lps = [routed[k][1] - routed[k][0] for k in keys if routed.get(k) and routed[k][1] > routed[k][0]]
    em2 = sum(x < 2 for x in lps) / len(lps) if lps else float("nan")
    routed_frac = sum(1 for k in keys if routed.get(k) != base[k]["pred"]) / n
    return dict(miou=miou, r05=r05, r07=r07, s0r05=s0r05, em2=em2, routed=routed_frac)


def route(base, alt, tau):
    """alt pred when baseline-predicted length < tau, else baseline pred."""
    out = {}
    for k in base:
        bp = base[k]["pred"]
        blen = (bp[1] - bp[0]) if bp else 1e9
        out[k] = alt.get(k) if blen < tau else bp
    return out


def main():
    base = load(BASE)
    dense = load(DENSE)
    recal = recal_preds(base)
    print(f"baseline n={len(base)} | dense overlap={len(set(base) & set(dense))} | DENSE={DENSE.split('/')[-2]}")
    b = metrics({k: base[k]["pred"] for k in base}, base)
    print(f"\n  BASELINE: mIoU={b['miou']:.4f} R@0.5={b['r05']:.4f} R@0.7={b['r07']:.4f} "
          f"[0,2)R@0.5={b['s0r05']:.3f} <2s_emit={b['em2']:.3f}")
    dense_spans = {k: v["pred"] for k, v in dense.items()}
    for name, alt in (("route->RECAL", recal), ("route->DENSE(fps8)", dense_spans)):
        print(f"\n=== {name} — tau sweep (tau on baseline predicted length) ===")
        print(f"  {'tau':>5} {'%routed':>8} {'mIoU':>7} {'dmIoU':>7} {'R@0.5':>7} {'R@0.7':>7} {'[0,2)R@.5':>9} {'<2s_emit':>9}")
        for tau in TAUS:
            m = metrics(route(base, alt, tau), base)
            tag = " <-- net+" if (m["miou"] >= b["miou"] and m["s0r05"] > b["s0r05"]) else ""
            print(f"  {tau:>5.0f} {m['routed']*100:>7.1f}% {m['miou']:>7.4f} {m['miou']-b['miou']:>+7.4f} "
                  f"{m['r05']:>7.4f} {m['r07']:>7.4f} {m['s0r05']:>9.3f} {m['em2']:>9.3f}{tag}")


if __name__ == "__main__":
    main()
