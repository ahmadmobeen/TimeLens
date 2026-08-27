"""Score a trained C boundary-head ARM through the IDENTICAL length-stratified path as the
text-decode baseline + D-lever arms.

Loads the POST-ANSWER eval cache, runs the head (arm read from the ckpt), converts
(ĉ,ŵ) -> (start,end) seconds, and reports per-length-bin ``med_abs_center_err`` (the
|Δcenter| floor) and ``R@1_IoU0.7`` via :func:`benchmarks.length_profile.signed_length_profile`,
then the head-vs-text-decode-baseline delta on the kill-switch strata ([0,2s), [2,3s)).

Run from the repo ROOT with the main venv (needs ``benchmarks``)::

  env -u VIRTUAL_ENV uv run python research/TimeLens/score_head.py --ckpt ans
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # so `import c_head.*` resolves when run from repo root

from benchmarks.datasets.public.charades_timelens import (  # noqa: E402
    CharadesTimeLensBenchmark,
    read_timelens_eval,
)
from benchmarks.length_profile import (  # noqa: E402
    length_fair_recall,
    length_ratio_slope,
    signed_length_profile,
)
from benchmarks.length_strata import LengthStratifiedGrounding  # noqa: E402
from needlescan.base.types import Prediction, Sample  # noqa: E402

from c_head import config as C  # noqa: E402
from c_head.head import BoundaryHead, CrossAttnBoundaryHead  # noqa: E402


def load_eval_cache(cache_dir: str, tag: str, need_vis: bool):
    files = sorted(Path(cache_dir).glob(f"{tag}.shard*of*.pt"))
    if not files:
        raise FileNotFoundError(f"no eval cache shards matching {tag}.shard*of*.pt in {cache_dir}")
    h_ans, vis, vis_mean, metas = [], [], [], []
    for f in files:
        d = torch.load(f, map_location="cpu")
        if not len(d["meta"]):
            continue
        h_ans.append(d["h_ans"])
        vis_mean.append(d["vis_mean"])
        metas.extend(d["meta"])
        if need_vis:
            vis.append(d["vis"])
    return {
        "h_ans": torch.cat(h_ans).float(),
        "vis_mean": torch.cat(vis_mean).float(),
        "vis": torch.cat(vis) if need_vis else None,  # fp16
        "metas": metas,
    }


@torch.no_grad()
def _predict(head, arm, cache, bs=256):
    if arm == "xattn":
        out = []
        n = cache["h_ans"].shape[0]
        for i in range(0, n, bs):
            out.append(head(cache["h_ans"][i:i + bs], cache["vis"][i:i + bs].float()))
        return torch.cat(out)
    if arm == "vismean":
        return head(cache["vis_mean"])
    return head(cache["h_ans"])


def rows_from_head(cache_dir: str, ckpt_path: Path, eval_tag: str):
    ck = torch.load(ckpt_path, map_location="cpu")
    arm = ck.get("arm", "ans")
    hs = ck.get("hidden_size", C.HIDDEN_SIZE)
    head = CrossAttnBoundaryHead(hs) if arm == "xattn" else BoundaryHead(hs)
    head.load_state_dict(ck["state_dict"])
    head.eval()
    cache = load_eval_cache(cache_dir, eval_tag, need_vis=(arm == "xattn"))
    cw = _predict(head, arm, cache)
    samples, preds = [], []
    for i, m in enumerate(cache["metas"]):
        dur = float(m["duration"])
        c, w = float(cw[i, 0]), float(cw[i, 1])
        s = max(0.0, (c - w / 2.0) * dur)
        e = min(dur, (c + w / 2.0) * dur)
        if e <= s:
            e = min(dur, s + 0.1)
        samples.append(Sample(
            sample_id=f"{m['video_id']}::{i}", video_id=m["video_id"], query=m["query"],
            duration=dur, target_spans=((float(m["gt_start"]), float(m["gt_end"])),),
        ))
        preds.append(Prediction(pred_spans=((s, e),)))
    return arm, LengthStratifiedGrounding(CharadesTimeLensBenchmark()).to_long_table(preds, samples)


def rows_from_eval_jsonl(path: Path):
    samples, preds = read_timelens_eval(path, from_answers=True)
    return LengthStratifiedGrounding(CharadesTimeLensBenchmark()).to_long_table(preds, samples)


def overall(rows):
    n = len(rows)
    return {
        "n": n,
        "mIoU": sum(float(r["iou"]) for r in rows) / n,
        "R@0.5": sum(1 for r in rows if float(r["iou"]) >= 0.5) / n,
        "R@0.7": sum(1 for r in rows if float(r["iou"]) >= 0.7) / n,
    }


def report(name, rows):
    ov = overall(rows)
    beta = length_ratio_slope(rows, n_boot=500, seed=0)
    sp = signed_length_profile(rows)
    fair = {d["bin"]: d.get("recall") for d in length_fair_recall(rows, alpha=0.25)}
    print(f"\n========== {name}  (n={ov['n']}) ==========")
    print(f"  overall: mIoU={ov['mIoU']:.4f}  R@0.5={ov['R@0.5']:.4f}  R@0.7={ov['R@0.7']:.4f}")
    print(f"  beta(L_pred~L_gt)={beta.get('beta'):.3f}  CI[{beta.get('ci_lo'):.3f},{beta.get('ci_hi'):.3f}]")
    print(f"  {'bin':>9} {'n':>5} {'med_log2':>9} {'|dctr|':>7} {'mIoU':>6} {'R@0.5':>6} {'R@0.7':>6} {'fair@.25':>8}")
    by_bin = {}
    for d in sp:
        if d["n"] == 0:
            continue
        by_bin[d["bin"]] = d
        fb = fair.get(d["bin"])
        print(f"  {d['bin']:>9} {int(d['n']):>5} {d.get('med_log2_ratio', float('nan')):>9.3f} "
              f"{d.get('med_abs_center_err', float('nan')):>7.2f} {d.get('mIoU', float('nan')):>6.3f} "
              f"{d.get('R@1_IoU0.5', float('nan')):>6.3f} {d.get('R@1_IoU0.7', float('nan')):>6.3f} "
              f"{(fb if fb is not None else float('nan')):>8.3f}")
    return by_bin


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="ckpt name under ckpt/ (or a full path)")
    ap.add_argument("--cache_dir", default=str(C.CACHE_DIR))
    ap.add_argument("--eval_tag", default="eval.postans")
    ap.add_argument("--baseline", default=str(C.BASELINE_EVAL_JSONL))
    ap.add_argument("--json", default=None, help="write machine-readable gate metrics here (for the research-loop)")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    if not ckpt_path.exists():
        ckpt_path = Path(C.CKPT_DIR) / f"{args.ckpt}.pt"

    arm, head_rows = rows_from_head(args.cache_dir, ckpt_path, args.eval_tag)
    head_bins = report(f"C-head [{ckpt_path.stem}] arm={arm}", head_rows)
    base_bins = {}
    if Path(args.baseline).exists():
        base_bins = report("text-decode baseline", rows_from_eval_jsonl(Path(args.baseline)))
    else:
        print(f"\n[warn] baseline jsonl not found at {args.baseline}; skipping baseline comparison")

    # Kill-switch focus: the sub-second strata. Head must LOWER |Δcenter| and RAISE R@0.7.
    print("\n========== KILL-SWITCH (head vs text-decode baseline) ==========")
    print(f"  {'bin':>9} {'|dctr| head':>12} {'|dctr| base':>12} {'R@0.7 head':>11} {'R@0.7 base':>11}")
    for b in ("[0,2)", "[2,3)"):
        h, bl = head_bins.get(b, {}), base_bins.get(b, {})
        print(f"  {b:>9} {h.get('med_abs_center_err', float('nan')):>12.2f} "
              f"{bl.get('med_abs_center_err', float('nan')):>12.2f} "
              f"{h.get('R@1_IoU0.7', float('nan')):>11.3f} {bl.get('R@1_IoU0.7', float('nan')):>11.3f}")
    print("\n  C target: head LOWERS [0,2s) |Δcenter| below the baseline 1.05s and RAISES R@0.7 above 0.015,")
    print("  with no mid/long regression. Compare arms ans / xattn / vismean to attribute the gain.")

    if args.json:
        import json as _json
        ml = [head_bins.get(b, {}).get("mIoU") for b in ("[7,10)", "[10,15)", "[15+)")]
        ml = [x for x in ml if isinstance(x, (int, float))]
        ov = overall(head_rows)
        gate = {
            "ckpt": ckpt_path.stem,
            "arm": arm,
            "s0_r07": head_bins.get("[0,2)", {}).get("R@1_IoU0.7"),
            "s0_dcenter": head_bins.get("[0,2)", {}).get("med_abs_center_err"),
            "s1_r07": head_bins.get("[2,3)", {}).get("R@1_IoU0.7"),
            "overall_miou": ov["mIoU"],
            "overall_r07": ov["R@0.7"],
            "midlong_miou": (sum(ml) / len(ml)) if ml else None,
            "baseline_s0_r07": base_bins.get("[0,2)", {}).get("R@1_IoU0.7"),
            "baseline_s0_dcenter": base_bins.get("[0,2)", {}).get("med_abs_center_err"),
        }
        Path(args.json).write_text(_json.dumps(gate, indent=2))
        print(f"\n[json] gate metrics -> {args.json}")


if __name__ == "__main__":
    main()
