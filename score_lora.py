"""Score the reward-1 LoRA predictions vs the text-decode baseline, side-by-side, through the
IDENTICAL length-stratified path (read_timelens_eval(from_answers=True) -> LengthStratifiedGrounding
-> signed_length_profile) used for the C-head and baseline.

Reuses score_head.py's rows_from_eval_jsonl() + report() verbatim so scoring is apples-to-apples.

Run from the repo ROOT with the main venv::

  env -u VIRTUAL_ENV uv run python research/TimeLens/score_lora.py \\
      --lora research/TimeLens/logs/lora_eval/reward1_final.jsonl \\
      --baseline research/TimeLens/logs/TimeLens-7B_20260615_085540/charades-timelens.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))  # so `import score_head` / `c_head.*` resolves from repo root

from score_head import report, overall  # noqa: E402
from benchmarks.length_profile import length_ratio_slope  # noqa: E402
from benchmarks.datasets.public.charades_timelens import (  # noqa: E402
    read_timelens_eval,
    CharadesTimeLensBenchmark,
)
from benchmarks.length_strata import LengthStratifiedGrounding  # noqa: E402


def _rows(path, from_answers):
    """Build length-stratified rows from a TimeLens eval JSONL.

    from_answers=True re-parses the 'X - Y seconds' answer text (full precision) — correct for the
    span-format baseline. from_answers=False reads the stored ``timestamps`` field — REQUIRED for a
    center-first (REPARAM_CENTER) adapter, whose answers ('centered at C ... lasts W') are NOT
    'X - Y' parseable (score_head.rows_from_eval_jsonl hardcodes from_answers=True and would hit the
    empty-parse fallback -> garbage). eval_lora.py already reconstructed [c-w/2, c+w/2] into
    ``timestamps``, so from_answers=False scores the true (full-precision) span.
    """
    samples, preds = read_timelens_eval(Path(path), from_answers=from_answers)
    return LengthStratifiedGrounding(CharadesTimeLensBenchmark()).to_long_table(preds, samples)


def _canon_key(raw_key: str) -> str | None:
    """Canonicalize a >>>-key to (vid_no_mp4, query, rounded-GT) so a float GT ([0.0,1.0]) and an
    int GT ([0,1]) for the same (video,query) match. The LoRA writer casts GT to float via _span();
    the baseline stored ints — same moment, different string. Match on the semantic identity."""
    parts = raw_key.split(">>>")
    if len(parts) < 3:
        return None
    vid = parts[0][:-4] if parts[0].endswith(".mp4") else parts[0]
    try:
        gt = json.loads(parts[2])
        gt_norm = (round(float(gt[0]), 3), round(float(gt[1]), 3))
    except Exception:
        return None
    return f"{vid}>>>{parts[1]}>>>{gt_norm}"


def _keys_in(path: Path) -> set[str]:
    """Canonical semantic keys present in a (JSONL) eval file."""
    out: set[str] = set()
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            ck = _canon_key(next(iter(json.loads(line))))
        except Exception:
            ck = None
        if ck is not None:
            out.add(ck)
    return out


def _restrict_jsonl(src: Path, keep: set[str]) -> Path:
    """Write a temp JSONL containing only the records whose CANONICAL key is in `keep`.

    Used to score the (full) baseline on the exact subset the partial LoRA covers, so per-bin
    n's align and the comparison is genuinely apples-to-apples on the identical (video,query) set.
    """
    tmp = Path(tempfile.mkstemp(suffix=".jsonl", prefix="baseline_matched_")[1])
    kept = 0
    with tmp.open("w", encoding="utf-8") as f:
        for line in src.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                ck = _canon_key(next(iter(json.loads(line))))
            except Exception:
                ck = None
            if ck is not None and ck in keep:
                f.write(line + "\n")
                kept += 1
    print(f"[match_keys] wrote {kept} baseline records matching the LoRA subset -> {tmp}")
    return tmp

# signed_length_profile bins (benchmarks/length_profile.DEFAULT_EDGES)
SHORT_BIN = "[0,2)"
S1_BINS = ("[2,3)", "[3,5)")
MID_BINS = ("[5,7)", "[7,10)", "[10,15)")
LONG_BIN = "[15+)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--lora", required=True, help="reward-1 LoRA predictions JSONL (merged shards)")
    ap.add_argument("--baseline", required=True, help="text-decode baseline JSONL")
    ap.add_argument("--label", default="reward-1 LoRA (final)")
    ap.add_argument("--match_keys", action="store_true",
                    help="Restrict the baseline to only the (video,query) keys present in --lora "
                         "so a PARTIAL LoRA run is scored apples-to-apples on the identical subset.")
    ap.add_argument("--lora_from_timestamps", action="store_true",
                    help="Score the LoRA from its stored `timestamps` field instead of re-parsing "
                         "the answer text. REQUIRED for center-first (REPARAM_CENTER) adapters whose "
                         "answers ('centered at C ... lasts W') are not 'X - Y seconds' parseable.")
    args = ap.parse_args()

    lora_path = Path(args.lora)
    base_path = Path(args.baseline)
    if args.match_keys:
        keep = _keys_in(lora_path)
        base_path = _restrict_jsonl(base_path, keep)
        print(f"[match_keys] baseline restricted to the {len(keep)} keys in {args.lora}")

    lora_rows = _rows(lora_path, from_answers=not args.lora_from_timestamps)
    base_rows = _rows(base_path, from_answers=True)

    lora_bins = report(args.label, lora_rows)
    base_bins = report("text-decode baseline", base_rows)

    # ---- side-by-side comparison table -------------------------------------------------
    print("\n" + "=" * 78)
    print("COMPARISON:  reward-1 LoRA  vs  text-decode baseline")
    print("=" * 78)
    hdr = f"{'bin':>8} | {'|dc| base':>9} {'|dc| LoRA':>9} {'d|dc|':>7} | " \
          f"{'R@.5 base':>9} {'R@.5 LoRA':>9} | {'R@.7 base':>9} {'R@.7 LoRA':>9} | {'n':>5}"
    print(hdr)
    print("-" * len(hdr))
    all_bins = [SHORT_BIN, *S1_BINS, *MID_BINS, LONG_BIN]
    for b in all_bins:
        L, B = lora_bins.get(b, {}), base_bins.get(b, {})
        dc_l = L.get("med_abs_center_err", float("nan"))
        dc_b = B.get("med_abs_center_err", float("nan"))
        ddc = dc_l - dc_b
        n = int(L.get("n", B.get("n", 0)) or 0)
        print(f"{b:>8} | {dc_b:>9.3f} {dc_l:>9.3f} {ddc:>7.3f} | "
              f"{B.get('R@1_IoU0.5', float('nan')):>9.3f} {L.get('R@1_IoU0.5', float('nan')):>9.3f} | "
              f"{B.get('R@1_IoU0.7', float('nan')):>9.3f} {L.get('R@1_IoU0.7', float('nan')):>9.3f} | {n:>5}")

    # ---- overall + beta ----------------------------------------------------------------
    ov_l, ov_b = overall(lora_rows), overall(base_rows)
    beta_l = length_ratio_slope(lora_rows, n_boot=500, seed=0)
    beta_b = length_ratio_slope(base_rows, n_boot=500, seed=0)
    print("\n  overall mIoU:   baseline={:.4f}   LoRA={:.4f}".format(ov_b["mIoU"], ov_l["mIoU"]))
    print("  overall R@0.5:  baseline={:.4f}   LoRA={:.4f}".format(ov_b["R@0.5"], ov_l["R@0.5"]))
    print("  overall R@0.7:  baseline={:.4f}   LoRA={:.4f}".format(ov_b["R@0.7"], ov_l["R@0.7"]))
    print("  beta(L_pred~L_gt): baseline={:.3f}   LoRA={:.3f}".format(
        beta_b.get("beta", float("nan")), beta_l.get("beta", float("nan"))))

    # ---- verdict ------------------------------------------------------------------------
    print("\n" + "=" * 78)
    print("VERDICT")
    print("=" * 78)
    s_l = lora_bins.get(SHORT_BIN, {})
    s_b = base_bins.get(SHORT_BIN, {})
    dc_l = s_l.get("med_abs_center_err", float("nan"))
    dc_b = s_b.get("med_abs_center_err", float("nan"))
    r07_l = s_l.get("R@1_IoU0.7", float("nan"))
    r07_b = s_b.get("R@1_IoU0.7", float("nan"))
    print(f"  <2s |Δc|:   baseline {dc_b:.3f}s -> LoRA {dc_l:.3f}s   (delta {dc_l - dc_b:+.3f}s)")
    print(f"  <2s R@0.7:  baseline {r07_b:.3f}  -> LoRA {r07_l:.3f}   (delta {r07_l - r07_b:+.3f})")
    # mid/long regression guard
    regressions = []
    for b in (*MID_BINS, LONG_BIN):
        L, B = lora_bins.get(b, {}), base_bins.get(b, {})
        if not L or not B:
            continue
        d_miou = L.get("mIoU", 0) - B.get("mIoU", 0)
        if d_miou < -0.01:  # >1 IoU-point drop
            regressions.append((b, d_miou))
    if regressions:
        print("  mid/long regression: " + ", ".join(f"{b} dmIoU={d:+.3f}" for b, d in regressions))
    else:
        print("  mid/long regression: NONE (no bin lost >0.01 mIoU)")


if __name__ == "__main__":
    main()
