"""NS-P1 reward-1 STEP 0 — the pre-GPU reward-variance kill-gate (no finetune).

Before spending any GPU on the LoRA-GRPO pilot, verify the center *process* reward actually has
exploitable variance on the <2s stratum. For each short Charades clip we draw G stochastic rollouts
and, within each group, compare:
  - std(tIoU)                 : the IoU reward's spread (our gradient audit says it VANISHES on
                                short moments -> IoU-based RL is blind there).
  - std(R_center, sigma=0.4s) : the center reward's spread (must be > 0 for RL to create signal).
  - |dc| of the argmax-R_center rollout vs the argmax-tIoU rollout: does selecting on center get
    closer to the true center than selecting on IoU (i.e. is the center reward distinct AND better)?

Verdict on the <2s bin (runbook-reward1-pilot.md step 0):
  KILL  if std(R_center) ~= 0  -> all rollouts identically-wrong center; exploration never samples a
                                  better center -> RL cannot create signal -> skip the pilot, go to
                                  loss-1 (SFT-based, needs no exploration).
  GREEN if std(tIoU) < 0.02 AND std(R_center) > 0.10 AND
         mean(|dc|_bestIoU - |dc|_bestCenter) >= 0.15s
                               -> IoU is flat, the center reward has spread AND points the right way.

Decode uses the FROZEN baseline recipe (c_head.config: FPS2/TOTAL14336/MIN64), identical to
fps_sweep, so the rollout distribution is the one GRPO would actually see. Metrics are inline (no
`benchmarks` dependency) so this runs in the TimeLens venv. Per-clip rollouts are written to JSONL
(resumable); `--aggregate` re-reads the shards and prints the verdict without any GPU.

Run (TimeLens venv, from research/TimeLens, PYTHONPATH=.), 2-GPU sharded:
  env -u VIRTUAL_ENV CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. OMP_NUM_THREADS=8 taskset -c 0-111 \\
    ./.venv/bin/python reward_gate.py --group 8 --limit 160 --shard 0 --num_shards 2 \\
      --out output/reward-1/gate/rollouts.shard0of2.jsonl
  (shard 1 on CUDA_VISIBLE_DEVICES=1, taskset -c 112-223)
Then aggregate (no GPU):
  ./.venv/bin/python reward_gate.py --aggregate --out output/reward-1/gate/rollouts.shard0of2.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
from statistics import pstdev
from types import SimpleNamespace

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from c_head import config as C
from timelens.dataset.timelens_data import DATASET_DICT
from timelens.utils import extract_time
from evaluation.utils import GroundingDataset

SIGMA = 0.4  # the reward-1 target center-reward width (precision floor)

# The gate sweeps sigma so a mis-scaled reward can't force a false KILL: on <2s moments predictions
# land several seconds off, where a sigma=0.4s Gaussian is a near-delta that reads ~0 for EVERY
# rollout (std(R_center)=0 regardless of real center spread). A wider sigma restores cold-start
# gradient. argmax-R_center == argmin|dc| for any sigma>0, so the argmax-gain is sigma-independent;
# only std(R_center) needs the grid.
SIGMA_GRID = (0.4, 1.0, 2.0)

# Verdict thresholds (runbook-reward1-pilot.md step 0), read off the aggregate [0,2) bin.
MIN_CENTER_SPREAD = 0.25  # mean std(center) seconds below this = rollouts ~static -> hard KILL
GREEN_IOU_STD = 0.02      # mean std(tIoU) below this (IoU flat = vanishing gradient confirmed)
GREEN_RC_STD = 0.10       # best-sigma mean std(R_center) must exceed this (reward has spread)
GREEN_DC_GAIN = 0.15      # argmax-R_center must beat argmax-tIoU by >= this on |dc| (seconds)

# sub-bins for texture + the aggregate [0,2) bin the verdict is read off.
BINS = (("[0,1)", 0.0, 1.0), ("[1,2)", 1.0, 2.0), ("[0,2)", 0.0, 2.0))


def _span(a: dict) -> tuple[float, float]:
    sp = a["span"][0] if isinstance(a["span"][0], (list, tuple)) else a["span"]
    return float(sp[0]), float(sp[1])


def _iou(p: tuple[float, float], g: tuple[float, float]) -> float:
    inter = max(0.0, min(p[1], g[1]) - max(p[0], g[0]))
    union = max(p[1], g[1]) - min(p[0], g[0])
    return inter / union if union > 0 else 0.0


def _center(sp: tuple[float, float]) -> float:
    return 0.5 * (sp[0] + sp[1])


def _r_center(c_pred: float, c_gt: float, sigma: float) -> float:
    return math.exp(-(((c_pred - c_gt) / sigma) ** 2))


def _parse_span(ans: str) -> tuple[float, float] | None:
    """extract_time -> first predicted (start,end) in seconds, or None if unparseable.

    extract_time returns a LIST of (start,end) tuples, e.g. [(0.0, 24.4)]; the first tuple is the
    prediction (same convention the baseline read_timelens_eval scores on).
    """
    try:
        ts = extract_time(ans)
    except Exception:
        return None
    if not ts:
        return None
    try:
        s, e = float(ts[0][0]), float(ts[0][1])
    except (TypeError, ValueError, IndexError):
        return None
    return (e, s) if e < s else (s, e)


def run_decode(args: argparse.Namespace) -> None:
    raw = DATASET_DICT["charades-timelens"].load_annos(split="test")
    annos = []
    for a in raw:
        s, e = _span(a)
        if (e - s) < args.max_moment_s and os.path.exists(a["video_path"]):
            annos.append(a)
    annos = annos[args.shard :: args.num_shards]
    if args.limit:
        annos = annos[: args.limit]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    done: set[str] = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    done.add(next(iter(json.loads(line))))
                except Exception:
                    pass
    print(f"[gate] {len(annos)} clips (<{args.max_moment_s}s) G={args.group} "
          f"temp={args.temperature} top_p={args.top_p}, resume {len(done)} -> {args.out}", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        C.MODEL_PATH, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(
        C.MODEL_PATH, padding_side="left", do_resize=False, trust_remote_code=True
    )
    ds_args = SimpleNamespace(model_path=C.MODEL_PATH, min_tokens=C.MIN_TOKENS,
                              total_tokens=C.TOTAL_TOKENS, fps=C.FPS, split="test")
    ds = GroundingDataset(annos, processor, ds_args)

    n = skipped = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for i in range(len(ds)):
            a = ds.annos[i]
            s, e = _span(a)
            vid = os.path.splitext(os.path.basename(a["video_path"]))[0]
            key = f"{vid}.mp4>>>{a['query']}>>>{[s, e]}"
            if key in done:
                continue
            try:
                inputs = ds[i]["inputs"].to("cuda")  # decord decode ONCE; reused across G rollouts
                prompt_len = inputs.input_ids.shape[1]
                torch.manual_seed(args.seed + i)     # reproducible-but-diverse group
                rollouts = []
                for _ in range(args.group):
                    out = model.generate(
                        **inputs, do_sample=True, temperature=args.temperature,
                        top_p=args.top_p, top_k=(args.top_k or None),
                        max_new_tokens=args.max_new_tokens,
                    )
                    cont = out[0][prompt_len:]
                    rollouts.append(processor.batch_decode([cont], skip_special_tokens=True)[0])
            except Exception as ex:  # one bad clip must not kill the run
                skipped += 1
                print(f"  [skip {skipped}] {vid} | {ex!r}", flush=True)
                continue
            f.write(json.dumps({key: {
                "gt": [s, e], "duration": float(a["duration"]), "answers": rollouts,
            }}) + "\n")
            f.flush()
            n += 1
            if n % 20 == 0:
                print(f"  [{n}/{len(annos)}] {args.group}-decoded (skipped {skipped})", flush=True)
    print(f"DONE shard {args.shard}: {n} clips, {skipped} skipped -> {args.out}", flush=True)


def _shard_paths(out: str) -> list[Path]:
    p = Path(out)
    if ".shard" in p.name:  # scoped glob in a known dir with a tight pattern (no fs discovery)
        prefix = p.name.split(".shard")[0]
        return sorted(p.parent.glob(f"{prefix}.shard*.jsonl"))
    return [p] if p.exists() else []


def _clip_metrics(v: dict) -> dict:
    gt = (float(v["gt"][0]), float(v["gt"][1]))
    c_gt = _center(gt)
    length = gt[1] - gt[0]
    ious, dcs, cs = [], [], []
    for ans in v["answers"]:
        sp = _parse_span(ans)
        if sp is None:
            continue
        c = _center(sp)
        ious.append(_iou(sp, gt))
        dcs.append(abs(c - c_gt))
        cs.append(c)
    n_valid = len(ious)
    if n_valid < 2:
        return {"length": length, "n_rollouts": len(v["answers"]), "n_valid": n_valid, "degenerate": True}
    bi_iou = max(range(n_valid), key=lambda k: ious[k])
    # argmax R_center == argmin |dc| for any sigma>0, so best-center selection is sigma-independent.
    return {
        "length": length, "n_rollouts": len(v["answers"]), "n_valid": n_valid, "degenerate": False,
        "std_iou": pstdev(ious), "std_center_s": pstdev(cs),
        "std_rcenter": {s: pstdev([_r_center(c, c_gt, s) for c in cs]) for s in SIGMA_GRID},
        "dc_best_iou": dcs[bi_iou], "dc_best_center": min(dcs), "mean_dc": sum(dcs) / n_valid,
    }


def aggregate(out: str) -> None:
    paths = _shard_paths(out)
    if not paths:
        raise FileNotFoundError(f"no rollout shards for {out}")
    per_clip = []
    n_bad = 0
    for path in paths:
        for line in open(path, encoding="utf-8"):
            line = line.strip()
            if not line:
                continue
            try:  # tolerate a torn final line if a shard is still being written
                _, v = next(iter(json.loads(line).items()))
            except (json.JSONDecodeError, StopIteration):
                n_bad += 1
                continue
            per_clip.append(_clip_metrics(v))
    if n_bad:
        print(f"[warn] skipped {n_bad} unparseable line(s) (shard still writing?)")

    n_valid_clips = sum(1 for c in per_clip if not c["degenerate"])
    rc_hdr = " ".join(f"std(Rc:{s})".rjust(11) for s in SIGMA_GRID)
    print(f"\n=== reward-variance gate: {len(per_clip)} clips ({n_valid_clips} usable) "
          f"across {len(paths)} shard(s) ===")
    print(f"  {'bin':>7} {'n':>4} {'std(tIoU)':>10} {'std(c)s':>8} {rc_hdr} "
          f"{'dc*IoU':>7} {'dc*ctr':>7} {'gain':>6} {'meanDc':>7}")

    report: dict[str, dict] = {}
    for name, lo, hi in BINS:
        valid = [c for c in per_clip if not c["degenerate"] and lo <= c["length"] < hi]
        if not valid:
            continue
        k = len(valid)
        m = {
            "n": k,
            "std_iou": sum(c["std_iou"] for c in valid) / k,
            "std_center_s": sum(c["std_center_s"] for c in valid) / k,
            "std_rcenter": {s: sum(c["std_rcenter"][s] for c in valid) / k for s in SIGMA_GRID},
            "dc_best_iou": sum(c["dc_best_iou"] for c in valid) / k,
            "dc_best_center": sum(c["dc_best_center"] for c in valid) / k,
            "mean_dc": sum(c["mean_dc"] for c in valid) / k,
        }
        m["gain"] = m["dc_best_iou"] - m["dc_best_center"]
        report[name] = m
        rc_cols = " ".join(f"{m['std_rcenter'][s]:>11.4f}" for s in SIGMA_GRID)
        print(f"  {name:>7} {k:>4} {m['std_iou']:>10.4f} {m['std_center_s']:>8.3f} {rc_cols} "
              f"{m['dc_best_iou']:>7.3f} {m['dc_best_center']:>7.3f} {m['gain']:>6.3f} {m['mean_dc']:>7.3f}")

    g = report.get("[0,2)")
    print("\n========== VERDICT (read off [0,2)) ==========")
    if not g:
        print("  no usable <2s clips — cannot decide. Check parse-rate / decode.")
        return
    best_sigma = max(SIGMA_GRID, key=lambda s: g["std_rcenter"][s])
    best_rc = g["std_rcenter"][best_sigma]
    explorable = g["std_center_s"] >= MIN_CENTER_SPREAD
    c_iou = g["std_iou"] < GREEN_IOU_STD
    c_rc = best_rc > GREEN_RC_STD
    c_gain = g["gain"] >= GREEN_DC_GAIN
    sigma_too_tight = g["std_rcenter"][SIGMA] <= GREEN_RC_STD < best_rc
    print(f"  explorable std(center) >= {MIN_CENTER_SPREAD}s : {g['std_center_s']:.3f}s -> {explorable}")
    print(f"  GREEN std(tIoU)     <  {GREEN_IOU_STD}       : {g['std_iou']:.4f}  -> {c_iou}")
    print(f"  GREEN std(R_center) >  {GREEN_RC_STD} @best sigma={best_sigma}: {best_rc:.4f} -> {c_rc}")
    print(f"  GREEN argmax-Rc gain>= {GREEN_DC_GAIN}s      : {g['gain']:.3f}s -> {c_gain}")
    if not explorable:
        verdict = ("KILL -> rollouts near-static (std(center) < %.2fs); no reward shaping can create "
                   "signal. Try a hotter decode; else pivot to loss-1 (SFT)." % MIN_CENTER_SPREAD)
    elif c_iou and c_rc and c_gain:
        note = (f"  [NOTE] target sigma={SIGMA}s is too tight for cold-start "
                f"(std(Rc)={g['std_rcenter'][SIGMA]:.3f}); needs a sigma-curriculum wide->narrow.\n"
                if sigma_too_tight else "")
        verdict = note + "GREEN -> greenlight the reward-1 LoRA-GRPO pilot (step 1)"
    else:
        verdict = "AMBIGUOUS -> inspect per-condition breakdown; default caution = start loss-1"
    print(f"\n  >>> {verdict}")

    gate_json = Path(out).with_name("gate_verdict.json")
    gate_json.write_text(json.dumps({
        "sigma_target": SIGMA, "sigma_grid": list(SIGMA_GRID), "best_sigma": best_sigma,
        "thresholds": {
            "min_center_spread": MIN_CENTER_SPREAD, "green_iou_std": GREEN_IOU_STD,
            "green_rc_std": GREEN_RC_STD, "green_dc_gain": GREEN_DC_GAIN,
        }, "bins": report, "verdict": verdict,
    }, indent=2))
    print(f"  [json] {gate_json}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="per-shard rollout JSONL (or any shard for --aggregate)")
    ap.add_argument("--aggregate", action="store_true", help="re-read shards + print verdict (no GPU)")
    ap.add_argument("--group", type=int, default=8, help="G stochastic rollouts per clip")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_moment_s", type=float, default=2.0)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--top_k", type=int, default=0, help="0 disables top-k (pure temp/top_p sampling)")
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.aggregate:
        aggregate(args.out)
    else:
        run_decode(args)


if __name__ == "__main__":
    main()
