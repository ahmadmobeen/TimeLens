"""NS-P1 temporal-zoom probe: does cropping the decode to a window that contains a short moment
resolve the ~1s CENTER error? Frozen TimeLens-7B (no LoRA) — this is a training-free method.

For each <2s test clip we crop the video to a `[start, end]` window (via video_start/video_end, which
qwen_vl_utils samples within), re-decode at high fps, and measure |Δcenter| vs the 1.05s full-video
baseline. Two window sources:
  - mode=ceiling  : an oracle window that CONTAINS the GT moment, placed OFF-center (frac `--place`)
                    so within-crop localization is still non-trivial. Tests the ceiling: can zoom
                    resolve the center at all? (uses GT only to position the crop)
  - mode=realistic: window from the model's own coarse first pass (baseline pred center in
                    --coarse_jsonl) +/- window_len/2. The deployable two-pass method.

Output timestamps are ABSOLUTE original-video seconds (verified: frame_idx/fps, absolute idx), so the
model's output compares directly to GT with no offset. (Self-check: if |Δc| ~ gt_center magnitude,
outputs are window-relative and window_start must be added — flagged at run end.)

Writes the same >>>-key JSONL as eval_lora.py; score with score_lora.py --match_keys.

Run per-shard per-GPU from research/TimeLens/ (main model venv):
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-111 env -u VIRTUAL_ENV PYTHONPATH=. \\
    .venv/bin/python eval_zoom.py --mode ceiling --window_len 8 --fps 8 \\
    --out logs/lora_eval/zoom_ceiling_shard0.jsonl --shard 0 --num_shards 2
"""
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from c_head import config as C
from timelens.dataset.timelens_data import DATASET_DICT
from timelens.utils import extract_time
from evaluation.utils import GroundingDataset


def _span(a):
    sp = a["span"][0] if isinstance(a["span"][0], (list, tuple)) else a["span"]
    return float(sp[0]), float(sp[1])


def _norm_query(q):
    """Canonical query key: the coarse JSONL keeps trailing periods, DATASET_DICT strips them."""
    return " ".join(q.lower().rstrip(". ").split())


def _load_coarse_centers(path):
    """Map canonical (vid_no_mp4, query) -> predicted center (seconds) from a baseline eval JSONL."""
    out = {}
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if not line:
            continue
        try:
            d = json.loads(line)
            k, v = next(iter(d.items()))
            parts = k.split(">>>")
            vid = parts[0]
            for ext in (".mp4", ".mkv", ".webm", ".avi", ".mov"):  # ANet mixes containers
                if vid.endswith(ext):
                    vid = vid[: -len(ext)]
                    break
            ts = v.get("timestamps") or []
            if ts:
                s, e = float(ts[0][0]), float(ts[0][1])
                out[(vid, _norm_query(parts[1]))] = ((s + e) / 2.0, e - s)  # (coarse center, predicted length)
        except Exception:
            continue
    return out


def _window(a, mode, wlen, place, coarse, max_pred_len=None):
    """Return [start, end] crop window (seconds) for clip `a`, or None to skip.

    Router gate (realistic): if `max_pred_len` is set and the coarse PREDICTED length is >= it,
    return None — this clip is not routed to zoom (the length-adaptive router leaves
    long-predicted queries at their baseline prediction).
    """
    s, e = _span(a)
    dur = float(a["duration"])
    if mode == "ceiling":
        gc = (s + e) / 2.0
        start = gc - place * wlen            # moment center at fraction `place` of the window
    else:  # realistic
        vid = os.path.splitext(os.path.basename(a["video_path"]))[0]
        entry = coarse.get((vid, _norm_query(a["query"])))
        if entry is None:
            return None
        c0, pred_len = entry
        if max_pred_len is not None and pred_len >= max_pred_len:
            return None
        start = c0 - wlen / 2.0
    start = max(0.0, min(start, max(0.0, dur - wlen)))
    end = min(dur, start + wlen)
    return [round(start, 3), round(end, 3)]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--patch_embed", default="matmul", choices=["matmul", "conv3d"],
                    help="Qwen3-VL (8B) vision patch embedding. 'matmul' avoids the PyTorch 2.9 "
                         "Conv3d regression that dispatches to aten::slow_conv_dilated3d "
                         "(~1200x on long video; see fast_patch_embed.py and issue #82). "
                         "'conv3d' reproduces 8B zoom runs made before 2026-08-14. No effect "
                         "on 7B, whose patch size is 14 and which never hit the bug.")
    ap.add_argument("--mode", choices=["ceiling", "realistic"], default="ceiling")
    ap.add_argument("--dataset", default="charades-timelens",
                    help="DATASET_DICT key (charades-timelens | activitynet-omniembed | ...)")
    ap.add_argument("--model", default=C.MODEL_PATH,
                    help="model path/id (override C.MODEL_PATH to run TimeLens-8B / other grounders)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--window_len", type=float, default=8.0, help="crop window length (s)")
    ap.add_argument("--place", type=float, default=0.6, help="ceiling: moment-center fraction in window")
    ap.add_argument("--fps", type=int, default=8, help="sampling fps within the crop")
    ap.add_argument("--max_moment_s", type=float, default=2.0, help="only clips with GT length < this")
    ap.add_argument("--min_moment_s", type=float, default=0.0,
                    help="only clips with GT length >= this; with --max_moment_s this selects a "
                         "stratum. Default 0.0 leaves the historical short-stratum behaviour "
                         "byte-identical, so existing runs reproduce.")
    ap.add_argument("--coarse_jsonl", default=None, help="realistic: baseline eval JSONL for coarse centers")
    ap.add_argument("--max_pred_len", type=float, default=None,
                    help="router gate (realistic): only zoom clips whose coarse PREDICTED length < this (s)")
    ap.add_argument("--total_tokens", type=int, default=C.TOTAL_TOKENS,
                    help="crop token budget (default C.TOTAL_TOKENS); higher = more spatial res per frame")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    coarse = _load_coarse_centers(args.coarse_jsonl) if (args.mode == "realistic" and args.coarse_jsonl) else {}

    raw = DATASET_DICT[args.dataset].load_annos(split="test")
    annos = []
    for a in raw:
        if not os.path.exists(a["video_path"]):
            continue
        s, e = _span(a)
        if not (args.min_moment_s <= (e - s) < args.max_moment_s):   # stratum select
            continue
        annos.append(a)
    annos = annos[args.shard :: args.num_shards]
    print(f"[shard {args.shard}/{args.num_shards}] mode={args.mode} wlen={args.window_len} fps={args.fps} "
          f"place={args.place} -> {len(annos)} clips in "
          f"[{args.min_moment_s},{args.max_moment_s})s -> {args.out}", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    done = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    done.add(next(iter(json.loads(line))))
                except Exception:
                    pass

    attn = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except Exception as ex:
        attn = "sdpa"
        print(f"[warn] flash_attn import failed ({ex!r}) -> sdpa", flush=True)

    if args.patch_embed == "matmul":
        import fast_patch_embed
        fast_patch_embed.patch()

    print(f"Loading frozen base {args.model} (attn={attn}, patch_embed={args.patch_embed}) ...",
          flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=attn, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model, padding_side="left", do_resize=False, trust_remote_code=True
    )
    ds_args = SimpleNamespace(
        model_path=args.model, min_tokens=C.MIN_TOKENS, total_tokens=args.total_tokens,
        fps=args.fps, split="test",
    )

    # attach the crop window to each anno, then let GroundingDataset inject video_start/video_end.
    # Track window containment: does the crop actually contain the GT moment? On long videos
    # (ActivityNet ~99s) the coarse pass is far off, so a realistic ±wlen/2 window often MISSES —
    # this diagnostic tells us the hit-rate and thus how wide / iterative the window must be.
    zannos, skipped_win, n_full, n_partial = [], 0, 0, 0
    for a in annos:
        w = _window(a, args.mode, args.window_len, args.place, coarse, args.max_pred_len)
        if w is None:
            skipped_win += 1
            continue
        gs, ge = _span(a)
        if w[0] <= gs and ge <= w[1]:
            n_full += 1
        elif min(w[1], ge) - max(w[0], gs) > 0:
            n_partial += 1
        a = dict(a)
        a["video_start"], a["video_end"] = w[0], w[1]
        zannos.append(a)
    ds = GroundingDataset(zannos, processor, ds_args)
    n_miss = len(zannos) - n_full - n_partial
    print(f"  {len(zannos)} windowed clips ({skipped_win} skipped: no coarse center)", flush=True)
    print(f"  window containment: {n_full} full, {n_partial} partial, {n_miss} miss "
          f"(of {len(zannos)}) — moment inside the crop", flush=True)

    n = skipped = 0
    rel_flag = 0  # count of outputs that look window-relative (center far below window_start)
    with open(args.out, "a", encoding="utf-8") as f:
        for i in range(len(ds)):
            a = ds.annos[i]
            s, e = _span(a)
            dur = float(a["duration"])
            vid = os.path.splitext(os.path.basename(a["video_path"]))[0]
            key = f"{vid}.mp4>>>{a['query']}>>>{[s, e]}"
            if key in done:
                continue
            try:
                inputs = ds[i]["inputs"].to("cuda")
                with torch.no_grad():
                    out = model.generate(
                        **inputs, do_sample=False, temperature=None, top_p=None, top_k=None,
                        max_new_tokens=args.max_new_tokens,
                    )
                trimmed = [o[len(j):] for j, o in zip(inputs.input_ids, out)]
                ans = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
                ts = extract_time(ans)
            except Exception as ex:
                skipped += 1
                print(f"  [skip {skipped}] {vid} | {ex!r}", flush=True)
                continue
            ws = float(a["video_start"])
            if ts:
                pc = (float(ts[0][0]) + float(ts[0][1])) / 2.0
                if pc < ws - 0.5:  # predicted center below the window -> outputs may be window-relative
                    rel_flag += 1
            f.write(json.dumps({key: {
                "timestamps": ts, "answers": ans, "duration": dur,
                "window": [a["video_start"], a["video_end"]],
            }}) + "\n")
            f.flush()
            n += 1
            if n % 50 == 0:
                print(f"  [{n}/{len(zannos)}] decoded (skipped {skipped}, rel-looking {rel_flag})", flush=True)

    print(f"DONE shard {args.shard}: {n} preds, {skipped} skipped, {rel_flag} window-relative-looking "
          f"-> {args.out}", flush=True)
    if rel_flag > n * 0.3:
        print("  [!] many outputs below window_start — timestamps may be WINDOW-RELATIVE; "
              "add window_start before scoring.", flush=True)


if __name__ == "__main__":
    main()
