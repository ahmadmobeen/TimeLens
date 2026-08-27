"""NS-P1 — fps-sweep on Charades <2s: does higher input frame-rate lower the center floor? (input-Nyquist test)

Decodes TimeLens-7B on the Charades-TimeLens <2s eval clips at a chosen --fps and writes the >>>-key
JSONL (scored later via read_timelens_eval + length_profile). Compare <2s |dc| across fps {2,4,8}:
- dropping toward sub-second  => the floor is INPUT-Nyquist (fix = foveal / adaptive fps, budget-1);
- flat                        => input resolution is not the bottleneck (fix = backbone finetune).
Charades ~30s videos are the CLEAN regime (token budget not starved, unlike ActivityNet ~125s in Exp 8).

Annos come from DATASET_DICT (which carries the correct video_path), so no --video_dir is needed.

Run (TimeLens venv, from research/TimeLens, PYTHONPATH=.):
  env -u VIRTUAL_ENV CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. ./.venv/bin/python -m c_head.fps_sweep \\
      --fps 4 --out logs/fps_sweep/charades_fps4.jsonl
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fps", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--max_moment_s", type=float, default=2.0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

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
    print(f"[fps={args.fps}] {len(annos)} clips (<{args.max_moment_s}s), resume {len(done)} -> {args.out}", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        C.MODEL_PATH, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(
        C.MODEL_PATH, padding_side="left", do_resize=False, trust_remote_code=True
    )
    ds_args = SimpleNamespace(model_path=C.MODEL_PATH, min_tokens=C.MIN_TOKENS,
                              total_tokens=C.TOTAL_TOKENS, fps=args.fps, split="test")
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
                inputs = ds[i]["inputs"].to("cuda")
                out = model.generate(**inputs, do_sample=False, temperature=None, top_p=None, top_k=None,
                                     max_new_tokens=args.max_new_tokens)
                trimmed = [o[len(j):] for j, o in zip(inputs.input_ids, out)]
                ans = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
                ts = extract_time(ans)
            except Exception as ex:
                skipped += 1
                print(f"  [skip {skipped}] {vid} | {ex!r}", flush=True)
                continue
            f.write(json.dumps({key: {"timestamps": ts, "answers": ans, "duration": float(a["duration"])}}) + "\n")
            f.flush()
            n += 1
            if n % 25 == 0:
                print(f"  [{n}/{len(annos)}] decoded (skipped {skipped})", flush=True)
    print(f"DONE fps={args.fps} shard {args.shard}: {n} preds, {skipped} skipped -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
