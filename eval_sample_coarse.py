"""NS-P1 Stage-2 gate diagnostic: K-sample coarse decodes for placement-recall@K.

For each Charades S0 clip, build the OFFICIAL eval inputs (GroundingDataset) and draw
K sampled decodes in one generate call (do_sample, temperature=1.0, top_p=0.95,
num_return_sequences=K). Stores all K extracted spans per clip; scoring is offline.

Run (TimeLens venv, from research/TimeLens):
  env -u VIRTUAL_ENV CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 PYTHONPATH=. \\
    taskset -c 112-223 .venv/bin/python -u eval_sample_coarse.py \\
    --out logs/foveate_s2gate/samples_k5.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from timelens.dataset.timelens_data import DATASET_DICT
from evaluation.utils import GroundingDataset
from timelens.utils import extract_time

MODEL = "TencentARC/TimeLens-7B"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--top_p", type=float, default=0.95)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--keys_file", default=None,
                    help="JSON list of clip keys to restrict to (far-tail probes)")
    args_cli = ap.parse_args()

    raw = DATASET_DICT["charades-timelens"].load_annos(split="test")
    def span_of(a):
        sp = a["span"][0] if isinstance(a["span"][0], (list, tuple)) else a["span"]
        return float(sp[0]), float(sp[1])
    s0 = [a for a in raw if os.path.exists(a["video_path"])
          and span_of(a)[1] - span_of(a)[0] < 2.0]
    if args_cli.keys_file:
        keep = set(json.load(open(args_cli.keys_file)))
        s0 = [a for a in s0
              if f"{os.path.basename(a['video_path'])}>>>{a['query']}>>>{list(span_of(a))}" in keep]
    if args_cli.limit:
        s0 = s0[: args_cli.limit]
    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)

    done = set()
    if os.path.exists(args_cli.out):
        for line in open(args_cli.out, encoding="utf-8"):
            if line.strip():
                try:
                    done.add(next(iter(json.loads(line))))
                except Exception:
                    pass
    print(f"[sample k={args_cli.k} T={args_cli.temperature}] {len(s0)} S0 clips, "
          f"resume {len(done)} -> {args_cli.out}", flush=True)

    processor = AutoProcessor.from_pretrained(MODEL, padding_side="left",
                                              do_resize=False, trust_remote_code=True)
    ds_args = SimpleNamespace(model_path=MODEL, split="test", min_tokens=64,
                              total_tokens=14336, fps=2, max_new_tokens=512)
    ds = GroundingDataset(s0, processor, ds_args)
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto").eval()

    n_done = skipped = 0
    with open(args_cli.out, "a", encoding="utf-8") as f:
        for i, a in enumerate(s0):
            sp = span_of(a)
            vid = os.path.basename(a["video_path"])
            key = f"{vid}>>>{a['query']}>>>{list(sp)}"
            if key in done:
                continue
            try:
                inputs = ds[i]["inputs"].to("cuda")
                out = model.generate(**inputs, do_sample=True,
                                     temperature=args_cli.temperature,
                                     top_p=args_cli.top_p,
                                     num_return_sequences=args_cli.k,
                                     max_new_tokens=512)
                plen = inputs.input_ids.shape[-1]
                answers = processor.batch_decode([o[plen:] for o in out],
                                                 skip_special_tokens=True)
                samples = [extract_time(ans) for ans in answers]
            except Exception as ex:
                skipped += 1
                print(f"  [skip {skipped}] {vid} | {ex!r}", flush=True)
                continue
            f.write(json.dumps({key: {"samples": samples, "answers": answers,
                                      "duration": a["duration"]}}) + "\n")
            f.flush()
            n_done += 1
            if n_done % 50 == 0:
                print(f"  [{n_done}/{len(s0)}] done (skipped {skipped})", flush=True)
    print(f"DONE: {n_done} clips, {skipped} skipped -> {args_cli.out}", flush=True)


if __name__ == "__main__":
    main()
