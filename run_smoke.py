# Smallest-viable TimeLens grounding smoke test on Charades-TimeLens.
# Reuses the official GroundingDataset + extract_time path; limits to --limit clips.
# NOT a full eval. Run from the TimeLens repo root with PYTHONPATH="./".
import argparse
import json
import os

import nncore
import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluation.utils import GroundingDataset
from timelens.dataset.timelens_data import DATASET_DICT
from timelens.utils import extract_time


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument("--dataset", default="charades-timelens")
    p.add_argument("--limit", type=int, default=5)
    p.add_argument("--out", default="logs/smoke_charades.json")
    p.add_argument("--min_tokens", type=int, default=64)
    p.add_argument("--total_tokens", type=int, default=14336)
    p.add_argument("--fps", type=int, default=2)
    return p.parse_args()


def main():
    args = parse_args()
    args.split = "test"

    print(f"Loading model: {args.model_path}")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model_path,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",  # flash-attn .so ABI-mismatched vs torch 2.9; sdpa for B200 repro
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model_path,
        padding_side="left",
        do_resize=False,
        trust_remote_code=True,
    )

    dataset_class = DATASET_DICT[args.dataset]
    annos = dataset_class.load_annos(split="test")
    # keep only clips whose video file actually exists, take first --limit
    video_root = dataset_class.VIDEO_ROOT
    kept = []
    for a in annos:
        if os.path.exists(a["video_path"]):
            kept.append(a)
        if len(kept) >= args.limit:
            break
    print(f"Total annos={len(annos)}; running on {len(kept)} clips with existing videos "
          f"(video_root={video_root})")
    if not kept:
        raise SystemExit("No matching video files found on disk; cannot run inference.")

    ds = GroundingDataset(kept, processor, args)
    results = {}
    for i in range(len(ds)):
        data = ds[i]
        anno = data["anno"]
        inputs = data["inputs"].to("cuda")
        out_ids = model.generate(
            **inputs, do_sample=False, temperature=None, top_p=None, top_k=None,
            max_new_tokens=512,
        )
        trimmed = [o[len(j):] for j, o in zip(inputs.input_ids, out_ids)]
        answer = processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]
        ts = extract_time(answer)
        unit = getattr(dataset_class, "UNIT", 1.0)
        ts = [[round(s / unit) * unit, round(e / unit) * unit] for s, e in ts]
        span = anno["span"]
        if isinstance(span[0], (list, tuple)):
            span = span[0]
        vid = os.path.basename(anno["video_path"])
        key = f"{vid}>>>{anno['query']}>>>{span}"
        results[key] = {"timestamps": ts, "answers": answer, "duration": anno["duration"]}
        print(f"[{i+1}/{len(ds)}] {vid} | q='{anno['query']}' | gt={span} | "
              f"pred={ts} | answer='{answer}'")

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved {len(results)} predictions -> {args.out}")


if __name__ == "__main__":
    main()
