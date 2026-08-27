"""NS-P1 round-1 #A (true per-frame gate): cache UN-POOLED per-frame, multi-layer visual hiddens.

Resolves the pooling confound of `probe_softargmax.py` (which used vis[256], adaptive-avg-pooled
from ~14k tokens). Here we keep per-TEMPORAL-POSITION vectors: reshape the visual-token hiddens via
``video_grid_thw = [T, H, W]`` into ``[T, H*W, d]`` and mean-pool over the spatial axis -> ``[T, d]``,
at several LLM layers, plus the answer-aware ``h_ans``. A soft-argmax over the T temporal positions
then tests whether sub-second center is extractable from the frozen PER-FRAME representation (vs the
pooled RED). Frozen model, greedy decode => cache-once. Expensive per clip -> use --limit / --max_moment_s.

Run (TimeLens venv, from research/TimeLens, PYTHONPATH=.):
  env -u VIRTUAL_ENV CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. ./.venv/bin/python -m c_head.cache_perframe \\
      --split eval --limit 5          # smoke: validate the frame mapping
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from c_head import config as C
from c_head.cache_hidden import load_annos  # reuse the aligned-anno loader

# hidden_states tuple indices to keep (0 = embeddings .. 28 = last layer, for Qwen2.5-VL-7B's 28 LLM layers)
LAYERS = (14, 21, 28)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=("train_natural", "train_speedaug", "eval"))
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max_moment_s", type=float, default=0.0,
                    help=">0 keeps only clips whose GT moment is shorter than this (seconds)")
    ap.add_argument("--out_dir", default=str(C.CACHE_DIR))
    ap.add_argument("--log_every", type=int, default=25)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}.perframe" + (f"_lim{args.limit}" if args.limit else "")
    out_path = out_dir / f"{tag}.shard{args.shard}of{args.num_shards}.pt"

    annos = [a for a in load_annos(args.split) if os.path.exists(a["video_path"])]
    if args.max_moment_s > 0:
        annos = [a for a in annos if (a["gt_end"] - a["gt_start"]) < args.max_moment_s]
    annos = annos[args.shard :: args.num_shards]
    if args.limit:
        annos = annos[: args.limit]
    print(f"[{tag}] {len(annos)} clips (layers={LAYERS}) -> {out_path}", flush=True)

    model = AutoModelForImageTextToText.from_pretrained(
        C.MODEL_PATH, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(
        C.MODEL_PATH, padding_side="left", do_resize=False, trust_remote_code=True
    )
    vis_ids = [t for t in (getattr(model.config, "image_token_id", None),
                           getattr(model.config, "video_token_id", None)) if t is not None]
    vis_ids_t = torch.tensor(vis_ids, device="cuda")
    print(f"  visual token ids={vis_ids}", flush=True)

    from evaluation.utils import GroundingDataset

    ds_args = SimpleNamespace(model_path=C.MODEL_PATH, min_tokens=C.MIN_TOKENS,
                              total_tokens=C.TOTAL_TOKENS, fps=C.FPS, split="test")
    ds = GroundingDataset(annos, processor, ds_args)

    per_clip: list[dict] = []
    metas: list[dict] = []
    skipped = n = 0
    t0 = time.time()
    for i in range(len(ds)):
        a = ds.annos[i]
        try:
            inputs = ds[i]["inputs"].to("cuda")
            prompt_ids = inputs["input_ids"][0]
            grid = inputs.get("video_grid_thw", inputs.get("image_grid_thw"))
            if grid is None:
                raise ValueError("no video_grid_thw/image_grid_thw in inputs")
            # TimeLens feeds the video as N frames-as-IMAGES: image_grid_thw has ONE ROW PER FRAME,
            # each [1, H, W] (so grid[0][0] is 1, not the frame count). T = number of rows = frames;
            # tokens are frame-major, uniform H*W/4 per frame => spatial = n_vis // T.
            T = int(grid.shape[0])
            with torch.no_grad():
                gen = model.generate(
                    **inputs, do_sample=False, temperature=None, top_p=None, top_k=None,
                    max_new_tokens=64, return_dict_in_generate=True, output_hidden_states=True,
                )
            vmask = torch.isin(prompt_ids, vis_ids_t)
            n_vis = int(vmask.sum())
            if n_vis == 0 or T == 0 or n_vis % T != 0:
                raise ValueError(f"grid/token mismatch: n_vis={n_vis} T={T}")
            spatial = n_vis // T
            layers = {}
            for L in LAYERS:
                vtok = gen.hidden_states[0][L][0][vmask].float()   # [n_vis, d] last-prefill layer L
                layers[L] = vtok.view(T, spatial, -1).mean(1).half().cpu()  # [T, d]
            h_ans = gen.hidden_states[-1][-1][0, -1, :].half().cpu()
        except Exception as ex:  # one bad clip must not kill a long cache
            skipped += 1
            print(f"  [skip {skipped}] {a['video_id']} | {ex!r}", flush=True)
            continue
        per_clip.append({"layers": layers, "h_ans": h_ans, "T": T})
        metas.append({k: a[k] for k in ("video_id", "query", "gt_start", "gt_end", "duration")})
        n += 1
        if n % args.log_every == 0:
            rate = n / max(1e-6, time.time() - t0) * 60.0
            print(f"  [{n}/{len(annos)}] {rate:.1f}/min  T~{T} spatial~{spatial}  (skipped {skipped})", flush=True)

    torch.save({"per_clip": per_clip, "meta": metas, "layers": list(LAYERS)}, out_path)
    print(f"DONE [{tag}] shard {args.shard}: {n} cached, {skipped} skipped -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
