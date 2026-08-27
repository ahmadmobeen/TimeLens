"""Cache frozen TimeLens-7B states for the C boundary head — POST-ANSWER readout.

For each (video, query) it greedily generates the model's textual answer, then captures,
from that single generation pass, three frozen representations the head arms read:

  * ``h_ans``   [H]      last-layer state at the FINAL generated token (post-answer) —
                         the readout that actually encodes the moment the model decided
                         (arm 1 = MLP on this; arm 2 = this as the cross-attn query).
  * ``vis``     [P, H]   the video's last-layer visual-token states, avg-pooled to ``vis_pool``
                         (=256) tokens — keys/values for arm 2's cross-attention.
  * ``vis_mean`` [H]     mean over all visual-token states — arm 3 (pooled-MLP baseline).

The earlier PRE-generation readout (`*.shard*.pt`, no `.postans`) regressed to the dataset
mean and could not localize short moments (METHODS-LOG 2026-06-25); reading the state AFTER
the answer is decoded is the fix. Greedy decode => deterministic => still cache-once.

Run in the TimeLens venv from ``research/TimeLens`` with ``PYTHONPATH=.``::

  env -u VIRTUAL_ENV CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \\
      ./.venv/bin/python -m c_head.cache_hidden --split train_natural --shard 0 --num_shards 2
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from types import SimpleNamespace

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from c_head import config as C

SPLITS = ("train_natural", "train_speedaug", "eval")


def _norm_span(span):
    """Charades spans are ``[s, e]`` or ``[[s, e]]`` -> ``(float s, float e)``."""
    if isinstance(span[0], (list, tuple)):
        span = span[0]
    return float(span[0]), float(span[1])


def _anno(video_path: str, query: str, span, duration) -> dict:
    s, e = _norm_span(span)
    return {
        "video_path": video_path,
        "query": query,
        "gt_start": s,
        "gt_end": e,
        "duration": float(duration),
        "video_id": Path(video_path).stem,
    }


def load_annos(split: str) -> list[dict]:
    """Aligned annos {video_path, query, gt_start, gt_end, duration, video_id} for a split."""
    if split == "eval":
        from timelens.dataset.timelens_data import DATASET_DICT

        raw = DATASET_DICT["charades-timelens"].load_annos(split="test")
        return [_anno(a["video_path"], a["query"], a["span"], a["duration"]) for a in raw]

    path = C.TRAIN_NATURAL_JSON if split == "train_natural" else C.TRAIN_SPEEDAUG_JSON
    raw = json.loads(Path(path).read_text(encoding="utf-8"))
    return [_anno(r["video_path"], r["query"], r["span"], r["duration"]) for r in raw]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--split", required=True, choices=SPLITS)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0, help="cap clips for a smoke run; 0=all")
    ap.add_argument("--save_every", type=int, default=1000)  # postans cache is large; flush less often
    ap.add_argument("--log_every", type=int, default=25)
    ap.add_argument("--vis_pool", type=int, default=256, help="pooled visual tokens for the cross-attn arm")
    ap.add_argument("--max_new_tokens", type=int, default=64)
    ap.add_argument("--out_dir", default=str(C.CACHE_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.split}.postans" + (f"_limit{args.limit}" if args.limit else "")
    out_path = out_dir / f"{tag}.shard{args.shard}of{args.num_shards}.pt"

    annos = [a for a in load_annos(args.split) if os.path.exists(a["video_path"])]
    annos = annos[args.shard :: args.num_shards]  # deterministic interleaved shard
    if args.limit:
        annos = annos[: args.limit]
    print(f"[{tag}] shard {args.shard}/{args.num_shards}: {len(annos)} clips, "
          f"vis_pool={args.vis_pool} -> {out_path}")

    # Resume: preload any prior cache, skip its (video_id, query) keys.
    h_ans: list[torch.Tensor] = []
    vis: list[torch.Tensor] = []
    vis_mean: list[torch.Tensor] = []
    metas: list[dict] = []
    done: set[tuple[str, str]] = set()
    if out_path.exists():
        prev = torch.load(out_path, map_location="cpu")
        for j, m in enumerate(prev["meta"]):
            h_ans.append(prev["h_ans"][j])
            vis.append(prev["vis"][j])
            vis_mean.append(prev["vis_mean"][j])
            metas.append(m)
            done.add((m["video_id"], m["query"]))
        print(f"  resume: {len(metas)} already cached")

    print(f"Loading model {C.MODEL_PATH} ...")
    model = AutoModelForImageTextToText.from_pretrained(
        C.MODEL_PATH,
        dtype=torch.bfloat16,
        attn_implementation="sdpa",  # flash-attn .so ABI-mismatched vs torch 2.9 on B200
        device_map="auto",
    ).eval()
    processor = AutoProcessor.from_pretrained(
        C.MODEL_PATH, padding_side="left", do_resize=False, trust_remote_code=True
    )
    # Visual patches are tagged with the IMAGE token id in Qwen2.5-VL even for video
    # (confirmed: image_token_id 151655 fills the frames; video_token_id 151656 is absent).
    # Use the union of both to be robust across configs.
    vis_ids = [t for t in (getattr(model.config, "image_token_id", None),
                           getattr(model.config, "video_token_id", None)) if t is not None]
    if not vis_ids:
        vis_ids = [processor.tokenizer.convert_tokens_to_ids("<|image_pad|>")]
    vis_ids_t = torch.tensor(vis_ids, device="cuda")
    print(f"  visual token ids={vis_ids}")

    from evaluation.utils import GroundingDataset

    ds_args = SimpleNamespace(
        model_path=C.MODEL_PATH,
        min_tokens=C.MIN_TOKENS,
        total_tokens=C.TOTAL_TOKENS,
        fps=C.FPS,
        split="test",
    )
    ds = GroundingDataset(annos, processor, ds_args)

    def flush() -> None:
        torch.save(
            {
                "h_ans": torch.stack(h_ans) if h_ans else torch.empty(0, C.HIDDEN_SIZE, dtype=torch.float16),
                "vis": torch.stack(vis) if vis else torch.empty(0, args.vis_pool, C.HIDDEN_SIZE, dtype=torch.float16),
                "vis_mean": torch.stack(vis_mean) if vis_mean else torch.empty(0, C.HIDDEN_SIZE, dtype=torch.float16),
                "meta": metas,
                "vis_pool": args.vis_pool,
            },
            out_path,
        )

    skipped = 0
    n_run = 0  # clips cached THIS run (excludes resumed) — for the live rate
    t0 = time.time()
    for i in range(len(ds)):
        a = ds.annos[i]
        if (a["video_id"], a["query"]) in done:
            continue
        try:
            inputs = ds[i]["inputs"].to("cuda")
            prompt_ids = inputs["input_ids"][0]
            with torch.no_grad():
                gen = model.generate(
                    **inputs,
                    do_sample=False, temperature=None, top_p=None, top_k=None,
                    max_new_tokens=args.max_new_tokens,
                    return_dict_in_generate=True, output_hidden_states=True,
                )
            # post-answer: last-layer state of the FINAL generated token
            h_post = gen.hidden_states[-1][-1][0, -1, :].float()
            # visual tokens: last-layer PREFILL states at the video-placeholder positions
            prefill = gen.hidden_states[0][-1][0]  # [prompt_len, H]
            vtok = prefill[torch.isin(prompt_ids, vis_ids_t)].float()  # [n_vis, H]
            if vtok.shape[0] == 0:
                raise ValueError("no visual tokens found in prompt")
            pooled = torch.nn.functional.adaptive_avg_pool1d(
                vtok.transpose(0, 1).unsqueeze(0), args.vis_pool
            )[0].transpose(0, 1)  # [vis_pool, H]
            vmean = vtok.mean(0)
            ans_ids = gen.sequences[0][prompt_ids.shape[0]:]
            answer = processor.batch_decode([ans_ids], skip_special_tokens=True)[0]
        except Exception as ex:  # one bad clip must not kill a multi-hour cache
            skipped += 1
            print(f"  [skip {skipped}] {a['video_id']} | {ex!r}", flush=True)
            continue
        h_ans.append(h_post.half().cpu())
        vis.append(pooled.half().cpu())
        vis_mean.append(vmean.half().cpu())
        metas.append(
            {**{k: a[k] for k in ("video_id", "query", "gt_start", "gt_end", "duration")},
             "answer": answer}
        )
        n_run += 1
        if n_run % args.log_every == 0:
            rate = n_run / max(1e-6, time.time() - t0) * 60.0  # clips/min, this run
            eta = (len(annos) - len(metas)) / max(1e-6, rate)
            print(f"  [{len(metas)}/{len(annos)}] {rate:.1f} clips/min  ETA {eta:.0f} min  (skipped {skipped})", flush=True)
        if len(metas) % args.save_every == 0:
            flush()

    flush()
    print(f"DONE [{tag}] shard {args.shard}: {len(metas)} cached, {skipped} skipped -> {out_path}")


if __name__ == "__main__":
    main()
