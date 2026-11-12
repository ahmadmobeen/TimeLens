"""Study X1 (NeedleScan METHODS-LOG 2026-11-12, wave 2, registered in 9e08ae6): what each frame
selector costs, on one B200 that runs nothing else (the CPU is shared with evaluations on the
other GPUs of the node).

  index  frames embedded per second and peak GPU memory of each scorer's frame encoder, on
         LVBench's cached 1 fps frames through the DataLoader of the real index build
         (prior_encode.py), and the index size per video-hour. Registered: Qwen3-VL-Embedding-8B
         and -2B and SigLIP so400m. Added for the cost-per-question curve: CLIP ViT-L/14 and BLIP
         ITM large, the scorers of BOLT and AKS as released.
  score  per question, the time to score every second of its video from the index, on the GPU:
         CaliFrame (8B, one text embedding per option, max over the options), GMM-EVA's selection
         (8B, one query with every option), BOLT (CLIP, the question), AKS (BLIP image-text
         matching of the question against every second's vision tokens). BLIP's matching runs on
         random tokens of the real shape, made outside the timed call: its cost does not depend
         on their values. The questions are those of the --timing runs of eval_lvbench_foveate.py,
         whose rows give the time to place and decode the frames (t_read) and to answer (t_answer).

  CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u nsp5_x1_cost.py index
  CUDA_VISIBLE_DEVICES=3 .venv/bin/python -u nsp5_x1_cost.py score --rows logs/nsp5/x1cost_uniform.jsonl

Each mode writes logs/nsp5/x1_cost_<mode>.json.
"""
from __future__ import annotations

import argparse
import json
import os
import statistics
import time
from typing import Callable

import numpy as np
import torch
from torch.utils.data import DataLoader

from prior_encode import (CHUNK, QUERY_INSTRUCTION, Chunks, is_qwen3_emb, load_qwen3_emb,
                          qwen3_conversation, qwen3_embed, qwen3_inputs, query_forms)
from t1_data import lvbench_items
from t1_decode import OUT as FRAMES

SCORERS = {"qwen3vl_emb_8b": "Qwen/Qwen3-VL-Embedding-8B",
           "qwen3vl_emb_2b": "Qwen/Qwen3-VL-Embedding-2B",
           "siglip_so400m": "google/siglip-so400m-patch14-384",
           "clip_l14": "openai/clip-vit-large-patch14",
           "blip_itm_l": "Salesforce/blip-itm-large-coco"}
PRIORS = "data/t1/priors"
OUT = "logs/nsp5/x1_cost_{}.json"
N_FRAMES = 2048          # timed frames per encoder, after one untimed warm-up chunk
WORKERS = 4              # the /dev/shm-safe setting (two 24-worker encoders once hung the node)
BLIP_ENC = 30523         # LAVIS's [ENC] token, as nsp5_extra_curves.run_blip sets it
BLIP_TOKENS = (577, 1024)  # BLIP ITM large: vision tokens per frame and their width


def load(name: str):
    """(model, processor, the processor Chunks hands the images to)."""
    mid = SCORERS[name]
    if is_qwen3_emb(mid):
        model, proc = load_qwen3_emb(mid)
        return model, proc, proc
    if name == "blip_itm_l":
        from transformers import BlipForImageTextRetrieval, BlipProcessor
        proc = BlipProcessor.from_pretrained(mid)
        model = BlipForImageTextRetrieval.from_pretrained(mid, dtype=torch.float16)
        return model.to("cuda").eval(), proc, proc.image_processor
    from transformers import AutoModel, AutoProcessor
    proc = AutoProcessor.from_pretrained(mid)
    model = AutoModel.from_pretrained(mid, dtype=torch.float16).to("cuda").eval()
    return model, proc, proc


@torch.no_grad()
def encode(name: str, model, px) -> torch.Tensor:
    if is_qwen3_emb(SCORERS[name]):
        return qwen3_embed(model, px)
    px = px.to("cuda", torch.float16)
    if name == "blip_itm_l":
        return model.vision_model(pixel_values=px).last_hidden_state
    return model.get_image_features(pixel_values=px)


def n_seconds(vid: str) -> int:
    with open(os.path.join(FRAMES, "lvbench", vid + ".json"), encoding="utf-8") as fh:
        return len(json.load(fh)["t"])


def lvbench_videos(n_frames: int) -> list[tuple[str, int]]:
    """LVBench videos in question order, cut so that their cached frames add up to n_frames."""
    vids, total = [], 0
    for vid in dict.fromkeys(it["key"] for it in lvbench_items()):
        if not os.path.exists(os.path.join(FRAMES, "lvbench", vid + ".json")):
            continue
        take = min(n_seconds(vid), n_frames - total)
        vids.append((vid, take))
        total += take
        if total >= n_frames:
            break
    return vids


def index_cost(name: str) -> dict:
    model, _, img_proc = load(name)
    torch.cuda.synchronize()
    weights = torch.cuda.memory_allocated()
    torch.cuda.reset_peak_memory_stats()
    q3 = is_qwen3_emb(SCORERS[name])
    chunk = 32 if q3 else CHUNK
    data = Chunks(lvbench_videos(N_FRAMES + chunk), img_proc, SCORERS[name] if q3 else "",
                  chunk=chunk, bench="lvbench")
    batches = iter(DataLoader(data, batch_size=None, num_workers=WORKERS, prefetch_factor=2))
    px, _ = next(batches)
    e = encode(name, model, px)                                    # warm-up chunk, not timed
    torch.cuda.synchronize()
    frames, t_gpu, t0 = 0, 0.0, time.perf_counter()
    for px, (_, a, b) in batches:
        g0 = time.perf_counter()
        e = encode(name, model, px)
        torch.cuda.synchronize()
        t_gpu += time.perf_counter() - g0
        frames += b - a
    wall = time.perf_counter() - t0
    per_frame = int(np.prod(e.shape[1:])) * 2                      # bytes at fp16, as stored
    out = {"model": SCORERS[name], "frames": frames, "frames_per_s": frames / wall,
           "frames_per_s_gpu_only": frames / t_gpu, "weights_gib": weights / 2**30,
           "peak_gib": torch.cuda.max_memory_allocated() / 2**30,
           "index_mb_per_video_hour": per_frame * 3600 / 1e6, "per_frame_shape": list(e.shape[1:])}
    if name == "blip_itm_l":
        out["note"] = ("vision tokens a matching model would keep per second; AKS as released "
                       "keeps none and runs the vision model again for every question")
    del model
    torch.cuda.empty_cache()
    return out


def summary(ts: list[float]) -> dict:
    return {"mean_s": statistics.fmean(ts), "median_s": statistics.median(ts), "max_s": max(ts),
            "per_question_s": [round(t, 5) for t in ts]}


def time_questions(label: str, qs: list[dict], one: Callable, prepare: Callable) -> dict:
    """Seconds per question for one(q) on the GPU; prepare(q) (untimed) loads the video's index."""
    prepare(qs[0])
    one(qs[0])                                                     # warm-up, not timed
    ts = []
    for q in qs:
        prepare(q)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        one(q)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    s = summary(ts)
    print(f"{label:28s} mean {s['mean_s'] * 1e3:8.1f} ms  median {s['median_s'] * 1e3:8.1f} ms",
          flush=True)
    return s


class VideoIndex:
    """The index of one video at a time on the GPU, loaded outside the timed call."""

    def __init__(self, scorer: str):
        self.path, self.vid, self.x = os.path.join(PRIORS, scorer, "lvbench_img.npz"), None, None

    def prepare(self, q: dict) -> None:
        if q["key"] != self.vid:
            with np.load(self.path) as z:
                self.x = torch.from_numpy(z[q["key"]].astype(np.float32)).cuda()
            self.vid = q["key"]


@torch.no_grad()
def score_cost(rows: str) -> dict:
    with open(rows, encoding="utf-8") as fh:
        keys = [json.loads(line)["key"] for line in fh]
    by_key = {f"{it['key']}>>>{it['uid']}": it for it in lvbench_items()}
    qs = [by_key[k] for k in keys if k in by_key]
    out = {"questions": len(qs), "not_found": len(keys) - len(qs), "videos": len({q["key"] for q in qs}),
           "seconds_per_question_video": [n_seconds(q["key"]) for q in qs]}

    model, proc, _ = load("qwen3vl_emb_8b")
    ix = VideoIndex("qwen3vl_emb_8b")

    def califrame(q: dict) -> torch.Tensor:
        stem, opts = query_forms(q["question"])
        convs = [qwen3_conversation(QUERY_INSTRUCTION, text=f"{stem} {o}") for o in opts]
        e = qwen3_embed(model, dict(qwen3_inputs(proc, convs)))
        return (ix.x @ e.T * 100.0).max(1).values.cpu()

    def gmm_eva(q: dict) -> torch.Tensor:
        conv = [qwen3_conversation(QUERY_INSTRUCTION, text=q["question"])]
        e = qwen3_embed(model, dict(qwen3_inputs(proc, conv)))
        return (ix.x @ e[0] * 100.0).cpu()

    out["califrame_8b_per_option"] = time_questions("califrame_8b_per_option", qs, califrame,
                                                    ix.prepare)
    out["gmm_eva_selection_8b_concat"] = time_questions("gmm_eva_selection_8b_concat", qs,
                                                        gmm_eva, ix.prepare)
    del model
    torch.cuda.empty_cache()

    model, proc, _ = load("clip_l14")
    scale = float(model.logit_scale.exp())
    ix = VideoIndex("clip_l14")

    def bolt(q: dict) -> torch.Tensor:
        stem, _ = query_forms(q["question"])
        t = proc(text=[stem], padding="max_length", truncation=True, max_length=64,
                 return_tensors="pt")
        e = torch.nn.functional.normalize(
            model.get_text_features(input_ids=t["input_ids"].to("cuda")).float(), dim=-1)
        return (ix.x @ e[0] * scale).cpu()

    out["bolt_clip_question"] = time_questions("bolt_clip_question", qs, bolt, ix.prepare)
    del model
    torch.cuda.empty_cache()

    from nsp5_extra_curves import pre_caption
    model, proc, _ = load("blip_itm_l")
    tokens = {"vid": None, "x": None}

    def blip_prepare(q: dict) -> None:
        if q["key"] != tokens["vid"]:
            tokens["x"] = None
            torch.cuda.empty_cache()
            tokens["x"] = torch.randn(n_seconds(q["key"]), *BLIP_TOKENS, dtype=torch.float16,
                                      device="cuda")
            tokens["vid"] = q["key"]

    def aks(q: dict) -> torch.Tensor:
        x = tokens["x"]
        stem, _ = query_forms(q["question"])
        tok = proc.tokenizer(pre_caption(stem), truncation=True, max_length=35,
                             return_tensors="pt").to("cuda")
        ids = tok.input_ids.clone()
        ids[:, 0] = BLIP_ENC
        probs = []
        for a in range(0, len(x), 512):
            b = min(a + 512, len(x))
            h = model.text_encoder(input_ids=ids.expand(b - a, -1),
                                   attention_mask=tok.attention_mask.expand(b - a, -1),
                                   encoder_hidden_states=x[a:b],
                                   encoder_attention_mask=torch.ones(x[a:b].shape[:-1],
                                                                     dtype=torch.long,
                                                                     device="cuda"))
            probs.append(model.itm_head(h.last_hidden_state[:, 0, :]).float().softmax(-1)[:, 1])
        return torch.cat(probs).cpu()

    out["aks_blip_matching_question"] = time_questions("aks_blip_matching_question", qs, aks,
                                                       blip_prepare)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["index", "score"])
    ap.add_argument("--rows", default="logs/nsp5/x1cost_uniform.jsonl",
                    help="score: the --timing rows whose questions to score")
    args = ap.parse_args()
    report = {"gpu": torch.cuda.get_device_name(0)}
    if args.mode == "index":
        for name in SCORERS:
            r = report[name] = index_cost(name)
            print(f"{name:16s} {r['frames_per_s']:8.1f} frames/s ({r['frames_per_s_gpu_only']:8.1f} "
                  f"GPU only)  peak {r['peak_gib']:6.1f} GiB  index "
                  f"{r['index_mb_per_video_hour']:9.1f} MB per video-hour", flush=True)
    else:
        report.update(score_cost(args.rows))
    with open(OUT.format(args.mode), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", OUT.format(args.mode), flush=True)


if __name__ == "__main__":
    main()
