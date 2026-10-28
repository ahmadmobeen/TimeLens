"""NS-P5: score curves for the pre-registered prior-art arms (METHODS-LOG 2026-10-28).

  concat  F2C's query form: the Qwen3-VL-Embedding-8B scores every second against ONE query, the
          question with all its lettered options as the answer model sees them (100 x cosine,
          the 8B frame embeddings prior_encode.py cached). -> qwen3vl_emb_8b/<tag>curves_concat.npz
  blip    AKS's scorer: BLIP image-text matching, large (Salesforce/blip-itm-large-coco, the
          weights of LAVIS blip_image_text_matching large), match probability of the question
          alone after LAVIS's caption preprocessing, at every cached second; the first text token
          is [ENC] as LAVIS's ITM forward sets it. 100 x probability -> blip_itm_l/<tag>curves_stem.npz

BOLT's scorer (CLIP ViT-L/14, question only) is prior_encode.py --model
openai/clip-vit-large-patch14 --name clip_l14, whose curves_stem.npz is that curve.

On LVBench each mode also fits the Kelly temperature by prior_encode.nll_cv (5-fold by video,
the evidence spans) and writes it beside the curves (tau_concat.json, tau_stem.json); the arms
use that value on every benchmark.

  env -u VIRTUAL_ENV PYTHONPATH=. .venv/bin/python -u nsp5_extra_curves.py --mode concat --bench lvbench
"""
from __future__ import annotations

import argparse
import json
import os
import re

import numpy as np
import torch

from prior_encode import CHUNK, Chunks, nll_cv
from t1_data import lvbench_items, span_seconds

PRIORS = "data/t1/priors"
BLIP_ID = "Salesforce/blip-itm-large-coco"


def questions(bench: str) -> list[dict]:
    """key, video, stem, full text with lettered options, evidence spans (LVBench only)."""
    if bench == "lvbench":
        from prior_encode import query_forms
        out = []
        for it in lvbench_items():
            stem, _ = query_forms(it["question"])
            out.append({"key": f"lvbench:{it['key']}>>>{it['uid']}", "vid": it["key"], "stem": stem,
                        "full": it["question"], "spans": [tuple(it["span"])]})
        return out
    import bench as _bench
    return [{"key": q.key, "vid": q.vid, "stem": q.stem, "full": q.prompt_question(), "spans": []}
            for q in _bench.load(bench)]


def fit_tau(curves: dict[str, np.ndarray], qs: list[dict], path: str) -> None:
    zs, spans, vids = [], [], []
    for q in qs:
        if q["key"] in curves and q["spans"]:
            z = curves[q["key"]].astype(np.float64)
            zs.append(z)
            spans.append([span_seconds(a, b, len(z)) for a, b in q["spans"]])
            vids.append(q["vid"])
    tau, nll = nll_cv(zs, spans, vids)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"tau": round(tau, 4), "heldout_nll": round(nll, 4), "n": len(zs)}, fh)
    print(f"tau {tau:.4f}  held-out NLL {nll:.4f}  ({len(zs)} questions) -> {path}", flush=True)


def run_concat(bench: str, tag: str) -> dict[str, np.ndarray]:
    from prior_encode import (QUERY_INSTRUCTION, load_qwen3_emb, qwen3_conversation, qwen3_embed,
                              qwen3_inputs)
    qs = questions(bench)
    img = np.load(os.path.join(PRIORS, "qwen3vl_emb_8b", f"{bench}_img.npz"))
    have = set(img.files)
    model, proc = load_qwen3_emb("Qwen/Qwen3-VL-Embedding-8B")
    curves, frames = {}, {}
    for i in range(0, len(qs), 64):
        part = [q for q in qs[i:i + 64] if q["vid"] in have]
        if not part:
            continue
        convs = [qwen3_conversation(QUERY_INSTRUCTION, text=q["full"]) for q in part]
        emb = qwen3_embed(model, dict(qwen3_inputs(proc, convs))).cpu().numpy()
        for q, e in zip(part, emb):
            if q["vid"] not in frames:
                frames = {q["vid"]: img[q["vid"]].astype(np.float32)}   # questions come grouped by video
            curves[q["key"]] = frames[q["vid"]] @ e * 100.0
    path = os.path.join(PRIORS, "qwen3vl_emb_8b", f"{tag}curves_concat.npz")
    np.savez(path, **{k: v.astype(np.float16) for k, v in curves.items()})
    print(f"{len(curves)} of {len(qs)} questions -> {path}", flush=True)
    return curves


def pre_caption(text: str, max_words: int = 50) -> str:
    """LAVIS BlipCaptionProcessor.pre_caption, the eval text processor AKS uses."""
    text = re.sub(r"([.!\"()*#:;~])", " ", text.lower())
    text = re.sub(r"\s{2,}", " ", text).rstrip("\n").strip(" ")
    return " ".join(text.split(" ")[:max_words])


@torch.no_grad()
def run_blip(bench: str, tag: str, workers: int) -> dict[str, np.ndarray]:
    from torch.utils.data import DataLoader
    from transformers import BlipForImageTextRetrieval, BlipProcessor
    from t1_decode import OUT as FRAMES
    qs = questions(bench)
    by_vid: dict[str, list[dict]] = {}
    for q in qs:
        by_vid.setdefault(q["vid"], []).append(q)
    proc = BlipProcessor.from_pretrained(BLIP_ID)
    model = BlipForImageTextRetrieval.from_pretrained(BLIP_ID, dtype=torch.float16).to("cuda").eval()
    # LAVIS adds [DEC] then [ENC] to bert-base-uncased (30522 + 1); the HF tokenizer lacks them,
    # but the checkpoint's text vocabulary keeps both rows (30524).
    assert model.config.text_config.vocab_size == 30524, model.config.text_config.vocab_size
    enc = 30523
    vids = []
    for v in sorted(by_vid):
        meta = os.path.join(FRAMES, bench, v + ".json")
        if os.path.exists(meta):
            with open(meta, encoding="utf-8") as fh:
                vids.append((v, len(json.load(fh)["t"])))
    loader = DataLoader(Chunks(vids, proc.image_processor, "", chunk=CHUNK, bench=bench),
                        batch_size=None, num_workers=workers, prefetch_factor=2)
    curves, cur, parts = {}, None, []

    def score_video(vid: str, embeds: torch.Tensor) -> None:
        atts = torch.ones(embeds.shape[:-1], dtype=torch.long, device="cuda")
        for q in by_vid[vid]:
            tok = proc.tokenizer(pre_caption(q["stem"]), truncation=True, max_length=35,
                                 return_tensors="pt").to("cuda")
            ids = tok.input_ids.clone()
            ids[:, 0] = enc
            probs = []
            for a in range(0, len(embeds), 512):
                b = min(a + 512, len(embeds))
                out = model.text_encoder(input_ids=ids.expand(b - a, -1),
                                         attention_mask=tok.attention_mask.expand(b - a, -1),
                                         encoder_hidden_states=embeds[a:b],
                                         encoder_attention_mask=atts[a:b])
                logits = model.itm_head(out.last_hidden_state[:, 0, :]).float()
                probs.append(logits.softmax(-1)[:, 1].cpu())
            curves[q["key"]] = torch.cat(probs).numpy() * 100.0

    for px, (vid, a, b) in loader:
        if cur is not None and vid != cur:
            score_video(cur, torch.cat(parts))
            parts = []
        cur = vid
        parts.append(model.vision_model(pixel_values=px.to("cuda", torch.float16)).last_hidden_state)
    if cur is not None:
        score_video(cur, torch.cat(parts))
    out_dir = os.path.join(PRIORS, "blip_itm_l")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"{tag}curves_stem.npz")
    np.savez(path, **{k: v.astype(np.float16) for k, v in curves.items()})
    print(f"{len(curves)} of {len(qs)} questions -> {path}", flush=True)
    return curves


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["concat", "blip", "tau"],
                    help="tau: fit the Kelly temperature of an existing LVBench curves file")
    ap.add_argument("--curves", default="", help="tau: the LVBench curves .npz")
    ap.add_argument("--out", default="", help="tau: where to write the json")
    ap.add_argument("--bench", default="lvbench",
                    choices=["lvbench", "videomme", "lvb", "mlvu", "cgbench"])
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    tag = "" if args.bench == "lvbench" else f"{args.bench}_"
    if args.mode == "tau":
        with np.load(args.curves) as z:
            fit_tau({k: z[k] for k in z.files}, questions("lvbench"), args.out)
        return
    if args.mode == "concat":
        curves = run_concat(args.bench, tag)
        tau_path = os.path.join(PRIORS, "qwen3vl_emb_8b", "tau_concat.json")
    else:
        curves = run_blip(args.bench, tag, args.workers)
        tau_path = os.path.join(PRIORS, "blip_itm_l", "tau_stem.json")
    if args.bench == "lvbench":
        fit_tau(curves, questions("lvbench"), tau_path)


if __name__ == "__main__":
    main()
