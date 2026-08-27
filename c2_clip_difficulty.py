"""NS-P1 Step 2 — a stronger c2 ("query difficulty") confound than word count, then re-run §5.

The §5 deconfound used c2 = query word count (a crude proxy; the code flags a better one as deferred).
This computes the deferred-style proxy: a **frozen-retriever (CLIP) span-spread** difficulty that is
independent of the grounder under test. For each (video, query): sample frames, get CLIP frame-vs-query
cosine similarities, softmax over frames, and take the **softmax-weighted std of the frame timestamps,
normalized by duration**. A query whose visual match is *temporally diffuse* (spread across the whole
video) is genuinely harder to localize than one with a *sharp peak* -> higher spread = harder.

We then re-fit the SAME §5 estimators (GEE / MixedLM / CEM) swapping this in for q_difficulty, and check
the short-moment effect still survives the stronger confound. Run with the TimeLens venv (CLIP+decord):
  env -u VIRTUAL_ENV CUDA_VISIBLE_DEVICES=1 .venv/bin/python c2_clip_difficulty.py
"""
import json
import os
import sys

import numpy as np
import torch

REPO = "/home/mobeen/codes/active_repos/NeedleScan"
sys.path.insert(0, REPO)
sys.path.insert(0, f"{REPO}/research/TimeLens")

from benchmarks.datasets.public.charades_timelens import read_timelens_eval, CharadesTimeLensBenchmark
from benchmarks.length_strata import LengthStratifiedGrounding
from benchmarks.confounds import attach_confounds
from benchmarks import deconfound

BASELINE = f"{REPO}/research/TimeLens/logs/TimeLens-7B_20260615_085540/charades-timelens.jsonl"
VIDROOT = f"{REPO}/research/TimeLens/data/TimeLens-Bench/videos/charades"
CACHE = f"{REPO}/research/TimeLens/logs/c2_clip_difficulty.json"
N_FRAMES = 32
TEMP = 0.01  # softmax temperature on cosine sims (sharp)


def compute_clip_difficulty(samples):
    """Return {(video_id, query): normalized temporal-spread difficulty}. Cached to disk."""
    if os.path.exists(CACHE):
        raw = json.load(open(CACHE))
        return {tuple(k.split("\t")): v for k, v in raw.items()}
    import decord
    from transformers import CLIPModel, CLIPProcessor
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(dev).eval()
    proc = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    by_vid = {}
    for s in samples:
        by_vid.setdefault(s.video_id, {"dur": float(s.duration or 0), "queries": []})
        by_vid[s.video_id]["queries"].append(s.query)

    out = {}
    vids = list(by_vid)
    for i, vid in enumerate(vids):
        vp = f"{VIDROOT}/{vid}.mp4"
        info = by_vid[vid]
        if not os.path.exists(vp) or info["dur"] <= 0:
            for q in info["queries"]:
                out[(vid, q)] = None
            continue
        try:
            vr = decord.VideoReader(vp, num_threads=2)
            n = len(vr)
            idx = np.linspace(0, n - 1, min(N_FRAMES, n)).round().astype(int)
            fps = vr.get_avg_fps() or (n / info["dur"])
            ts = idx / fps
            frames = vr.get_batch(list(idx)).asnumpy()
            with torch.no_grad():
                pix = proc(images=list(frames), return_tensors="pt").to(dev)
                fe = model.get_image_features(**pix)
                fe = fe / fe.norm(dim=-1, keepdim=True)
                qs = info["queries"]
                txt = proc(text=qs, return_tensors="pt", padding=True, truncation=True).to(dev)
                te = model.get_text_features(**txt)
                te = te / te.norm(dim=-1, keepdim=True)
                sims = (te @ fe.T).float().cpu().numpy()
            for q, sim in zip(qs, sims):
                w = np.exp((sim - sim.max()) / TEMP)
                w = w / w.sum()
                mean_t = float((w * ts).sum())
                std_t = float(np.sqrt((w * (ts - mean_t) ** 2).sum()))
                out[(vid, q)] = std_t / info["dur"]
        except Exception:  # noqa: BLE001
            for q in info["queries"]:
                out[(vid, q)] = None
        if (i + 1) % 200 == 0:
            print(f"  CLIP difficulty: {i+1}/{len(vids)} videos", flush=True)
    json.dump({f"{k[0]}\t{k[1]}": v for k, v in out.items()}, open(CACHE, "w"))
    return out


def main():
    samples, preds = read_timelens_eval(BASELINE, from_answers=True)
    rows = LengthStratifiedGrounding(CharadesTimeLensBenchmark()).to_long_table(preds, samples)
    attach_confounds(rows, samples)
    for r in rows:
        r["q_difficulty_wc"] = r["q_difficulty"]
    diff = compute_clip_difficulty(samples)
    matched = 0
    for r in rows:
        d = diff.get((r["video_id"], r.get("query")))
        r["q_difficulty_clip"] = d
        if d is not None:
            matched += 1
    print(f"CLIP difficulty attached to {matched}/{len(rows)} rows")

    def run(label, col):
        rs = [dict(r, q_difficulty=r[col]) for r in rows if r.get(col) is not None]
        gee = deconfound.clustered_short_effect(rs)
        mlm = deconfound.mixedlm_iou_effect(rs)
        cem = deconfound.cem_short_effect(rs)
        print(f"\n===== c2 = {label}  (n={len(rs)}) =====")
        print(f"  GEE     : {gee}")
        print(f"  MixedLM : {mlm}")
        print(f"  CEM     : {cem}")

    run("WORD COUNT (original §5)", "q_difficulty_wc")
    run("CLIP span-spread (stronger)", "q_difficulty_clip")
    pairs = [(r["q_difficulty_wc"], r["q_difficulty_clip"]) for r in rows if r.get("q_difficulty_clip") is not None]
    a = np.asarray(pairs, float)
    print(f"\ncorr(word_count, clip_spread) = {np.corrcoef(a[:,0], a[:,1])[0,1]:.3f}  (low => independent signal)")


if __name__ == "__main__":
    main()
