"""Score LVBench's cached 1 fps frames with any HF image-text model, and compare priors.

The frame cache (t1_decode.py) makes swapping the encoder cheap: no video decoding.
For a model it writes per-second image embeddings for all 103 LVBench videos, then
the query-form curves T1/A2 use -- stem, and max over options of (stem + option_k) --
and reports the within-video localization metrics on the N3/T1 window protocol, plus
the 5-fold cross-validated posterior temperature and its held-out NLL.

  .venv/bin/python -u prior_encode.py --model google/siglip2-giant-opt-patch16-384 \\
      --name siglip2_giant
"""
from __future__ import annotations

import argparse
import io
import json
import os
import random
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from t1_data import lvbench_items, span_seconds
from t1_decode import OUT as FRAMES
from t1_train import dist_to_span, summarize, window_scores

CHUNK = 256


class Chunks(Dataset):
    def __init__(self, vids: list[tuple[str, int]], proc):
        self.items = [(v, a, min(a + CHUNK, n)) for v, n in vids for a in range(0, n, CHUNK)]
        self.proc = proc

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        vid, a, b = self.items[i]
        with open(os.path.join(FRAMES, "lvbench", vid + ".json"), encoding="utf-8") as fh:
            off = json.load(fh)["off"]
        with open(os.path.join(FRAMES, "lvbench", vid + ".bin"), "rb") as fh:
            fh.seek(off[a])
            blob = fh.read(off[b] - off[a])
        imgs = [Image.open(io.BytesIO(blob[off[k] - off[a]:off[k + 1] - off[a]])).convert("RGB")
                for k in range(a, b)]
        px = self.proc(images=imgs, return_tensors="pt")["pixel_values"]
        return px, (vid, a, b)


def encode_images(model, proc, out_dir: str, workers: int) -> dict[str, np.ndarray]:
    path = os.path.join(out_dir, "lvbench_img.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    vids = []
    for name in sorted(os.listdir(os.path.join(FRAMES, "lvbench"))):
        if name.endswith(".json"):
            with open(os.path.join(FRAMES, "lvbench", name), encoding="utf-8") as fh:
                vids.append((name[:-5], len(json.load(fh)["t"])))
    loader = DataLoader(Chunks(vids, proc), batch_size=None, num_workers=workers,
                        prefetch_factor=2)
    acc: dict[str, list] = {}
    done = 0
    for px, (vid, a, b) in loader:
        with torch.no_grad():
            e = model.get_image_features(pixel_values=px.to("cuda", torch.float16))
        acc.setdefault(vid, []).append((a, torch.nn.functional.normalize(e.float(), dim=-1)
                                        .cpu().numpy().astype(np.float16)))
        done += b - a
        if done % (CHUNK * 100) < CHUNK:
            print(f"  {done} frames", flush=True)
    feats = {v: np.concatenate([p[1] for p in sorted(parts, key=lambda p: p[0])])
             for v, parts in acc.items()}
    np.savez(path, **feats)
    return feats


def query_forms(q: str) -> tuple[str, list[str]]:
    stem = q.split("\n(A)")[0].strip()
    opts = [o.strip() for _, o in re.findall(r"\(([ABCD])\)\s*(.*?)(?=\n\([ABCD]\)|$)", q, flags=re.S)]
    return stem, opts


def nll_cv(curves: list[np.ndarray], spans: list[tuple[int, int]], vids: list[str]) -> tuple[float, float]:
    taus = np.exp(np.linspace(np.log(0.3), np.log(30), 40))
    def nll(z, lo, hi, t):
        x = z / t
        m, ms = x.max(), x[lo:hi].max()
        return m + np.log(np.exp(x - m).sum()) - ms - np.log(np.exp(x[lo:hi] - ms).sum())
    tab = np.array([[nll(z.astype(np.float64), lo, hi, t) for t in taus]
                    for z, (lo, hi) in zip(curves, spans)])
    u = sorted(set(vids))
    random.Random(0).shuffle(u)
    fold = {v: i % 5 for i, v in enumerate(u)}
    f = np.array([fold[v] for v in vids])
    held, chosen = np.zeros(len(curves)), []
    for k in range(5):
        j = int(np.argmin(tab[f != k].mean(0)))
        held[f == k] = tab[f == k, j]
        chosen.append(taus[j])
    return float(np.median(chosen)), float(held.mean())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    from transformers import AutoModel, AutoProcessor
    out_dir = os.path.join("data/t1/priors", args.name)
    os.makedirs(out_dir, exist_ok=True)
    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model, dtype=torch.float16).to("cuda").eval()
    feats = encode_images(model, proc, out_dir, args.workers)
    scale = float(model.logit_scale.exp()) if hasattr(model, "logit_scale") else 100.0
    bias = float(model.logit_bias) if getattr(model, "logit_bias", None) is not None else 0.0

    items = [i for i in lvbench_items() if i["key"] in feats]
    texts, index = [], []
    for it in items:
        stem, opts = query_forms(it["question"])
        index.append((len(texts), len(opts)))
        texts += [stem] + [f"{stem} {o}" for o in opts]
    emb = []
    for i in range(0, len(texts), 256):
        t = proc(text=texts[i:i + 256], padding="max_length", truncation=True, max_length=64,
                 return_tensors="pt")
        with torch.no_grad():
            e = model.get_text_features(input_ids=t["input_ids"].to("cuda"))
        emb.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
    emb = np.concatenate(emb)

    res = {"stem": [], "max(stem+opt_k)": []}
    curves = {"stem": [], "max(stem+opt_k)": []}
    spans = []
    for it, (s, n) in zip(items, index):
        x = feats[it["key"]].astype(np.float32)
        zs = x @ emb[s] * scale + bias
        zo = (x @ emb[s + 1:s + 1 + n].T * scale + bias).max(1)
        lo, hi = span_seconds(it["span"][0], it["span"][1], len(x))
        spans.append((lo, hi))
        for k, z in (("stem", zs), ("max(stem+opt_k)", zo)):
            curves[k].append(z)
            res[k].append({"windows": window_scores(z, it["duration"], it["center"]),
                           "dist": dist_to_span(z, lo, hi)})
    vids = [it["key"] for it in items]
    print(f"\n=== prior: {args.name} ({args.model}), n = {len(items)} ===")
    print(f"{'query form':<18}{'paired':>8}{'R@1':>8}{'w60':>8}{'median':>8}{'tau*':>7}{'NLL':>7}")
    for k, recs in res.items():
        s = summarize(recs)
        tau, nll = nll_cv(curves[k], spans, vids)
        print(f"{k:<18}{s['paired_win']:>8.3f}{s['recall_at_1']:>8.3f}{s['within_60s']:>8.3f}"
              f"{s['median_dist']:>8.0f}{tau:>7.2f}{nll:>7.3f}")
    np.savez(os.path.join(out_dir, "curves_maxopt.npz"),
             **{f"lvbench:{it['key']}>>>{it['uid']}": z.astype(np.float16)
                for it, z in zip(items, curves["max(stem+opt_k)"])})
    print("  reference, SigLIP so400m: stem paired 0.722 / R@1 0.401 / w60 0.374 / NLL 4.340;"
          " max-option 0.774 / 0.471 / 0.411 / 3.937", flush=True)


if __name__ == "__main__":
    main()
