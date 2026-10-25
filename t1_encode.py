"""T1 stage A, second half: frozen SigLIP features for every cached second, and for
every query.

Reads the 1 fps JPEG cache written by t1_decode.py. Per second it stores
  pooled  [T, 1152]     the attention-pooled image embedding (SigLIP's own image
                        vector, the one B1 scored with)
  grid    [T, 9, 1152]  the last-layer patch tokens averaged onto a 3x3 grid, so a
                        learned head can see *where* in the frame something is
both fp16, under data/t1/siglip/<src>/<vid>.npz, with the frame times.

Preprocessing reproduces SiglipImageProcessor (bicubic 384x384, no crop, mean=std=0.5)
and the model runs in fp16 as in B1, so pooled features are the ones B1 used.

  --mode frames --shard i/n   one process per GPU, videos dealt by frame count
  --mode text                 query features -> data/t1/siglip_text.npz

Text keys: "charades:<row>", "anet:<row>" (row of the *_natural.json annotation
list) and "lvbench:<key>>>><uid>". LVBench questions are encoded as the stem only,
options stripped, exactly as B1's prior did.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from t1_decode import ANET, CHARADES, LVB, OUT as FRAMES

MODEL = "google/siglip-so400m-patch14-384"
FEATS = "data/t1/siglip"
CHUNK = 256


def video_list(srcs: list[str]) -> list[tuple[str, str, int]]:
    """(src, vid, T) for every cached video that has no features yet."""
    out = []
    for src in srcs:
        d = os.path.join(FRAMES, src)
        for name in sorted(os.listdir(d)):
            if not name.endswith(".json") or name == "failures.json":
                continue
            vid = name[:-5]
            if os.path.exists(os.path.join(FEATS, src, vid + ".npz")):
                continue
            with open(os.path.join(d, name), encoding="utf-8") as fh:
                out.append((src, vid, len(json.load(fh)["t"])))
    return out


class Chunks(Dataset):
    """One item = up to CHUNK consecutive seconds of one video, as uint8 384x384."""

    def __init__(self, videos: list[tuple[str, str, int]]):
        self.items = [(s, v, a, min(a + CHUNK, n), n)
                      for s, v, n in videos for a in range(0, n, CHUNK)]

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        src, vid, a, b, n = self.items[i]
        with open(os.path.join(FRAMES, src, vid + ".json"), encoding="utf-8") as fh:
            off = json.load(fh)["off"]
        with open(os.path.join(FRAMES, src, vid + ".bin"), "rb") as fh:
            fh.seek(off[a])
            blob = fh.read(off[b] - off[a])
        arr = np.stack([
            np.asarray(Image.open(io.BytesIO(blob[off[k] - off[a]:off[k + 1] - off[a]]))
                       .convert("RGB").resize((384, 384), Image.BICUBIC))
            for k in range(a, b)])
        return torch.from_numpy(arr).permute(0, 3, 1, 2), (src, vid, a, b, n)


def encode_frames(args) -> None:
    from transformers import AutoModel
    k, n_shards = (int(x) for x in args.shard.split("/"))
    videos = sorted(video_list(args.src), key=lambda v: -v[2])
    videos = videos[k::n_shards]                      # deal by length: balanced shards
    print(f"shard {k}/{n_shards}: {len(videos)} videos, {sum(v[2] for v in videos)} "
          f"frames", flush=True)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
    vis = model.vision_model
    loader = DataLoader(Chunks(videos), batch_size=None, num_workers=args.workers,
                        prefetch_factor=4, pin_memory=True)
    acc: dict[tuple[str, str], list] = {}
    t0, done, frames = time.time(), 0, 0
    for x, (src, vid, a, b, n) in loader:
        with torch.no_grad():
            x = x.to("cuda", non_blocking=True).half().div_(127.5).sub_(1.0)
            out = vis(pixel_values=x)
            h = out.last_hidden_state                 # [n, 729, 1152], post-LN
            g = h.view(h.shape[0], 27, 27, -1).permute(0, 3, 1, 2)
            g = torch.nn.functional.adaptive_avg_pool2d(g.float(), 3)
            g = g.flatten(2).transpose(1, 2).half()   # [n, 9, 1152]
        acc.setdefault((src, vid), []).append(
            (a, out.pooler_output.cpu().numpy(), g.cpu().numpy()))
        frames += b - a
        if b == n:                                    # chunks arrive in order
            parts = sorted(acc.pop((src, vid)), key=lambda p: p[0])
            with open(os.path.join(FRAMES, src, vid + ".json"), encoding="utf-8") as fh:
                t = np.array(json.load(fh)["t"], dtype=np.float32)
            os.makedirs(os.path.join(FEATS, src), exist_ok=True)
            tmp = os.path.join(FEATS, src, vid + ".tmp.npz")
            np.savez(tmp, pooled=np.concatenate([p[1] for p in parts]).astype(np.float16),
                     grid=np.concatenate([p[2] for p in parts]).astype(np.float16), t=t)
            os.replace(tmp, os.path.join(FEATS, src, vid + ".npz"))
            done += 1
            if done % 200 == 0 or done == len(videos):
                el = time.time() - t0
                print(f"  [{done}/{len(videos)}] {frames} frames, {frames / el:.0f} fps, "
                      f"{el / 60:.1f} min", flush=True)
    print(f"DONE shard {k}: {done} videos, {frames} frames", flush=True)


def queries() -> list[tuple[str, str]]:
    """(key, text) for every query T1 trains or evaluates on."""
    import pandas as pd
    out = []
    for src, ann in (("charades", CHARADES), ("anet", ANET)):
        with open(ann, encoding="utf-8") as fh:
            out += [(f"{src}:{i}", r["query"]) for i, r in enumerate(json.load(fh))]
    df = pd.read_parquet(f"{LVB}/data/train-00000-of-00001.parquet")
    for r in df.itertuples(index=False):
        stem = r.question.split("\n(A)")[0].strip() or r.question.strip()
        out.append((f"lvbench:{r.key}>>>{r.uid}", stem))
    return out


def encode_text(args) -> None:
    from transformers import AutoModel, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL, torch_dtype=torch.float16).to("cuda").eval()
    qs = queries()
    toks, pooled, ntok = [], [], []
    for i in range(0, len(qs), 256):
        batch = tok([q for _, q in qs[i:i + 256]], padding="max_length", truncation=True,
                    max_length=64, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.text_model(input_ids=batch["input_ids"])
        toks.append(out.last_hidden_state.cpu().numpy().astype(np.float16))
        pooled.append(out.pooler_output.cpu().numpy().astype(np.float16))
        ntok.append((batch["input_ids"] != tok.pad_token_id).sum(1).cpu().numpy())
    out_path = "data/t1/siglip_text.npz"
    np.savez(out_path, key=np.array([k for k, _ in qs]), tok=np.concatenate(toks),
             pooled=np.concatenate(pooled), ntok=np.concatenate(ntok).astype(np.int16),
             logit_scale=float(model.logit_scale.exp()), logit_bias=float(model.logit_bias))
    print(f"wrote {out_path}: {len(qs)} queries, truncated at 64 tokens: "
          f"{int((np.concatenate(ntok) >= 64).sum())}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["frames", "text"], required=True)
    ap.add_argument("--src", nargs="+", default=["lvbench", "charades", "anet"])
    ap.add_argument("--shard", default="0/1")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    if args.mode == "frames":
        encode_frames(args)
    else:
        encode_text(args)


if __name__ == "__main__":
    main()
