"""Score LVBench's cached 1 fps frames with any HF image-text model, and compare priors.

The frame cache (t1_decode.py) makes swapping the encoder cheap: no video decoding.
For a model it writes per-second image embeddings for all 103 LVBench videos, then
the query-form curves T1/A2 use -- stem, and max over options of (stem + option_k) --
and reports the within-video localization metrics on the N3/T1 window protocol, plus
the 5-fold cross-validated posterior temperature and its held-out NLL.

  .venv/bin/python -u prior_encode.py --model google/siglip2-giant-opt-patch16-384 \\
      --name siglip2_giant

Qwen3-VL-Embedding (2B/8B) is a second family: a VLM embedder, scored by cosine x 100.
Frames use the model's default instruction and last-token pooling, as in the reference
implementation (QwenLM/Qwen3-VL-Embedding, src/models/qwen3_vl_embedding.py); queries
use QUERY_INSTRUCTION. Its Conv3d patch embedding is swapped for the identical matmul
(fast_patch_embed.py), without which bf16 on a B200 takes seconds per frame.
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
QUERY_INSTRUCTION = ("Find the video frame that shows what this question about the video "
                     "asks about.")
_QPROC = None


def is_qwen3_emb(model_id: str) -> bool:
    return "Qwen3-VL-Embedding" in model_id


def qwen3_conversation(instruction: str, image=None, text: str | None = None) -> list[dict]:
    content = []
    if image is not None:
        content.append({"type": "image", "image": image, "min_pixels": 4 * 32 * 32,
                        "max_pixels": 1800 * 32 * 32})
    if text is not None:
        content.append({"type": "text", "text": text})
    return [{"role": "system", "content": [{"type": "text", "text": instruction}]},
            {"role": "user", "content": content}]


def qwen3_inputs(proc, conversations: list) -> dict:
    from qwen_vl_utils.vision_process import process_vision_info
    text = proc.apply_chat_template(conversations, add_generation_prompt=True, tokenize=False)
    images, _ = process_vision_info(conversations, image_patch_size=16)
    return proc(text=text, images=images, padding=True, do_resize=False, return_tensors="pt")


def load_qwen3_emb(model_id: str):
    """(model, processor). The checkpoint stores weights under `model.*` (the reference
    wraps a Qwen3VLModel as `.model`), so a bare Qwen3VLModel.from_pretrained loads
    NOTHING and runs randomly initialised -- verified the hard way. Mirror the wrapper,
    and refuse to continue if any key is missing or unexpected."""
    import fast_patch_embed
    from transformers.models.qwen3_vl.modeling_qwen3_vl import (Qwen3VLModel,
                                                                Qwen3VLPreTrainedModel)
    from transformers.models.qwen3_vl.processing_qwen3_vl import Qwen3VLProcessor
    fast_patch_embed.patch(verbose=False)

    class Qwen3VLForEmbedding(Qwen3VLPreTrainedModel):
        _checkpoint_conversion_mapping: dict = {}

        def __init__(self, config):
            super().__init__(config)
            self.model = Qwen3VLModel(config)
            self.post_init()

        def forward(self, **kwargs):
            return self.model(**kwargs)

    model, info = Qwen3VLForEmbedding.from_pretrained(model_id, dtype=torch.bfloat16,
                                                      output_loading_info=True)
    bad = list(info["missing_keys"]) + list(info["unexpected_keys"])
    if bad:
        raise RuntimeError(f"{model_id}: weights did not load ({len(info['missing_keys'])} "
                           f"missing, {len(info['unexpected_keys'])} unexpected, e.g. {bad[:3]})")
    return (model.to("cuda").eval(),
            Qwen3VLProcessor.from_pretrained(model_id, padding_side="right"))


@torch.no_grad()
def qwen3_embed(model, inputs: dict) -> torch.Tensor:
    inputs = {k: v.to("cuda") for k, v in inputs.items()}
    h = model(**inputs).last_hidden_state
    last = inputs["attention_mask"].sum(1) - 1                 # right padding
    return torch.nn.functional.normalize(h[torch.arange(len(h), device=h.device), last].float(), dim=-1)


class Chunks(Dataset):
    def __init__(self, vids: list[tuple[str, int]], proc, model_id: str = "", chunk: int = CHUNK,
                 bench: str = "lvbench"):
        self.items = [(v, a, min(a + chunk, n)) for v, n in vids for a in range(0, n, chunk)]
        self.proc, self.model_id, self.bench = proc, model_id, bench

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int):
        vid, a, b = self.items[i]
        with open(os.path.join(FRAMES, self.bench, vid + ".json"), encoding="utf-8") as fh:
            off = json.load(fh)["off"]
        with open(os.path.join(FRAMES, self.bench, vid + ".bin"), "rb") as fh:
            fh.seek(off[a])
            blob = fh.read(off[b] - off[a])
        imgs = [Image.open(io.BytesIO(blob[off[k] - off[a]:off[k + 1] - off[a]])).convert("RGB")
                for k in range(a, b)]
        if is_qwen3_emb(self.model_id):
            convs = [qwen3_conversation("Represent the user's input.", image=im) for im in imgs]
            return dict(qwen3_inputs(self.proc, convs)), (vid, a, b)
        px = self.proc(images=imgs, return_tensors="pt")["pixel_values"]
        return px, (vid, a, b)


def encode_images(model, proc, out_dir: str, workers: int, model_id: str = "",
                  bench: str = "lvbench") -> dict[str, np.ndarray]:
    path = os.path.join(out_dir, f"{bench}_img.npz")
    if os.path.exists(path):
        with np.load(path) as z:
            return {k: z[k] for k in z.files}
    vids = []
    for name in sorted(os.listdir(os.path.join(FRAMES, bench))):     # local cache, not /gpfs
        if name.endswith(".json") and name != "failures.json":
            with open(os.path.join(FRAMES, bench, name), encoding="utf-8") as fh:
                vids.append((name[:-5], len(json.load(fh)["t"])))
    q3 = is_qwen3_emb(model_id)
    loader = DataLoader(Chunks(vids, proc, model_id, chunk=32 if q3 else CHUNK, bench=bench),
                        batch_size=None,
                        num_workers=workers, prefetch_factor=2)
    acc: dict[str, list] = {}
    done = 0
    for px, (vid, a, b) in loader:
        with torch.no_grad():
            if q3:
                e = qwen3_embed(model, px)
            else:
                e = model.get_image_features(pixel_values=px.to("cuda", torch.float16))
        acc.setdefault(vid, []).append((a, torch.nn.functional.normalize(e.float(), dim=-1)
                                        .cpu().numpy().astype(np.float16)))
        done += b - a
        if done % (CHUNK * 100) < (b - a):
            print(f"  {done} frames", flush=True)
    feats = {v: np.concatenate([p[1] for p in sorted(parts, key=lambda p: p[0])])
             for v, parts in acc.items()}
    np.savez(path, **feats)
    return feats


def query_forms(q: str) -> tuple[str, list[str]]:
    stem = q.split("\n(A)")[0].strip()
    opts = [o.strip() for _, o in re.findall(r"\(([ABCD])\)\s*(.*?)(?=\n\([ABCD]\)|$)", q, flags=re.S)]
    return stem, opts


def nll_cv(curves: list[np.ndarray], spans: list, vids: list[str]) -> tuple[float, float]:
    """5-fold (by video) CV temperature and held-out NLL of the evidence location.
    `spans[i]` is (lo, hi) or a list of them; the evidence is their union."""
    taus = np.exp(np.linspace(np.log(0.3), np.log(30), 40))
    masks = []
    for z, sp in zip(curves, spans):
        m = np.zeros(len(z), bool)
        for lo, hi in ([sp] if isinstance(sp[0], (int, np.integer)) else sp):
            m[lo:hi] = True
        masks.append(m)

    def nll(z, m, t):
        x = z / t
        mx, ms = x.max(), x[m].max()
        return mx + np.log(np.exp(x - mx).sum()) - ms - np.log(np.exp(x[m] - ms).sum())
    tab = np.array([[nll(z.astype(np.float64), m, t) for t in taus]
                    for z, m in zip(curves, masks)])
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
    ap.add_argument("--bench", default="lvbench",
                    choices=["lvbench", "videomme", "lvb", "mlvu", "cgbench"])
    args = ap.parse_args()
    from transformers import AutoModel, AutoProcessor
    out_dir = os.path.join("data/t1/priors", args.name)
    os.makedirs(out_dir, exist_ok=True)
    q3 = is_qwen3_emb(args.model)
    if q3:
        model, proc = load_qwen3_emb(args.model)
        scale, bias = 100.0, 0.0
    else:
        proc = AutoProcessor.from_pretrained(args.model)
        model = AutoModel.from_pretrained(args.model, dtype=torch.float16).to("cuda").eval()
        scale = float(model.logit_scale.exp()) if hasattr(model, "logit_scale") else 100.0
        bias = float(model.logit_bias) if getattr(model, "logit_bias", None) is not None else 0.0
    feats = encode_images(model, proc, out_dir, args.workers, args.model, args.bench)

    if args.bench == "lvbench":                  # the exact LVBench path, unchanged
        items = []
        for i in lvbench_items():
            if i["key"] in feats:
                stem, opts = query_forms(i["question"])
                items.append({"vid": i["key"], "qkey": f"lvbench:{i['key']}>>>{i['uid']}",
                              "stem": stem, "opts": opts, "spans": [tuple(i["span"])],
                              "duration": i["duration"], "center": i["center"]})
    else:
        import bench as _bench
        items = []
        for q in _bench.load(args.bench):
            if q.vid in feats:
                sp = list(q.spans)
                items.append({"vid": q.vid, "qkey": q.key, "stem": q.stem, "opts": list(q.options),
                              "spans": sp, "duration": q.duration or float(len(feats[q.vid])),
                              "center": (sp[0][0] + sp[0][1]) / 2 if sp else None})
    texts, index = [], []
    for it in items:
        index.append((len(texts), len(it["opts"])))
        texts += [it["stem"]] + [f"{it['stem']} {o}" for o in it["opts"]]
    emb = []
    for i in range(0, len(texts), 256):
        if q3:
            convs = [qwen3_conversation(QUERY_INSTRUCTION, text=t) for t in texts[i:i + 256]]
            e = qwen3_embed(model, dict(qwen3_inputs(proc, convs)))
        else:
            t = proc(text=texts[i:i + 256], padding="max_length", truncation=True, max_length=64,
                     return_tensors="pt")
            with torch.no_grad():
                e = model.get_text_features(input_ids=t["input_ids"].to("cuda"))
        emb.append(torch.nn.functional.normalize(e.float(), dim=-1).cpu().numpy())
    emb = np.concatenate(emb)

    res = {"stem": [], "max(stem+opt_k)": []}
    curves = {"stem": [], "max(stem+opt_k)": []}
    all_curves = {"stem": {}, "max(stem+opt_k)": {}}
    spans, vids = [], []
    for it, (s, n) in zip(items, index):
        x = feats[it["vid"]].astype(np.float32)
        zs = x @ emb[s] * scale + bias
        zo = (x @ emb[s + 1:s + 1 + n].T * scale + bias).max(1)
        all_curves["stem"][it["qkey"]], all_curves["max(stem+opt_k)"][it["qkey"]] = zs, zo
        if not it["spans"]:
            continue                             # no evidence labels: curves only
        sp = [span_seconds(a, b, len(x)) for a, b in it["spans"]]
        spans.append(sp)
        vids.append(it["vid"])
        for k, z in (("stem", zs), ("max(stem+opt_k)", zo)):
            curves[k].append(z)
            res[k].append({"windows": window_scores(z, it["duration"], it["center"]),
                           "dist": min(dist_to_span(z, lo, hi) for lo, hi in sp)})
    tag = "" if args.bench == "lvbench" else f"{args.bench}_"
    for k, name in (("max(stem+opt_k)", "curves_maxopt"), ("stem", "curves_stem")):
        np.savez(os.path.join(out_dir, f"{tag}{name}.npz"),
                 **{q: z.astype(np.float16) for q, z in all_curves[k].items()})
    print(f"\n=== prior: {args.name} ({args.model}) on {args.bench}: {len(items)} questions, "
          f"{len(spans)} with evidence labels ===")
    if not spans:
        print("  no evidence labels on this benchmark: curves written, no localization metrics",
              flush=True)
        return
    print(f"{'query form':<18}{'paired':>8}{'R@1':>8}{'w60':>8}{'median':>8}{'tau*':>7}{'NLL':>7}")
    for k, recs in res.items():
        s = summarize(recs)
        tau, nll = nll_cv(curves[k], spans, vids)
        print(f"{k:<18}{s['paired_win']:>8.3f}{s['recall_at_1']:>8.3f}{s['within_60s']:>8.3f}"
              f"{s['median_dist']:>8.0f}{tau:>7.2f}{nll:>7.3f}")
    if args.bench == "lvbench":
        print("  reference, SigLIP so400m: stem paired 0.722 / R@1 0.401 / w60 0.374 / NLL 4.340;"
              " max-option 0.774 / 0.471 / 0.411 / 3.937", flush=True)


if __name__ == "__main__":
    main()
