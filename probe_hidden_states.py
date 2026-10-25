"""Is region answerability READABLE from the answerer's activations, if not promptable?

The gate between "no paper" and "a method paper". Prompting Qwen2.5-VL-7B whether a
240 s window answers its question gives AUROC 0.564 against adjacent-window negatives
(probe_answerability.py). Grounding-with-Confidence (arXiv 2609.39883) showed the same
contrast in a shorter-video regime: a prompted verifier is a bad ranker, while a 0.66M
MLP reading the model's own pooled states reaches 0.754-0.818 on the same label.

So the question is not whether the model *knows*, but whether we have been asking the
wrong way. This extracts frozen features and trains a small probe.

Two readout positions, because NS-P1 found the choice decisive -- a head on the
pre-generation state collapsed to the dataset mean, and tapping after the model commits
to an answer is what made it work:
  vis   mean over the window's VIDEO token positions
  last  the final prompt position, where the model is about to answer
each at the last layer and a middle layer (DAFS reads text->vision attention at L11-L15).

**Splits are by VIDEO, not by question.** LVBench has ~15 questions per video; a random
row split would put windows from one video on both sides and inflate AUROC through
memorised scenery rather than learned answerability.

Phase 1 (GPU):   --phase extract --out feats.npz
Phase 2 (CPU):   --phase train   --feats feats.npz
"""
from __future__ import annotations

import argparse
import os
import random

import numpy as np
import torch
from decord import VideoReader
from transformers import AutoModelForImageTextToText, AutoProcessor

from eval_lvbench_foveate import load_questions
from probe_answerability import B, PROMPT, clip_frames, windows_for

LAYERS = {"last": -1, "mid": 14}      # the 7B has 28 layers


@torch.no_grad()
def features(model, proc, frames, idx, fps, total, question, new_api, video_token_id):
    """Pooled hidden states for one window: {readout_layer: vector}."""
    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"},
                                      {"type": "text", "text": PROMPT.format(question)}]}],
        tokenize=False, add_generation_prompt=True)
    meta = {"fps": fps, "frames_indices": idx, "total_num_frames": total,
            "video_backend": "decord"}
    if new_api:
        inp = proc(text=[text], images=None, videos=[frames], video_metadata=[meta],
                   do_sample_frames=False, padding=True, return_tensors="pt").to("cuda")
    else:
        inp = proc(text=[text], images=None, videos=[(frames, meta)],
                   padding=True, return_tensors="pt").to("cuda")
    out = model(**inp, output_hidden_states=True)
    ids = inp["input_ids"][0]
    is_vis = (ids == video_token_id)
    feats = {}
    for name, li in LAYERS.items():
        h = out.hidden_states[li][0].float()
        feats[f"vis_{name}"] = (h[is_vis].mean(0) if is_vis.any() else h.mean(0)).cpu().numpy()
        feats[f"last_{name}"] = h[-1].cpu().numpy()
    return feats


def extract(args) -> None:
    items, census = load_questions(f"{B}/data/train-00000-of-00001.parquet",
                                   f"{B}/video_info.meta.jsonl", f"{B}/videos")
    print(f"population: {census}", flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(items)
    items = items[: args.n_questions]

    proc = AutoProcessor.from_pretrained(args.model, padding_side="left",
                                         trust_remote_code=True)
    new_api = getattr(proc, "video_processor", None) is not None
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda", trust_remote_code=True).eval()
    vid_tok = model.config.video_token_id
    print(f"video_token_id={vid_tok}, extracting {len(items)} questions", flush=True)

    store: dict[str, list] = {}
    labels, videos, cands = [], [], []
    done = skipped = 0
    for item in items:
        try:
            vr = VideoReader(item["video_path"])
            total, fps = len(vr), vr.get_avg_fps()
            rows = []
            for c in windows_for(item):
                fr, idx = clip_frames(vr, fps, total, c["start"], c["end"])
                rows.append((c, features(model, proc, fr, idx, fps, total,
                                         item["question"], new_api, vid_tok)))
        except Exception as ex:
            skipped += 1
            print(f"  [skip {skipped}] {item['key']} | {ex!r}", flush=True)
            if done == 0 and skipped >= 10:
                raise RuntimeError(f"10 failures, nothing extracted: {ex!r}") from ex
            continue
        for c, f in rows:
            for k, v in f.items():
                store.setdefault(k, []).append(v.astype(np.float16))
            labels.append(int(c["is_positive"]))
            videos.append(item["key"])
            cands.append(c["cand"])
        done += 1
        if done % 25 == 0:
            print(f"  [{done}/{len(items)}] {len(labels)} windows", flush=True)

    arrays = {k: np.stack(v) for k, v in store.items()}
    np.savez_compressed(args.out, label=np.array(labels, dtype=np.uint8),
                        video=np.array(videos), cand=np.array(cands), **arrays)
    print(f"\nwrote {args.out}: {len(labels)} windows from {done} questions "
          f"({skipped} skipped), {int(np.sum(labels))} positive, "
          f"{len(set(videos))} videos, dim={next(iter(arrays.values())).shape[1]}",
          flush=True)


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """Rank-based AUROC, ties averaged."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores), float)
    ranks[order] = np.arange(1, len(scores) + 1)
    for v in np.unique(scores):
        m = scores == v
        if m.sum() > 1:
            ranks[m] = ranks[m].mean()
    p, n = labels == 1, labels == 0
    if not p.any() or not n.any():
        return float("nan")
    return (ranks[p].sum() - p.sum() * (p.sum() + 1) / 2) / (p.sum() * n.sum())


def train(args) -> None:
    d = np.load(args.feats, allow_pickle=True)
    labels, videos, cands = d["label"], d["video"], d["cand"]
    feat_names = [k for k in d.files if k not in ("label", "video", "cand")]
    uniq = sorted(set(videos.tolist()))
    rng = random.Random(args.seed)
    rng.shuffle(uniq)
    cut = int(0.7 * len(uniq))
    tr_v = set(uniq[:cut])
    tr = np.array([v in tr_v for v in videos])
    te = ~tr
    hard = np.array([c in ("gt", "hard_neg") for c in cands])
    print(f"  {len(uniq)} videos -> {len(tr_v)} train / {len(uniq)-len(tr_v)} test "
          f"(split BY VIDEO)")
    print(f"  {tr.sum()} train windows, {te.sum()} test, "
          f"{int(labels[te].sum())} test positives\n")
    print(f"  {'readout':<12} {'AUROC all':>10} {'AUROC hard':>11}")

    for name in sorted(feat_names):
        X = d[name].astype(np.float32)
        X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
        xtr = torch.tensor(X[tr]); ytr = torch.tensor(labels[tr], dtype=torch.float32)
        xte = torch.tensor(X[te])
        torch.manual_seed(args.seed)
        net = torch.nn.Sequential(
            torch.nn.LayerNorm(X.shape[1]), torch.nn.Linear(X.shape[1], 256),
            torch.nn.GELU(), torch.nn.Dropout(0.1), torch.nn.Linear(256, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-2)
        # positives are ~1 in 12, so weight them or the probe predicts all-negative
        w = (ytr == 0).sum() / max(int((ytr == 1).sum()), 1)
        lossf = torch.nn.BCEWithLogitsLoss(pos_weight=torch.tensor(float(w)))
        net.train()
        for _ in range(args.epochs):
            perm = torch.randperm(len(xtr))
            for i in range(0, len(xtr), 64):
                b = perm[i:i + 64]
                opt.zero_grad()
                lossf(net(xtr[b]).squeeze(-1), ytr[b]).backward()
                opt.step()
        net.eval()
        with torch.no_grad():
            s = net(xte).squeeze(-1).numpy()
        hm = hard[te]
        print(f"  {name:<12} {auroc(s, labels[te]):>10.3f} "
              f"{auroc(s[hm], labels[te][hm]):>11.3f}")

    print("\n  PROMPTED baseline on the same task: AUROC(hard) = 0.564")
    print("  >=0.75 -> the signal is there and prompting was the bug; build the method.")
    print("  ~0.56  -> the signal is not in these features; the direction closes.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--phase", choices=["extract", "train"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--n_questions", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--out", default="logs/nsp5/probe_feats.npz")
    ap.add_argument("--feats", default="logs/nsp5/probe_feats.npz")
    args = ap.parse_args()
    if args.phase == "extract":
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        extract(args)
    else:
        train(args)


if __name__ == "__main__":
    main()
