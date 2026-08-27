"""Train a boundary-head ARM on cached POST-ANSWER frozen states (NS-P1 method bet C).

Three ablation arms, all head-only on a frozen TimeLens-7B (cache from cache_hidden.py):
  --arm ans      BoundaryHead (MLP) on the post-answer token  h_ans      [arm 1, fair baseline]
  --arm xattn    CrossAttnBoundaryHead: h_ans query x 256 visual tokens   [arm 2, AdaVTG-style]
  --arm vismean  BoundaryHead (MLP) on the mean-pooled visual vector      [arm 3, pooled baseline]

Trains in minutes on cached features. Run (TimeLens venv, from research/TimeLens, PYTHONPATH=.):
  env -u VIRTUAL_ENV PYTHONPATH=. ./.venv/bin/python -m c_head.train_head \\
      --tag train_natural.postans --arm ans --name ans
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.data import DataLoader, TensorDataset, WeightedRandomSampler

from c_head import config as C
from c_head.head import (
    BoundaryHead,
    CrossAttnBoundaryHead,
    boundary_loss,
    cw_to_startend,
    giou_1d,
)

MLP_INPUT = {"ans": "h_ans", "vismean": "vis_mean"}  # single-vector input per MLP arm
ARMS = ("ans", "xattn", "vismean")


def _gt_cw(metas):
    """Duration-normalized (center,width) targets; drop degenerate spans. Returns (cw, keep)."""
    cw, keep = [], []
    for i, m in enumerate(metas):
        dur, s, e = float(m["duration"]), float(m["gt_start"]), float(m["gt_end"])
        if dur <= 0 or e <= s:
            continue
        c = ((s + e) / 2.0) / dur
        w = (e - s) / dur
        cw.append([min(max(c, 0.0), 1.0), min(max(w, 1e-4), 1.0)])
        keep.append(i)
    return torch.tensor(cw, dtype=torch.float32), keep


def load_cache(cache_dir, tags, need_vis):
    """Glob one or more splits' post-answer shards -> {h_ans, vis_mean, vis?, gt_cw, metas}.

    ``tags`` may be a single tag or a list (e.g. natural + speed-aug) whose shards are
    concatenated. ``vis`` (the large [N,P,H] tensor) is loaded only when an arm needs it,
    and kept fp16 (cast to fp32 per batch) to bound RAM.
    """
    if isinstance(tags, str):
        tags = [tags]
    files = []
    for tag in tags:
        fs = sorted(Path(cache_dir).glob(f"{tag}.shard*of*.pt"))
        if not fs:
            raise FileNotFoundError(f"no cache shards matching {tag}.shard*of*.pt in {cache_dir}")
        files.extend(fs)
    h_ans, vis_mean, vis, metas = [], [], [], []
    for f in files:
        d = torch.load(f, map_location="cpu")
        if not len(d["meta"]):
            continue
        h_ans.append(d["h_ans"])
        vis_mean.append(d["vis_mean"])
        metas.extend(d["meta"])
        if need_vis:
            vis.append(d["vis"])
    gt_cw, keep = _gt_cw(metas)
    return {
        "h_ans": torch.cat(h_ans).float()[keep],
        "vis_mean": torch.cat(vis_mean).float()[keep],
        "vis": torch.cat(vis)[keep] if need_vis else None,  # fp16; cast per batch
        "gt_cw": gt_cw,
        "metas": [metas[i] for i in keep],
    }


def make_dataset(data, arm, idx):
    gt = data["gt_cw"][idx]
    if arm == "xattn":
        return TensorDataset(data["h_ans"][idx], data["vis"][idx], gt)
    return TensorDataset(data[MLP_INPUT[arm]][idx], gt)


def run_head(head, arm, batch, device):
    if arm == "xattn":
        q, kv, _ = batch
        return head(q.to(device), kv.to(device).float())
    x, _ = batch
    return head(x.to(device))


@torch.no_grad()
def val_metrics(head, arm, ds, device, bs=512):
    head.eval()
    n = dc = dw = miou = 0.0
    for batch in DataLoader(ds, batch_size=bs):
        gt = batch[-1].to(device)
        pred = run_head(head, arm, batch, device)
        n += gt.shape[0]
        dc += (pred[:, 0] - gt[:, 0]).abs().sum().item()
        dw += (pred[:, 1] - gt[:, 1]).abs().sum().item()
        iou, _ = giou_1d(cw_to_startend(pred), cw_to_startend(gt))
        miou += iou.clamp(min=0).sum().item()
    head.train()
    return {"mean_|dc|": dc / n, "mean_|dw|": dw / n, "mIoU(norm)": miou / n}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=ARMS)
    ap.add_argument("--name", required=True, help="checkpoint name under ckpt/")
    ap.add_argument("--tag", default="train_natural.postans",
                    help="cache tag, or comma-separated tags to concatenate (e.g. natural + speed-aug)")
    ap.add_argument("--short_upweight", type=float, default=1.0,
                    help="oversample factor for sub-threshold (short) GT moments during training")
    ap.add_argument("--short_thresh", type=float, default=2.0, help="seconds; GT moments below this are 'short'")
    ap.add_argument("--cache_dir", default=str(C.CACHE_DIR))
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch_size", type=int, default=256)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--weight_decay", type=float, default=1e-4)
    ap.add_argument("--lambda_l1", type=float, default=1.0)
    ap.add_argument("--lambda_giou", type=float, default=1.0)
    ap.add_argument("--val_frac", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    data = load_cache(args.cache_dir, [t.strip() for t in args.tag.split(",")], need_vis=(args.arm == "xattn"))
    n = data["gt_cw"].shape[0]
    n_val = max(1, int(n * args.val_frac)) if n > 20 else 0
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(args.seed))
    val_idx, tr_idx = perm[:n_val], perm[n_val:]
    print(f"[{args.tag}|{args.arm}] N={n} train={len(tr_idx)} val={len(val_idx)} | "
          f"GT(c,w) mean={data['gt_cw'].mean(0).tolist()}")

    head = (CrossAttnBoundaryHead(C.HIDDEN_SIZE) if args.arm == "xattn"
            else BoundaryHead(C.HIDDEN_SIZE)).to(device)
    opt = torch.optim.AdamW(head.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    train_ds = make_dataset(data, args.arm, tr_idx)
    if args.short_upweight != 1.0:
        # oversample short GT moments so the head gets real sub-second training signal
        lens_s = [float(data["metas"][i]["gt_end"]) - float(data["metas"][i]["gt_start"]) for i in tr_idx.tolist()]
        w = torch.tensor([args.short_upweight if l < args.short_thresh else 1.0 for l in lens_s])
        print(f"  short_upweight={args.short_upweight} (<{args.short_thresh}s): "
              f"{int((w != 1.0).sum())}/{len(lens_s)} train moments up-weighted")
        sampler = WeightedRandomSampler(w, num_samples=len(tr_idx), replacement=True)
        loader = DataLoader(train_ds, batch_size=args.batch_size, sampler=sampler)
    else:
        loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    val_ds = make_dataset(data, args.arm, val_idx) if n_val else None
    total = max(1, args.epochs * len(loader))
    warm = max(1, int(0.03 * total))

    def lr_at(s):
        if s < warm:
            return s / warm
        return 0.5 * (1.0 + math.cos(math.pi * (s - warm) / max(1, total - warm)))

    step = 0
    for ep in range(args.epochs):
        el = el1 = eg = 0.0
        for batch in loader:
            gt = batch[-1].to(device)
            for g in opt.param_groups:
                g["lr"] = args.lr * lr_at(step)
            pred = run_head(head, args.arm, batch, device)
            loss, parts = boundary_loss(pred, gt, args.lambda_l1, args.lambda_giou)
            opt.zero_grad()
            loss.backward()
            opt.step()
            el += float(loss.detach()); el1 += parts["l1"]; eg += parts["giou_loss"]; step += 1
        if ep % 10 == 0 or ep == args.epochs - 1:
            nb = len(loader)
            msg = f"epoch {ep:3d} | loss {el/nb:.4f} l1 {el1/nb:.4f} giou {eg/nb:.4f}"
            if val_ds is not None:
                msg += " | val " + " ".join(f"{k}={v:.4f}" for k, v in val_metrics(head, args.arm, val_ds, device).items())
            print(msg, flush=True)

    ckpt_dir = Path(C.CKPT_DIR)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    p = ckpt_dir / f"{args.name}.pt"
    torch.save(
        {"state_dict": head.state_dict(), "hidden_size": C.HIDDEN_SIZE, "arm": args.arm, "args": vars(args)},
        p,
    )
    print(f"saved -> {p}")


if __name__ == "__main__":
    main()
