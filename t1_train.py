"""T1 stage 1a: train the head on synthetic timelines; evaluate on held-out synthetic
and on LVBench, always beside the zero-shot SigLIP curve and a shuffled-query twin.

Pre-registered 2026-10-21, amended 2026-10-25 (NS-P5 METHODS-LOG).

Evaluation reuses N3's window protocol verbatim (probe_answerability.windows_for):
per question a 240 s window centred on the evidence, the nearest non-overlapping
neighbour 360 s away (the hard negative), a 600 s placebo and up to 20 uniform
fillers. A window's score is the log-sum-exp of the per-second logits inside it.
The pre-registered number is AUROC(gt vs hard_neg), pooled over questions as in N3;
the paired win rate (gt beats ITS OWN hard negative) is reported beside it because it
is free of cross-question calibration.

  --phase train          train, pick the checkpoint on VAL, report TEST synthetic
  --phase eval_lvbench   the stage-2 transfer test on the 1,534 E3/N3/B1 questions
"""
from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import threading
import time

import numpy as np
import torch
import torch.nn.functional as F

from probe_answerability import windows_for
from probe_hidden_states import auroc
from t1_data import (Bank, Clusters, Text, exclusions, load_queries, lvbench_items,
                     make_timeline, span_seconds, split_videos)
from t1_model import T1Head, zero_shot_logits

RUNS = "data/t1/runs"
EVAL_LEN = 3600            # pre-registered: ~60 min timelines


# ----------------------------------------------------------------------------- data

class Corpus:
    """One source corpus: bank, queries per split, distractor pools and exclusions."""

    def __init__(self, src: str, text: Text, tau: float, seed: int):
        self.src, self.tau = src, tau
        self.bank = Bank(src)
        qs = [q for q in load_queries(src) if q.vid in self.bank.row]
        self.split = split_videos(self.bank.vids, seed)
        self.queries = {k: [q for q in qs if q.vid in v] for k, v in self.split.items()}
        self.excl = {k: exclusions(self.queries[k], text, self.bank, tau)
                     for k in self.queries}
        self.pool = {k: np.array([v in self.split[k] for v in self.bank.vids])
                     for k in self.split}
        # ~one 60 min timeline per cluster: 150 Charades clips, 60 ANet videos
        per = 150 if src == "charades" else 60
        self.clusters = Clusters(self.bank, self.pool["train"], per, seed)

    def timeline(self, split: str, i: int, length: int, mode: str, rng: random.Random):
        allowed = self.pool[split] & ~self.excl[split][i]
        return make_timeline(self.queries[split][i], self.bank, allowed, length, mode,
                             rng, self.clusters)


def text_inputs(text: Text, keys: list[str]):
    rows = torch.tensor([text.row[k] for k in keys], device=text.tok.device)
    qmask = torch.arange(64, device=rows.device)[None, :] < text.ntok[rows][:, None]
    return text.tok[rows], qmask, text.pooled[rows]


def sampler(corpora: list[Corpus], weights: list[float], args, out: queue.Queue,
            stop: threading.Event, worker: int) -> None:
    """Background producer of CPU-side training batches: one corpus and one timeline
    length per batch, so a batch needs no padding. Each worker has its own stream."""
    rng = random.Random(args.seed * 1000 + worker + 1)
    while not stop.is_set():
        c = rng.choices(corpora, weights)[0]
        L = int(math.exp(rng.uniform(math.log(args.min_len), math.log(args.max_len))))
        B = max(4, min(64, args.tokens_per_batch // L))
        n = len(c.queries["train"])
        tls, keys = [], []
        for _ in range(B):
            i = rng.randrange(n)
            mode = "cluster" if rng.random() < args.p_cluster else "random"
            tls.append(c.timeline("train", i, L, mode, rng))
            keys.append(c.queries["train"][i].key)
        out.put((c, tls, keys))


def loss_fn(logits: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """Listwise: -log(probability mass on the evidence seconds), softmax over the
    whole timeline; plus a pos-weighted BCE so per-second scores stay calibrated."""
    z = logits.float()
    logp = z.log_softmax(-1)
    listwise = -logp.masked_fill(~label, float("-inf")).logsumexp(-1).mean()
    pos = label.sum().clamp(min=1)
    w = ((label.numel() - pos) / pos).clamp(max=100.0)
    bce = F.binary_cross_entropy_with_logits(z, label.float(), pos_weight=w)
    return listwise + 0.5 * bce


# ----------------------------------------------------------------------------- eval

def window_scores(z: np.ndarray, duration: float, center: float) -> list[dict]:
    T = len(z)
    out = []
    for c in windows_for({"duration": duration, "center": center}):
        s = min(max(int(math.floor(c["start"])), 0), T - 1)
        e = min(max(int(math.ceil(c["end"])), s + 1), T)
        seg = z[s:e].astype(np.float64)
        m = seg.max()
        out.append({"cand": c["cand"], "score": float(m + np.log(np.exp(seg - m).sum()))})
    return out


def summarize(per_item: list[dict]) -> dict:
    """per_item: {"windows": [...], "dist": seconds from argmax to the span}."""
    gt, hard, alln, paired, r1 = [], [], [], [], []
    for it in per_item:
        w = it["windows"]
        g = [x["score"] for x in w if x["cand"] == "gt"][0]
        h = [x["score"] for x in w if x["cand"] == "hard_neg"]
        gt.append(g)
        hard += h
        alln += [x["score"] for x in w if x["cand"] != "gt"]
        if h:
            paired.append(float(g > h[0]) + 0.5 * float(g == h[0]))
        r1.append(float(all(g > x["score"] for x in w if x["cand"] != "gt")))
    gt, hard, alln = map(np.array, (gt, hard, alln))
    d = np.array([it["dist"] for it in per_item])
    lab = lambda p, n: (np.concatenate([p, n]), np.r_[np.ones(len(p)), np.zeros(len(n))])
    return {"n": len(per_item),
            "auroc_hard": float(auroc(*lab(gt, hard))),
            "auroc_all": float(auroc(*lab(gt, alln))),
            "paired_win": float(np.mean(paired)) if paired else float("nan"),
            "recall_at_1": float(np.mean(r1)),
            "within_30s": float(np.mean(d <= 30)), "within_60s": float(np.mean(d <= 60)),
            "within_120s": float(np.mean(d <= 120)), "median_dist": float(np.median(d))}


def dist_to_span(z: np.ndarray, lo: int, hi: int) -> float:
    t = int(np.argmax(z))
    return 0.0 if lo <= t < hi else float(min(abs(t - lo), abs(t - (hi - 1))))


def derangement(n: int, seed: int, group: list | None = None,
                forbid: np.ndarray | None = None) -> list[int]:
    """A permutation with no fixed points. Given `group`, no item draws from its own
    group (LVBench: the placebo question is about another video). Given `forbid`
    [n, n] bool, no item draws a forbidden partner (synthetic: no paraphrase of its own
    query, which could still match the true evidence)."""
    rng = random.Random(seed)
    bad = lambda i, j: (i == j or (group is not None and group[i] == group[j])
                        or (forbid is not None and forbid[i, j]))
    p = list(range(n))
    rng.shuffle(p)
    for _ in range(100):
        viol = [i for i in range(n) if bad(i, p[i])]
        if not viol:
            return p
        for i in viol:                       # swap each violator with a random partner
            j = rng.randrange(n)              # whenever the swap leaves both legal
            if not bad(i, p[j]) and not bad(j, p[i]):
                p[i], p[j] = p[j], p[i]
    raise RuntimeError("no valid placebo derangement found")


@torch.no_grad()
def run_items(model, text: Text, feats_fn, groups: list[list[int]], keys: list[str]):
    """Logits for every item. `groups` lists item indices sharing one input length;
    feats_fn(group) -> [len(group), L, 10, 1152]. Returns (model, zero-shot) logits."""
    out_m, out_z = [None] * len(keys), [None] * len(keys)
    for g in groups:
        for i in range(0, len(g), 16):
            sub = g[i:i + 16]
            feats = feats_fn(sub)
            qtok, qmask, qpool = text_inputs(text, [keys[j] for j in sub])
            with torch.autocast("cuda", dtype=torch.bfloat16):
                prior = zero_shot_logits(feats[:, :, 0], qpool, text.scale, text.bias)
                zm = model(feats, qtok, qmask, prior) if model is not None else prior
            for j, a, b in zip(sub, zm.float().cpu().numpy(), prior.float().cpu().numpy()):
                out_m[j], out_z[j] = a, b
    return out_m, out_z


def eval_synthetic(model, c: Corpus, text: Text, split: str, mode: str, seed: int,
                   limit: int = 0) -> dict:
    """Fixed seeded timelines; metrics for the head, zero-shot SigLIP, and the
    shuffled-query twin of each."""
    qs = c.queries[split][:limit] if limit else c.queries[split]
    tls = [c.timeline(split, i, EVAL_LEN, mode, random.Random(seed * 100003 + i))
           for i in range(len(qs))]
    by_len: dict[int, list[int]] = {}
    for i, tl in enumerate(tls):
        by_len.setdefault(len(tl.idx), []).append(i)
    feats_fn = lambda sub: c.bank.feats[torch.stack([tls[j].idx for j in sub]).cuda()]
    keys = [q.key for q in qs]
    emb = F.normalize(text_inputs(text, keys)[2].float(), dim=-1)
    perm = derangement(len(keys), seed, forbid=(emb @ emb.T >= c.tau).cpu().numpy())
    res = {}
    for tag, ks in (("", keys), ("placebo_", [keys[p] for p in perm])):
        zm, zz = run_items(model, text, feats_fn, list(by_len.values()), ks)
        for name, zs in (("head", zm), ("zeroshot", zz)):
            items = [{"windows": window_scores(z, len(tl.idx), (tl.lo + tl.hi) / 2),
                      "dist": dist_to_span(z, tl.lo, tl.hi)} for z, tl in zip(zs, tls)]
            res[tag + name] = summarize(items)
    return res


# ----------------------------------------------------------------------------- train

def train(args) -> None:
    run_dir = os.path.join(RUNS, args.run)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(vars(args), fh, indent=1)
    torch.manual_seed(args.seed)
    text = Text()
    tau = {k: float(v) for k, v in (x.split("=") for x in args.tau)}
    corpora = [Corpus(s, text, tau[s], args.seed) for s in args.corpora]
    for c in corpora:
        print(f"{c.src}: " + ", ".join(f"{k} {len(v)} q / {len(c.split[k])} vids"
                                      for k, v in c.queries.items())
              + f" | excluded per query (train): {c.excl['train'].sum(1).mean():.1f}",
              flush=True)
    weights = [len(c.queries["train"]) for c in corpora]

    model = T1Head(d=args.d, layers=args.layers, dropout=args.dropout,
                   use_grid=not args.no_grid).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / args.warmup)
                                              * 0.5 * (1 + math.cos(math.pi * min(1.0, s / args.steps))))
    q: queue.Queue = queue.Queue(maxsize=16)
    stop = threading.Event()
    threads = [threading.Thread(target=sampler, args=(corpora, weights, args, q, stop, w),
                                daemon=True) for w in range(2)]
    for t in threads:
        t.start()

    best, log = -1.0, open(os.path.join(run_dir, "log.jsonl"), "a", encoding="utf-8")
    t0, run_loss = time.time(), []
    for step in range(1, args.steps + 1):
        c, tls, keys = q.get()
        idx = torch.stack([tl.idx for tl in tls]).cuda(non_blocking=True)
        label = torch.stack([tl.label for tl in tls]).cuda(non_blocking=True)
        feats = c.bank.feats[idx]
        qtok, qmask, qpool = text_inputs(text, keys)
        if args.qdrop > 0:                   # drop query tokens: robustness to wording
            keep = torch.rand(qmask.shape, device=qmask.device) > args.qdrop
            keep[:, 0] = True
            qmask = qmask & keep
        model.train()
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prior = zero_shot_logits(feats[:, :, 0], qpool, text.scale, text.bias)
            loss = loss_fn(model(feats, qtok, qmask, prior), label)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        sched.step()
        run_loss.append(loss.item())
        if step % 100 == 0:
            print(f"  step {step} loss {np.mean(run_loss):.4f} lr {sched.get_last_lr()[0]:.2e} "
                  f"L={idx.shape[1]} B={idx.shape[0]} {(time.time() - t0) / 60:.1f} min",
                  flush=True)
            run_loss = []
        if step % args.eval_every == 0 or step == args.steps:
            model.eval()
            val = {c.src: eval_synthetic(model, c, text, "val", "random", 7, limit=300)
                   for c in corpora}
            score = float(np.mean([v["head"]["auroc_hard"] for v in val.values()]))
            log.write(json.dumps({"step": step, "val": val}) + "\n")
            log.flush()
            print(f"  [val {step}] " + " | ".join(
                f"{s}: head {v['head']['auroc_hard']:.3f} (placebo "
                f"{v['placebo_head']['auroc_hard']:.3f}) zs {v['zeroshot']['auroc_hard']:.3f}"
                for s, v in val.items()), flush=True)
            if score > best:
                best = score
                torch.save({"model": model.state_dict(), "step": step, "args": vars(args)},
                           os.path.join(run_dir, "best.pt"))
    stop.set()

    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location="cuda")
    model.load_state_dict(ck["model"])
    model.eval()
    test = {c.src: {m: eval_synthetic(model, c, text, "test", m, 11)
                    for m in ("random", "cluster")} for c in corpora}
    with open(os.path.join(run_dir, "test_synthetic.json"), "w", encoding="utf-8") as fh:
        json.dump({"best_step": ck["step"], "test": test}, fh, indent=1)
    print(f"\n=== TEST synthetic (best val step {ck['step']}) ===")
    report(test)


def report(test: dict) -> None:
    cols = ("auroc_hard", "paired_win", "auroc_all", "recall_at_1", "within_60s",
            "median_dist")
    for src, by_mode in test.items():
        for mode, r in by_mode.items():
            print(f"  {src}/{mode}")
            print("    " + "".join(f"{c:>13}" for c in ("", *cols)))
            for arm in ("head", "placebo_head", "zeroshot", "placebo_zeroshot"):
                print("    " + f"{arm:>13}" + "".join(f"{r[arm][c]:>13.3f}" for c in cols))


# ----------------------------------------------------------------------------- LVBench

def eval_lvbench(args) -> None:
    run_dir = os.path.join(RUNS, args.run)
    text = Text()
    ck = torch.load(os.path.join(run_dir, "best.pt"), map_location="cuda")
    a = ck["args"]
    model = T1Head(d=a["d"], layers=a["layers"], dropout=a["dropout"],
                   use_grid=not a["no_grid"]).cuda().eval()
    model.load_state_dict(ck["model"])
    bank = Bank("lvbench")
    items = [i for i in lvbench_items() if i["key"] in bank.row]
    keys = [f"lvbench:{i['key']}>>>{i['uid']}" for i in items]
    groups: dict[str, list[int]] = {}
    for j, it in enumerate(items):
        groups.setdefault(it["key"], []).append(j)
    feats_fn = lambda sub: bank.feats[bank.start[items[sub[0]]["key"]]:
                                      bank.start[items[sub[0]]["key"]] + bank.T[items[sub[0]]["key"]]
                                      ][None].expand(len(sub), -1, -1, -1)
    perm = derangement(len(items), 13, group=[it["key"] for it in items])
    res, curves = {}, {}
    for tag, ks in (("", keys), ("placebo_", [keys[p] for p in perm])):
        zm, zz = run_items(model, text, feats_fn, list(groups.values()), ks)
        if not tag:
            curves = {k: z.astype(np.float16) for k, z in zip(keys, zm)}
        for name, zs in (("head", zm), ("zeroshot", zz)):
            per = []
            for z, it in zip(zs, items):
                lo, hi = span_seconds(it["span"][0], it["span"][1], len(z))
                per.append({"windows": window_scores(z, it["duration"], it["center"]),
                            "dist": dist_to_span(z, lo, hi)})
            res[tag + name] = summarize(per)
    np.savez(os.path.join(run_dir, "lvbench_curves.npz"), **curves)
    with open(os.path.join(run_dir, "lvbench.json"), "w", encoding="utf-8") as fh:
        json.dump({"step": ck["step"], "result": res}, fh, indent=1)
    print(f"\n=== LVBench zero-shot transfer (n={len(items)}, ckpt step {ck['step']}) ===")
    report({"lvbench": {"full": res}})
    print("\n  pre-registered: PASS needs AUROC(hard) >= 0.70 "
          "(prompted 0.564, probed 0.551)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["train", "eval_lvbench"], required=True)
    ap.add_argument("--run", required=True)
    ap.add_argument("--corpora", nargs="+", default=["charades", "anet"])
    ap.add_argument("--tau", nargs="+", default=["charades=0.80", "anet=0.85"],
                    help="per corpus: text cosine at which a distractor's annotation "
                         "counts as a paraphrase of the query (set by inspection, "
                         "METHODS-LOG 2026-10-25)")
    ap.add_argument("--p_cluster", type=float, default=0.5)
    ap.add_argument("--min_len", type=int, default=600)
    ap.add_argument("--max_len", type=int, default=7200)
    ap.add_argument("--tokens_per_batch", type=int, default=115200)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--eval_every", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--dropout", type=float, default=0.1)
    ap.add_argument("--qdrop", type=float, default=0.1)
    ap.add_argument("--no_grid", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    if args.phase == "train":
        train(args)
    else:
        eval_lvbench(args)


if __name__ == "__main__":
    main()
