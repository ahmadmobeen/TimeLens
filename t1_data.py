"""T1 data: per-second feature banks, queries, splits, and synthetic timelines.

A synthetic timeline is a list of index ranges into a feature bank -- the "manifest"
of the pre-registration, assembled in feature space so building one costs a gather.

Distractor hygiene, decided before training (METHODS-LOG 2026-10-25):
  * distractors come from the query's own corpus, so the corpus is never a cue;
  * a distractor video is excluded when ANY of its own annotations is a near-paraphrase
    of the query (SigLIP text cosine >= tau), because an unlabelled second copy of
    the queried event would be a false negative;
  * two regimes -- "random" draws distractors uniformly (the pre-registered design,
    and the primary evaluation), "cluster" draws them from the queried video's own
    visual cluster (spherical k-means on mean SigLIP embeddings), so the right window
    cannot be found by scene type alone. Clusters, not the queried video's nearest
    neighbours: a timeline of one video's neighbours makes that video its hub, which
    a model could find without reading the query.
Splits are by source video: no video's frames appear in two splits, as query or as
distractor.
"""
from __future__ import annotations

import json
import math
import os
import random
from dataclasses import dataclass

import numpy as np
import torch

from t1_decode import ANET, CHARADES, LVB

FEATS = "data/t1/siglip"
TEXT = "data/t1/siglip_text.npz"
# Consolidated banks are a rebuildable cache of FEATS; keep them on fast local disk.
BANK_DIR = os.environ.get("T1_BANK_DIR", "data/t1")
ANNOTATIONS = {"charades": CHARADES, "anet": ANET}


class Bank:
    """All per-second features of one corpus, resident on the GPU.

    feats[i] = [pooled, grid_0..grid_8] for global second i, fp16 [N, 10, 1152].
    """

    def __init__(self, src: str, vids: list[str] | None = None, device: str = "cuda"):
        cache = os.path.join(BANK_DIR, f"bank_{src}")
        if not os.path.exists(cache + ".json"):
            self._consolidate(src, cache)
        with open(cache + ".json", encoding="utf-8") as fh:
            index = json.load(fh)
        arr = np.load(cache + ".npy", mmap_mode="r")
        self.src = src
        self.vids = [v for v in index["vids"] if vids is None or v in set(vids)]
        start = dict(zip(index["vids"], index["start"]))
        length = dict(zip(index["vids"], index["T"]))
        self.T = {v: length[v] for v in self.vids}
        self.feats = torch.from_numpy(np.ascontiguousarray(arr)).to(device)
        self.start = {v: start[v] for v in self.vids}
        emb = torch.stack([self.feats[start[v]:start[v] + length[v], 0].float().mean(0)
                           for v in self.vids])
        self.vid_emb = torch.nn.functional.normalize(emb, dim=-1)   # [V, 1152]
        self.row = {v: i for i, v in enumerate(self.vids)}

    @staticmethod
    def _consolidate(src: str, cache: str) -> None:
        d = os.path.join(FEATS, src)
        vids = sorted(n[:-4] for n in os.listdir(d)
                      if n.endswith(".npz") and not n.endswith(".tmp.npz"))
        lengths = []
        for v in vids:
            with np.load(os.path.join(d, v + ".npz")) as z:
                lengths.append(int(z["pooled"].shape[0]))
        out = np.lib.format.open_memmap(cache + ".npy.tmp", mode="w+", dtype=np.float16,
                                        shape=(sum(lengths), 10, 1152))
        pos = 0
        for v, n in zip(vids, lengths):
            with np.load(os.path.join(d, v + ".npz")) as z:
                out[pos:pos + n, 0] = z["pooled"]
                out[pos:pos + n, 1:] = z["grid"]
            pos += n
        out.flush()
        del out
        os.replace(cache + ".npy.tmp", cache + ".npy")
        starts = np.concatenate([[0], np.cumsum(lengths)[:-1]]).tolist()
        with open(cache + ".json", "w", encoding="utf-8") as fh:
            json.dump({"vids": vids, "start": starts, "T": lengths}, fh)
        print(f"consolidated {src}: {len(vids)} videos, {pos} seconds", flush=True)


class Text:
    """Frozen SigLIP text features for every query, resident on the GPU."""

    def __init__(self, device: str = "cuda"):
        z = np.load(TEXT)
        self.row = {k: i for i, k in enumerate(z["key"].tolist())}
        self.tok = torch.from_numpy(z["tok"]).to(device)            # [Nq, 64, 1152]
        self.pooled = torch.from_numpy(z["pooled"]).to(device)      # [Nq, 1152]
        self.ntok = torch.from_numpy(z["ntok"].astype(np.int64)).to(device)
        self.scale, self.bias = float(z["logit_scale"]), float(z["logit_bias"])


@dataclass(frozen=True)
class Query:
    key: str        # text key, e.g. "charades:17"
    vid: str
    a: float        # span start, seconds into the source video
    b: float


def load_queries(src: str) -> list[Query]:
    with open(ANNOTATIONS[src], encoding="utf-8") as fh:
        rows = json.load(fh)
    return [Query(f"{src}:{i}", os.path.splitext(os.path.basename(r["video_path"]))[0],
                  float(r["span"][0]), float(r["span"][1])) for i, r in enumerate(rows)]


def split_videos(vids: list[str], seed: int, val: float = 0.05,
                 test: float = 0.10) -> dict[str, set[str]]:
    order = sorted(vids)
    random.Random(seed).shuffle(order)
    n_te, n_va = int(len(order) * test), int(len(order) * val)
    return {"test": set(order[:n_te]), "val": set(order[n_te:n_te + n_va]),
            "train": set(order[n_te + n_va:])}


def span_seconds(a: float, b: float, T: int) -> tuple[int, int]:
    """Second indices [lo, hi) whose [k, k+1) overlaps the span; never empty."""
    lo = min(max(int(math.floor(a)), 0), T - 1)
    hi = min(max(int(math.ceil(b)), lo + 1), T)
    return lo, hi


def exclusions(queries: list[Query], text: Text, bank: Bank, tau: float) -> np.ndarray:
    """[Nq, V] bool on the CPU: video v may NOT serve as a distractor for query i.

    True for the query's own video and for every video holding an annotation whose
    text is a near-paraphrase of the query.
    """
    rows = torch.tensor([text.row[q.key] for q in queries], device=bank.feats.device)
    emb = torch.nn.functional.normalize(text.pooled[rows].float(), dim=-1)
    owner = torch.tensor([bank.row.get(q.vid, -1) for q in queries], device=emb.device)
    out = torch.zeros(len(queries), len(bank.vids), dtype=torch.bool, device=emb.device)
    for i in range(0, len(queries), 1024):
        sim = emb[i:i + 1024] @ emb.T                                  # [b, Nq]
        hit = (sim >= tau) & (owner[None, :] >= 0)
        r, c = hit.nonzero(as_tuple=True)
        out[i + r, owner[c]] = True
    valid = owner >= 0
    out[torch.arange(len(queries), device=emb.device)[valid], owner[valid]] = True
    return out.cpu().numpy()


@dataclass
class Timeline:
    idx: torch.Tensor     # [L] global second index into the bank
    label: torch.Tensor   # [L] bool, the query's evidence seconds
    lo: int               # evidence span in timeline seconds [lo, hi)
    hi: int
    src_lo: int           # where the queried video sits in the timeline
    src_hi: int


def make_timeline(q: Query, bank: Bank, allowed: np.ndarray, length: int,
                  mode: str, rng: random.Random, clusters: "Clusters | None" = None) -> Timeline:
    """Queried video at a random slot among distractors, cut to `length` seconds.

    `allowed` [V] bool marks usable distractors (split pool minus exclusions). The
    queried video is never truncated: the timeline is at least as long as it.
    """
    T_s = bank.T[q.vid]
    length = max(length, T_s)
    if mode == "cluster":
        order = []
        for c in clusters.near[clusters.assign[bank.row[q.vid]]]:
            members = [v for v in clusters.members[c] if allowed[v]]
            rng.shuffle(members)
            order += members                   # own cluster first, then the nearest ones
    else:
        order = np.flatnonzero(allowed).tolist()
        rng.shuffle(order)
    chosen, total = [], 0
    for v in order:
        if total >= length - T_s:
            break
        chosen.append(bank.vids[v])
        total += bank.T[bank.vids[v]]
    if mode == "cluster":
        rng.shuffle(chosen)
    offs = np.concatenate([[0], np.cumsum([bank.T[v] for v in chosen])])
    slot = rng.choice([k for k in range(len(chosen) + 1) if offs[k] + T_s <= length])
    seq = chosen[:slot] + [q.vid] + chosen[slot:]
    pieces, t, src_lo = [], 0, -1
    for v in seq:
        n = min(bank.T[v], length - t)
        if n <= 0:
            break
        if v == q.vid:
            src_lo = t
        pieces.append(torch.arange(bank.start[v], bank.start[v] + n))
        t += n
    assert src_lo >= 0
    idx = torch.cat(pieces)
    lo, hi = span_seconds(q.a, q.b, T_s)
    label = torch.zeros(len(idx), dtype=torch.bool)
    label[src_lo + lo:src_lo + hi] = True
    return Timeline(idx, label, src_lo + lo, src_lo + hi, src_lo, src_lo + T_s)


class Clusters:
    """Spherical k-means over videos' mean SigLIP embeddings, fit on TRAIN videos only
    and then used to assign every video. near[c] lists clusters by centroid similarity,
    c itself first."""

    def __init__(self, bank: Bank, train_pool: np.ndarray, per_cluster: int, seed: int,
                 iters: int = 30):
        x = bank.vid_emb[torch.from_numpy(train_pool).to(bank.vid_emb.device)]
        k = max(2, len(x) // per_cluster)
        g = torch.Generator(device="cpu").manual_seed(seed)
        cent = x[torch.randperm(len(x), generator=g)[:k].to(x.device)].clone()
        for _ in range(iters):
            a = (x @ cent.T).argmax(1)
            for c in range(k):
                m = a == c
                if m.any():
                    cent[c] = torch.nn.functional.normalize(x[m].mean(0), dim=0)
        self.assign = (bank.vid_emb @ cent.T).argmax(1).cpu().numpy()
        self.near = (cent @ cent.T).argsort(1, descending=True).cpu().numpy()
        self.members = [np.flatnonzero(self.assign == c).tolist() for c in range(k)]
        self.k = k


def lvbench_items() -> list[dict]:
    """The same 1,534 scoreable LVBench questions as E3, N3 and B1."""
    from eval_lvbench_foveate import load_questions
    items, census = load_questions(f"{LVB}/data/train-00000-of-00001.parquet",
                                   f"{LVB}/video_info.meta.jsonl", f"{LVB}/videos")
    assert census["scoreable"] == 1534, census
    return items
