#!/usr/bin/env python3
"""F2C, "From Frames to Clips: Training-free Adaptive Key Clip Selection for Long-Form Video
Understanding" (Sun et al., arXiv 2510.02262v2), re-implemented from the paper for NS-P5. No code is
released: the project page's code link is commented out and github.com/imguangyu/F2C answers 404
(checked 2026-11-16).

Per question, from one relevance score per second and the per-second frame embeddings:

1. Relevance r_t = cos(E(f_t), E(Q)) at 1 fps (Eq. 2, Sec. 5.1); for multiple choice the query holds
   the question and its options (Sec. 4.1).
2. Anchor key frames (Sec. 4.1; Algorithm 1 in App. D.2): the valleys (local minima) of r cut the
   video into basins and the peak of each basin is a candidate. With more than K_anchor candidates,
   k-means on their temporal indices forms K_anchor clusters and the best-scoring candidate of each
   cluster is an anchor; otherwise every candidate is an anchor.
3. Clip length per anchor (Sec. 4.2.2, Eqs. 6-8), by exhaustive search over 1 <= l <= l_max:
   l* = argmax S(l) - lam_r R(l) + lam_l l / l_max, with S the mean relevance of the clip's frames
   (Eq. 6), R their mean pairwise frame-frame cosine (Eq. 7) and l_max = s_max^2 K / K_anchor.
4. Resolution (Eqs. 3-5): the clip's frames are downscaled by s* = sqrt(K_anchor l* / K) per side,
   so each frame holds 1 / s*^2 of a full frame's visual tokens and each clip costs K / K_anchor full
   frames: the K_anchor clips spend the budget of K full frames.
5. Overlapping clips with identical resolutions merge (end of Sec. 4.2.2): "overlapping frames are
   encoded only once" (Sec. 6.4).
Hyperparameters (App. D.3): lam_r = 0.5, lam_l = 0.05, s_max = 2, K_anchor = K.

Not specified in the paper, chosen here:
- A clip of length l has exactly l frames, [k - floor((l-1)/2), k - floor((l-1)/2) + l - 1], moved
  inside the video when it would cross an end. The paper's interval [k - floor((l-1)/2),
  k + floor((l-1)/2)] holds an odd number of frames, but its Figs. 9-11 show clips of 2, 3 and 4
  frames at the l = 2, 3 and 4 resolutions.
- R(1) = 1. Eq. 7 is undefined for one frame; a lone frame counts as fully redundant, the value of
  the pairwise mean with the diagonal included. With R(1) = 0, lam_r R(l) for neighbouring frames
  (about 0.45: adjacent seconds have cosine 0.87-0.94 in the 8B embedder) outweighs the other terms
  and every clip is a single frame, which Figs. 9-11 do not show (nsp5_f2c_dryrun.py prints both).
- Valleys are local minima of r, a flat stretch counted once at its middle, never the first or last
  second (scipy.signal.find_peaks on -r); a basin runs from one valley up to the next, and its peak
  is its first highest second.
- k-means: Lloyd's algorithm on the candidate seconds with greedy k-means++ seeding, the best of 10
  runs by inertia, at most 300 iterations, seeded (scikit-learn's KMeans defaults with
  random_state = seed; scikit-learn is not installed in the harness venv). A cluster's best-score
  ties go to the earlier second, and ties in Eq. 8 to the shorter clip.
- s* >= 1: no frame is sharper than a full frame, so l >= K / K_anchor. This only binds when
  K_anchor < K, or in the host variant.
- A second that clips of different resolutions share is encoded once, at the sharper resolution
  (Sec. 6.4 says overlapping frames are encoded once; Sec. 4.2.2 only names the equal-resolution case).
- Frames reach the 28-pixel grid as F2C's do: scaled to their share (H/s x W/s) and rounded to the
  nearest grid size by Qwen's smart_resize. When rounding would take a question over its budget, the
  frames it enlarged most are rounded down instead, one at a time, until the total fits (grid_sizes).

Host variant (NS-P5, not in the paper): another placement's frame times take the place of step 2.
Each run of adjacent seconds that the placement fills is one anchor, at its second with the most
frames, with the budget of the m full frames it holds; its clip is searched over m <= l <= s_max^2 m,
l_max = s_max^2 m. With one frame in an isolated second this is F2C's step 3 exactly.

    .venv/bin/python nsp5_f2c_dryrun.py      # CPU-only check on a few questions per benchmark
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

LAM_R, LAM_L, S_MAX = 0.5, 0.05, 2.0         # App. D.3
R_SINGLE = 1.0                                # R(1); see the docstring
N_INIT, MAX_ITER = 10, 300                    # k-means, scikit-learn's KMeans defaults
EPS = 1e-9


@dataclass(frozen=True)
class Clip:
    anchor: int        # anchor second
    start: int         # first second of the clip
    frames: int        # seconds in the clip; more than `length` after a merge
    length: int        # l*, which sets the resolution
    budget: float      # the anchor's budget in full frames (K / K_anchor in F2C)

    @property
    def end(self) -> int:
        return self.start + self.frames - 1

    @property
    def scale(self) -> float:
        """Visual tokens of each frame as a fraction of a full frame's: 1 / s*^2 (Eqs. 4-5)."""
        return self.budget / self.length


def valleys(r) -> list[int]:
    """Local minima of r: runs of equal values below both neighbouring runs, each at its middle
    second; the first and last runs never count."""
    r = np.asarray(r, dtype=np.float64)
    if len(r) < 3:
        return []
    change = np.flatnonzero(np.diff(r) != 0) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [len(r)]])                 # exclusive
    vals = r[starts]
    return [int((starts[j] + ends[j] - 1) // 2) for j in range(1, len(starts) - 1)
            if vals[j] < vals[j - 1] and vals[j] < vals[j + 1]]


def candidates(r) -> np.ndarray:
    """One candidate second per basin (Algorithm 1): the first highest second between two valleys."""
    r = np.asarray(r, dtype=np.float64)
    cuts = [0] + valleys(r) + [len(r)]
    return np.array([a + int(np.argmax(r[a:b])) for a, b in zip(cuts[:-1], cuts[1:]) if b > a],
                    dtype=int)


def _kmeanspp(x: np.ndarray, k: int, rng: np.random.Generator) -> np.ndarray:
    """Greedy k-means++ seeding: each new centre is the best of 2 + ln k draws proportional to D^2."""
    n, trials = len(x), 2 + int(math.log(k))
    centres = [x[rng.integers(n)]]
    d2 = (x - centres[0]) ** 2
    for _ in range(1, k):
        total = d2.sum()
        if total <= 0:
            break
        draws = rng.choice(n, size=trials, p=d2 / total)
        pots = [np.minimum(d2, (x - x[i]) ** 2).sum() for i in draws]
        pick = int(draws[int(np.argmin(pots))])
        centres.append(x[pick])
        d2 = np.minimum(d2, (x - x[pick]) ** 2)
    return np.array(centres, dtype=np.float64)


def _lloyd(x: np.ndarray, c: np.ndarray) -> tuple[np.ndarray, float]:
    """Lloyd iterations from centres c; an emptied cluster restarts at the point farthest from its
    centre, as scikit-learn does. Returns (labels, inertia)."""
    lab = np.abs(x[:, None] - c[None, :]).argmin(1)
    for _ in range(MAX_ITER):
        new = c.copy()
        dist = np.abs(x - c[lab])
        for j in range(len(c)):
            members = lab == j
            if members.any():
                new[j] = x[members].mean()
            else:
                far = int(np.argmax(dist))
                new[j], dist[far] = x[far], -1.0
        new_lab = np.abs(x[:, None] - new[None, :]).argmin(1)
        done = np.array_equal(new_lab, lab) and np.allclose(new, c)
        c, lab = new, new_lab
        if done:
            break
    return lab, float(((x - c[lab]) ** 2).sum())


def kmeans_1d(x, k: int, seed: int = 0) -> np.ndarray:
    """Cluster labels of the points x in k groups, the best of N_INIT seeded runs by inertia."""
    x = np.asarray(x, dtype=np.float64)
    rng = np.random.default_rng(seed)
    best_lab, best_inertia = None, np.inf
    for _ in range(N_INIT):
        lab, inertia = _lloyd(x, _kmeanspp(x, k, rng))
        if inertia < best_inertia:
            best_lab, best_inertia = lab, inertia
    return best_lab


def watershed_anchors(r, k_anchor: int, seed: int = 0) -> list[int]:
    """Anchor seconds, sorted (Sec. 4.1, Algorithm 1): every basin peak, or with more than k_anchor
    of them the best-scoring peak of each of k_anchor k-means clusters over their seconds."""
    r = np.asarray(r, dtype=np.float64)
    cand = candidates(r)
    if len(cand) <= k_anchor:
        return sorted(int(c) for c in cand)
    lab = kmeans_1d(cand.astype(np.float64), k_anchor, seed)
    out = []
    for j in np.unique(lab):
        members = cand[lab == j]                               # in time order
        out.append(int(members[int(np.argmax(r[members]))]))
    return sorted(out)


def window_start(k: int, length: int, T: int) -> int:
    """First second of the `length`-second clip around anchor k, kept inside [0, T)."""
    return min(max(0, k - (length - 1) // 2), T - length)


def choose_length(r: np.ndarray, emb: np.ndarray, k: int, budget: float, s_max: float = S_MAX,
                  lam_r: float = LAM_R, lam_l: float = LAM_L,
                  r_single: float = R_SINGLE) -> tuple[int, int]:
    """(l*, first second) of the clip around anchor k by Eq. 8, over budget <= l <= s_max^2 budget."""
    T = len(r)
    l_max = s_max ** 2 * budget                                # s_max^2 K / K_anchor
    hi = min(int(math.floor(l_max + EPS)), T)
    lo = min(max(1, int(math.ceil(budget - EPS))), hi)
    starts = {ln: window_start(k, ln, T) for ln in range(lo, hi + 1)}
    a = min(starts.values())
    b = max(s + ln for ln, s in starts.items())
    e = np.asarray(emb[a:b], dtype=np.float32)
    e = e / np.maximum(np.linalg.norm(e, axis=1, keepdims=True), 1e-12)
    gram = (e @ e.T).astype(np.float64)
    best = (-np.inf, lo, starts[lo])
    for ln in range(lo, hi + 1):
        s = starts[ln]
        rel = float(np.mean(r[s:s + ln]))                                   # Eq. 6
        if ln > 1:
            g = gram[s - a:s - a + ln, s - a:s - a + ln]
            red = float((g.sum() - np.trace(g)) / (ln * (ln - 1)))          # Eq. 7
        else:
            red = r_single
        obj = rel - lam_r * red + lam_l * ln / l_max                        # Eq. 8
        if obj > best[0] + EPS:
            best = (obj, ln, s)
    return best[1], best[2]


def merge(clips: list[Clip]) -> list[Clip]:
    """Overlapping clips with identical resolutions become one clip (end of Sec. 4.2.2)."""
    groups: dict[float, list[Clip]] = {}
    for c in clips:
        groups.setdefault(round(c.scale, 9), []).append(c)
    out = []
    for group in groups.values():
        group = sorted(group, key=lambda c: (c.start, c.end))
        cur = group[0]
        for c in group[1:]:
            if c.start <= cur.end:                             # they share a frame
                end = max(cur.end, c.end)
                cur = Clip(cur.anchor, cur.start, end - cur.start + 1, cur.length, cur.budget)
            else:
                out.append(cur)
                cur = c
        out.append(cur)
    return sorted(out, key=lambda c: (c.start, -c.scale))


def placed_anchors(r: np.ndarray, placed) -> tuple[list[int], list[float]]:
    """Host variant: one anchor per run of adjacent seconds the placement fills, at the run's second
    with the most frames (ties: the more relevant, then the earlier), with the run's frame count as its
    budget in full frames."""
    secs = np.clip(np.floor(np.asarray(placed, dtype=np.float64)).astype(int), 0, len(r) - 1)
    uniq, counts = np.unique(secs, return_counts=True)
    runs = np.split(np.arange(len(uniq)), np.flatnonzero(np.diff(uniq) > 1) + 1)
    anchors, budgets = [], []
    for run in runs:
        best = max(run, key=lambda i: (counts[i], r[uniq[i]], -uniq[i]))
        anchors.append(int(uniq[best]))
        budgets.append(float(counts[run].sum()))
    return anchors, budgets


def select(r, emb, K: int, k_anchor: int = 0, placed=None, s_max: float = S_MAX,
           lam_r: float = LAM_R, lam_l: float = LAM_L, seed: int = 0,
           r_single: float = R_SINGLE) -> tuple[list[Clip], int]:
    """F2C's key clips for one question: (clips after merging, number of anchors).

    r        relevance per second, in cosine units (Eq. 2)
    emb      the per-second frame embeddings of the same encoder, at least len(r) rows
    K        the budget, in full frames
    k_anchor anchors (0: K_anchor = K, the paper's setting)
    placed   frame times (seconds) of another placement for the host variant; None: F2C's anchors
    """
    r = np.asarray(r, dtype=np.float64)
    if len(emb) < len(r):
        raise ValueError(f"F2C: {len(emb)} frame embeddings for a {len(r)}-second curve")
    if placed is None:
        ka = k_anchor or K
        anchors = watershed_anchors(r, ka, seed)
        budgets = [K / ka] * len(anchors)
    else:
        anchors, budgets = placed_anchors(r, placed)
    clips = []
    for k, b in zip(anchors, budgets):
        ln, s = choose_length(r, emb, k, b, s_max, lam_r, lam_l, r_single)
        clips.append(Clip(k, s, ln, ln, b))
    return merge(clips), len(anchors)


def frame_plan(clips: list[Clip]) -> tuple[list[int], list[float]]:
    """(second, scale) of every frame to encode, in time order, each second once: a second that clips
    of different resolutions share keeps the sharper one."""
    best: dict[int, float] = {}
    for c in clips:
        for t in range(c.start, c.end + 1):
            best[t] = max(best.get(t, 0.0), c.scale)
    secs = sorted(best)
    return secs, [best[t] for t in secs]


def native_tokens(h: int, w: int, min_pixels: int, max_pixels: int) -> float:
    """Visual tokens of a full frame at the video's own resolution: Qwen's smart_resize at the given
    pixel limits (max_pixels 0: the Qwen2.5-VL processor's 16,384-token cap)."""
    from qwen_vl_utils.vision_process import smart_resize
    hh, ww = smart_resize(h, w, factor=28, min_pixels=min_pixels,
                          max_pixels=max_pixels or 16384 * 28 * 28)
    return hh * ww / (28 * 28)


def grid_sizes(h0: int, w0: int, shares, budget: float,
               mode: str = "nearest") -> list[tuple[int, int]]:
    """(h, w) on the 28-pixel grid for frames of h0 x w0 holding `shares` visual tokens each.

    floor    the largest grid size within each share (the harness's --input images resize)
    nearest  each frame scaled to its share and rounded to the nearest grid size, as Qwen's processor
             rounds F2C's H/s x W/s frames; if the question's total would then exceed `budget`, the
             frames that rounding enlarged most take the floor size instead, one at a time, until the
             total fits (the floor sizes sum to at most the shares, so it always does)
    Neither enlarges a frame beyond the video's own resolution.
    """
    from qwen_vl_utils.vision_process import smart_resize
    lo = 4 * 28 * 28
    floor = [smart_resize(h0, w0, factor=28, min_pixels=lo, max_pixels=max(int(s * 28 * 28), lo))
             for s in shares]
    if mode == "floor":
        return floor
    out = []
    for s in shares:
        a = min(1.0, math.sqrt(s * 28 * 28 / (h0 * w0)))
        out.append(smart_resize(max(1, int(h0 * a)), max(1, int(w0 * a)), factor=28, min_pixels=lo,
                                max_pixels=1 << 40))
    tok = [h * w // (28 * 28) for h, w in out]
    total = sum(tok)
    if total > budget:
        for i in sorted(range(len(out)), key=lambda i: (-(tok[i] - shares[i]), i)):
            if total <= budget:
                break
            total -= tok[i] - floor[i][0] * floor[i][1] // (28 * 28)
            out[i] = floor[i]
    return out
