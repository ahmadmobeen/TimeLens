"""NS-P5 E3: does query-conditioned frame allocation pay off on HOUR-LONG video?

E1b and E2 measured oracle ceilings of +2.2 mIoU (Charades, ~30 s clips) and +0.5
(ActivityNet, ~2 min). Both corpora are short enough that uniform sampling already
covers them densely, so there is little to redistribute. LVBench's median video is
60.3 minutes: at a 64-frame budget the uniform sampler takes one frame every ~57 s.
If redistribution cannot win there, it cannot win anywhere.

What this measures and what it does NOT. LVBench is four-way multiple choice, so the
metric is ANSWER ACCURACY, never mIoU. This is a ceiling test for the *allocation*
principle at hour scale, not a grounding result, and the two must not be quoted as if
they were the same axis.

The sampling kernel is imported from eval_foveate rather than copied, so E3 uses
literally the same code as E1a/E1b/E2 and cannot drift from them.

Arms (all budget-matched -- identical N and identical per-frame resolution, only the
PLACEMENT of the N frames differs):
  uniform                    baseline
  gauss_s<sigma>             oracle-centred on the evidence midpoint
  offpeak_s<sigma>           placebo: same concentration, displaced off the evidence

The placebo is load-bearing. A concentrated kernel anywhere might beat uniform simply
by giving the model locally dense context; only the gap between gauss and offpeak shows
that it is *where* the mass sits that matters.

Run (TimeLens venv, from research/TimeLens, PYTHONPATH=.):
  env -u VIRTUAL_ENV CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \\
    .venv/bin/python -u eval_lvbench_foveate.py --arm uniform \\
    --out logs/nsp5/e3_uniform.jsonl
"""
from __future__ import annotations

import argparse
import threading
from collections import deque
from concurrent.futures import ThreadPoolExecutor
import json
import os
import re
import time
import zlib

import qwen_vl_utils.vision_process as _qwen_vision
import torch
from decord import VideoReader
from qwen_vl_utils import process_vision_info
from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

_torchvision_read = _qwen_vision.VIDEO_READER_BACKENDS["torchvision"]


def _torchvision_read_one_cpu(ele: dict):
    """qwen_vl_utils falls back to torchvision when decord fails on a video, and torchvision decodes
    the whole video. FFmpeg's scaler then starts one worker thread per visible CPU (224 here) for
    every scaling graph; on 2026-11-13 such readers filled the container's task limit and nothing
    in it could fork. With one visible CPU the scaler starts no workers. The affinity applies to
    the calling thread only and is restored after the read; the frames are the same."""
    old = os.sched_getaffinity(0)
    os.sched_setaffinity(0, {min(old)})
    try:
        return _torchvision_read(ele)
    finally:
        os.sched_setaffinity(0, old)


_qwen_vision.VIDEO_READER_BACKENDS["torchvision"] = _torchvision_read_one_cpu

# Same kernel code as E1a/E1b/E2 -- imported, never copied.
from eval_foveate import kernel_times, times_to_indices

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT = (
    "{}\n\n"
    "Answer with the single letter of the correct option ({}) and nothing else."
)
LETTERS = "ABCDEFGH"


def letter_list(n: int) -> str:
    """"A, B, C or D" for 4 options -- the original LVBench wording, exactly."""
    return ", ".join(LETTERS[:n - 1]) + " or " + LETTERS[n - 1]

_TR = re.compile(r"^(\d+):(\d+)(?::(\d+))?-(\d+):(\d+)(?::(\d+))?$")
_LETTER = {n: re.compile(rf"\b([{LETTERS[:n]}])\b") for n in range(2, len(LETTERS) + 1)}

# Abort an arm after this many failures with nothing scored (broken harness).
FAIL_FAST_SKIPS = 25


def parse_time_reference(text: str) -> tuple[float, float] | None:
    """`MM:SS-MM:SS` or `HH:MM:SS-HH:MM:SS` -> (start, end) seconds, else None.

    Returns zero-length spans unchanged: 250 of LVBench's 1,549 references are an
    instant (`01:01-01:01`), which is a perfectly good kernel centre. Treating those
    as malformed would silently discard 16% of the benchmark.
    """
    m = _TR.match(str(text).strip())
    if not m:
        return None
    g = m.groups()
    start = (int(g[0]) * 3600 + int(g[1]) * 60 + int(g[2])) if g[2] else (int(g[0]) * 60 + int(g[1]))
    end = (int(g[3]) * 3600 + int(g[4]) * 60 + int(g[5])) if g[5] else (int(g[3]) * 60 + int(g[4]))
    return float(start), float(end)


def parse_letter(answer: str, n_opts: int = 4) -> str | None:
    """First standalone valid option letter in the reply, else None (counted, not guessed).
    For 4 options this is the original A/B/C/D rule."""
    m = _LETTER[n_opts].search(answer.upper())
    return m.group(1) if m else None


def aks_leaves(s, n: int, t1: float, t2: float, depth_max: int) -> list[tuple[int, int, int]]:
    """AKS's judge-and-split (CVPR 2025; frame_select.meanstd, ported line for line): a segment
    stops splitting when the mean of its n best scores exceeds its mean by more than t1 (and
    its std exceeds t2); otherwise it is halved, down to depth_max. Returns (lo, hi, depth)."""
    import numpy as np
    leaves, todo = [], [(0, len(s), 0)]
    while todo:
        nxt = []
        for lo, hi, d in todo:
            seg = s[lo:hi]
            if np.mean(np.sort(seg)[::-1][:n]) - np.mean(seg) > t1 and np.std(seg) > t2:
                leaves.append((lo, hi, d))
            elif d < depth_max:
                mid = lo + (hi - lo) // 2
                nxt += [(lo, mid, d + 1), (mid, hi, d + 1)]
            else:
                leaves.append((lo, hi, d))
        todo = nxt
    return leaves


def times_to_indices_nearest(times: list[float], native_fps: float, total: int, n: int) -> list[int]:
    """times_to_indices, except that a time whose source frame is already taken gets the nearest
    unused frame (the later one on a tie), so a crowded second spills into its neighbours instead
    of the start of the video. A budget still short after that (fewer times than n) is filled
    from the start, as before."""
    taken, idx = set(), []
    for t in sorted(times):
        w = min(total - 1, max(0, round(t * native_fps)))
        for d in range(total):
            c = next((x for x in (w + d, w - d) if 0 <= x < total and x not in taken), None)
            if c is not None:
                taken.add(c)
                idx.append(c)
                break
    cursor = 0
    while len(idx) < n and cursor < total:
        if cursor not in taken:
            taken.add(cursor)
            idx.append(cursor)
        cursor += 1
    return sorted(idx)[:n]


def curve_times(z, dur: float, n: int, alloc: str, tau: float, eps: float, power: float = 3.0,
                aks: tuple[float, float, int] = (0.8, -100.0, 5),
                short_uniform: bool = False, seed: int = 0, candidates: int = 128,
                stack: int = 4, nms_radius: float = 10.0, clip_len: int = 4) -> list[float]:
    """Frame times from a per-second score curve (second k covers [k, k+1)).

    topk   the n highest-scoring seconds. A video shorter than n seconds gets one frame per
           second, and times_to_indices makes up the rest from the start of the video; with
           short_uniform it gets n uniform frames instead, as aks does.
    kelly  inverse-CDF quantiles of (1 - eps) * softmax(z / tau) + eps * uniform: frames
           in proportion to posterior mass. When accuracy grows with log(frames on the
           evidence), as E3's ladder shows, proportional allocation maximises expected
           accuracy; eps of the budget stays uniform as a hedge against a confidently
           wrong posterior (E3's placebo: concentration in the wrong place costs 5 pts).
    its    BOLT's inverse transform sampling (CVPR 2025; select_frames.py): min-max
           normalised scores to the power `power`, frames at the linspace(1/n, 1 - 1/n, n)
           quantiles, one per second at its start. A second that would take two frames (BOLT
           repeats the frame) gets them read off inside it, as kelly does, so all n differ.
    aks    AKS's adaptive keyframe sampling (CVPR 2025; frame_select.py, released defaults):
           min-max normalised scores, judge-and-split (aks_leaves), the top int(n / 2**depth)
           seconds of each leaf. A video shorter than n seconds gets n uniform frames (AKS
           returns every frame of it).
    """
    import numpy as np
    z = np.asarray(z, dtype=np.float64)[:max(1, int(np.ceil(dur)))]
    T = len(z)
    if alloc == "qframe":
        # Q-Frame (ICCV 2025, arXiv 2506.22139, Eqs. 3-5; no code released): `candidates` uniform
        # candidate frames (its protocol: 128; 0 = every second), softmax(z / tau) over them, and
        # the n largest of log-probability plus Gumbel noise, i.e. n draws without replacement.
        # The noise is seeded per question, so a rerun repeats the same frames.
        rng = np.random.default_rng(seed)
        cand = (np.array([dur * (i + 0.5) / candidates for i in range(candidates)])
                if candidates > 0 else np.arange(T) + 0.5)
        logits = z[np.minimum(cand.astype(int), T - 1)] / tau
        picks: list[float] = []
        while len(picks) < n:                    # a fresh draw only when n exceeds the candidates
            order = np.argsort(-(logits + rng.gumbel(size=len(cand))), kind="stable")
            picks += [float(cand[j]) for j in order[:n - len(picks)]]
        return sorted(min(dur - 1e-3, t) for t in picks)
    if alloc == "topk":
        if short_uniform and T < n:
            return [dur * (i + 0.5) / n for i in range(n)]
        return sorted(float(k) + 0.5 for k in np.argsort(-z)[:n])
    if alloc == "aks":
        if T < n:
            return [dur * (i + 0.5) / n for i in range(n)]
        s = (z - z.min()) / (z.max() - z.min())
        picks = []
        for lo, hi, d in aks_leaves(s, n, *aks):
            picks += [lo + int(j) for j in np.argsort(-s[lo:hi], kind="stable")[:int(n / 2 ** d)]]
        return sorted(float(k) for k in picks)
    if alloc == "its":
        s = z - z.min()
        s = (s / s.max()) ** power if s.max() > 0 else np.ones(T)
        p = s / s.sum()
        cdf = np.cumsum(p)
        grid = np.linspace(1 / n, 1 - 1 / n, n)
        ks = np.searchsorted(cdf, grid)
        out = []
        for q, k in zip(grid, ks):
            k = min(int(k), T - 1)
            if np.count_nonzero(ks == k) == 1:
                out.append(float(k))
            else:
                prev = cdf[k - 1] if k > 0 else 0.0
                out.append(min(dur - 1e-3, k + (q - prev) / p[k]))
        return out
    if alloc == "waterfill":
        # NS-P5 Study Y: whole frames n_t maximizing sum_t p_t log(1 + n_t) with sum_t n_t = n, the
        # optimum under the measured log(1 + frames) relation (greedy on marginal gains, exact for
        # this concave objective); the n_t frames of a second are spread evenly inside it.
        import heapq
        p = np.exp((z - z.max()) / tau)
        p = (1 - eps) * p / p.sum() + eps / T
        cnt = np.zeros(T, int)
        heap = [(-float(pt) * np.log(2.0), t) for t, pt in enumerate(p)]
        heapq.heapify(heap)
        for _ in range(n):
            _, t = heapq.heappop(heap)
            cnt[t] += 1
            heapq.heappush(heap, (-float(p[t]) * np.log((cnt[t] + 2) / (cnt[t] + 1)), t))
        out = [float(t) + (j + 0.5) / cnt[t] for t in np.flatnonzero(cnt) for j in range(cnt[t])]
        return sorted(min(dur - 1e-3, x) for x in out)
    if alloc in ("topk_nms", "keyclip"):
        # NS-P5 Study X5. topk_nms: the highest-scoring seconds at least nms_radius s apart.
        # keyclip: the highest peaks, each expanded to clip_len consecutive seconds (clips do not
        # overlap), one frame per second. If too few seconds qualify, the rest go by score.
        order = np.argsort(-z, kind="stable")
        blocked = np.zeros(T, bool)
        secs: list[int] = []
        if alloc == "topk_nms":
            r = max(1, int(np.ceil(nms_radius)))
            for k in order:
                if not blocked[k]:
                    secs.append(int(k))
                    blocked[max(0, k - r + 1):min(T, k + r)] = True
                    if len(secs) == n:
                        break
        else:
            c = max(1, int(clip_len))
            for k in order:
                lo = max(0, min(int(k) - c // 2, T - c))
                hi = min(T, lo + c)
                if blocked[lo:hi].any():
                    continue
                secs.extend(range(lo, hi))
                blocked[lo:hi] = True
                if len(secs) >= n:
                    break
        taken = set(secs[:n])
        for k in order:
            if len(taken) >= min(n, T):
                break
            taken.add(int(k))
        return sorted(min(dur - 1e-3, k + 0.5) for k in taken)[:n]
    if alloc == "topk_stack":
        # NS-P5 Study T: top-k that may stack frames inside a second, as CaliFrame can: the
        # ceil(n / stack) highest-scoring seconds, `stack` frames evenly inside each.
        m = max(1, min(T, int(np.ceil(n / stack))))
        c = int(np.ceil(n / m))
        out = [float(k) + (j + 0.5) / c for k in sorted(np.argsort(-z, kind="stable")[:m])
               for j in range(c)]
        return sorted(min(dur - 1e-3, t) for t in out)[:n]
    if alloc == "kelly_cap1" and T > n:
        # NS-P5 Study T: CaliFrame's probability with at most one frame per second. Water-filling
        # with capacity one, q_t = min(1, lam p_t) with sum q = n; seconds with q_t = 1 take a
        # frame, the rest are placed at the quantiles of the remaining mass (q_t < 1, so at most
        # one frame lands in a second). Videos with no more seconds than n fall through to kelly.
        p = np.exp((z - z.max()) / tau)
        p = (1 - eps) * p / p.sum() + eps / T
        lo_l, hi_l = np.log(n), np.log(n / p.min() + 1.0)
        for _ in range(200):
            mid = (lo_l + hi_l) / 2
            if np.minimum(1.0, np.exp(mid) * p).sum() < n:
                lo_l = mid
            else:
                hi_l = mid
        q = np.minimum(1.0, np.exp(hi_l) * p)
        full = np.flatnonzero(q >= 1.0 - 1e-9)
        rest = q.copy()
        rest[full] = 0.0
        m = n - len(full)
        picks = set(int(k) for k in full)
        if m > 0 and rest.sum() > 0:
            cdf = np.cumsum(rest)
            for i in range(m):
                picks.add(min(int(np.searchsorted(cdf, (i + 0.5) * cdf[-1] / m)), T - 1))
        for k in np.argsort(-p, kind="stable"):    # rounding left a gap: next most probable
            if len(picks) >= n:
                break
            picks.add(int(k))
        return sorted(float(k) + 0.5 for k in picks)[:n]
    p = np.exp((z - z.max()) / tau)
    p = (1 - eps) * p / p.sum() + eps / T
    cdf = np.cumsum(p)
    out = []
    for i in range(n):
        q = (i + 0.5) / n
        k = min(int(np.searchsorted(cdf, q)), T - 1)
        prev = cdf[k - 1] if k > 0 else 0.0
        out.append(min(dur - 1e-3, k + (q - prev) / p[k]))
    return out


def dose_times(spans, dur: float, n: int, k: int, margin: float = 30.0) -> list[float]:
    """NS-P5 Study V: exactly k frames inside the annotated evidence, spread evenly over the evidence
    seconds (several spans are treated as one), and the other n - k spread evenly over the video
    away from the evidence by at least `margin` seconds (the margin shrinks if nothing is left)."""
    import numpy as np
    from t1_data import span_seconds
    T = max(1, int(np.ceil(dur)))
    inside = np.zeros(T, bool)
    for a, b in spans:
        lo, hi = span_seconds(a, b, T)
        inside[lo:hi] = True
    if not inside.any():
        raise ValueError("dose arm: question without evidence seconds")
    def spread(mask: np.ndarray, m: int) -> list[float]:
        secs = np.flatnonzero(mask)
        return [float(secs[min(int((i + 0.5) * len(secs) / m), len(secs) - 1)])
                + ((i + 0.5) * len(secs) / m) % 1.0 for i in range(m)]
    out = spread(inside, k) if k > 0 else []
    for mg in (margin, margin / 2, 0.0):
        away = ~inside
        for s in np.flatnonzero(inside):
            away[max(0, s - int(mg)):min(T, s + int(mg) + 1)] = False
        if away.sum() > 0:
            break
    if n - k > 0:
        out += spread(away, n - k) if away.any() else spread(np.ones(T, bool), n - k)
    return sorted(min(dur - 1e-3, t) for t in out)


@torch.no_grad()
def prior_center(frames, duration: float, question: str, model, proc,
                 true_center: float, smooth: int = 5) -> tuple[float, float]:
    """Predicted evidence centre from query-frame similarity, and its error.

    `frames` are the uniform reference frames already decoded for this item, so the
    prior reads nothing new off disk and the expensive VLM still runs exactly once --
    the single-pass constraint this project is built around.

    The option list is stripped from the question: only the stem says what to look
    for, and the three distractors would blur the query embedding.

    Returns (centre in seconds, |centre - true centre|). The error is logged per item
    so the prior's localization quality can be read independently of whether the
    downstream answer happened to come out right.
    """
    stem = question.split("\n(A)")[0].strip() or question.strip()
    n = frames.shape[0]
    imgs = [frames[i].permute(1, 2, 0).to(torch.uint8).cpu().numpy() for i in range(n)]
    batch = proc(text=[stem], images=imgs, padding="max_length", truncation=True,
                 return_tensors="pt").to("cuda")
    sim = model(**batch).logits_per_text[0].float()            # one score per frame
    if n >= smooth:                                            # join single-frame spikes
        k = torch.ones(1, 1, smooth, device=sim.device) / smooth
        sm = torch.nn.functional.conv1d(sim[None, None], k, padding=smooth // 2)[0, 0][:n]
        sim = torch.maximum(sm, sim)
    centre = duration * (int(torch.argmax(sim)) + 0.5) / n     # uniform frame times
    return centre, abs(centre - true_center)


def load_questions(parquet: str, meta_jsonl: str, video_dir: str) -> tuple[list[dict], dict]:
    """Scoreable LVBench questions plus a census of everything dropped and why."""
    import pandas as pd

    durations: dict[str, float] = {}
    with open(meta_jsonl, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                durations[rec["key"]] = float(rec["video_info"]["duration_minutes"]) * 60.0

    df = pd.read_parquet(parquet)
    items: list[dict] = []
    census = {"total": len(df), "no_duration": 0, "no_video": 0,
              "unparseable_span": 0, "reversed_span": 0, "span_past_end": 0}

    for row in df.itertuples(index=False):
        if row.key not in durations:
            census["no_duration"] += 1
            continue
        path = os.path.join(video_dir, row.video_path)
        if not os.path.exists(path):
            census["no_video"] += 1
            continue
        span = parse_time_reference(row.time_reference)
        if span is None:
            census["unparseable_span"] += 1
            continue
        start, end = span
        if end < start:
            census["reversed_span"] += 1          # 13 of them; dropped, never silently swapped
            continue
        duration = durations[row.key]
        if start < 0 or end > duration + 1.0:
            census["span_past_end"] += 1
            continue
        items.append({"key": row.key, "uid": str(row.uid), "video_path": path,
                      "question": row.question, "gold": str(row.answer).strip().upper(),
                      "center": (start + end) / 2.0, "duration": duration,
                      "span": [start, end]})
    census["scoreable"] = len(items)
    census["videos"] = len({i["key"] for i in items})
    return items, census


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", required=True, help="uniform | gauss_s<sigma> | offpeak_s<sigma>")
    ap.add_argument("--offpeak_shift", type=float, default=600.0,
                    help="placebo displacement in seconds; reported as realized at the end")
    ap.add_argument("--n_frames", type=int, default=64,
                    help="frame budget, identical across arms")
    # The reference read's pixel settings. The defaults are this harness's protocol (14,336
    # visual tokens shared by the frames); Study C (METHODS-LOG 2026-11-05) sets lmms-eval's
    # Qwen2.5-VL defaults instead, which qwen_vl_utils caps at 768 tokens per frame.
    ap.add_argument("--min_pixels", type=int, default=64 * 28 * 28,
                    help="reference read: per-frame pixel floor")
    ap.add_argument("--total_pixels", type=int, default=14336 * 28 * 28,
                    help="reference read: pixel budget shared by the frames (0: qwen_vl_utils' default)")
    ap.add_argument("--max_pixels", type=int, default=0,
                    help="reference read: per-frame pixel cap to request (0: none requested)")
    ap.add_argument("--model", default=MODEL)
    ap.add_argument("--parquet",
                    default="/home/mobeen/codes/active_repos/NeedleScan-vtg-landscape/"
                            "research/TimeLens/data/lvbench/data/train-00000-of-00001.parquet")
    ap.add_argument("--meta",
                    default="/home/mobeen/codes/active_repos/NeedleScan-vtg-landscape/"
                            "research/TimeLens/data/lvbench/video_info.meta.jsonl",
                    help="duration metadata; kept beside the parquet on NFS because "
                         "/tmp does not survive a pod restart")
    ap.add_argument("--video_dir",
                    default="/home/mobeen/codes/active_repos/NeedleScan-vtg-landscape/"
                            "research/TimeLens/data/lvbench/videos")
    ap.add_argument("--out", required=True)
    ap.add_argument("--center", choices=["gt", "prior"], default="gt",
                    help="gt = oracle evidence midpoint (a CEILING). prior = SigLIP "
                         "query-frame similarity over the reference frames already "
                         "decoded, i.e. an achievable method with no extra video decode.")
    ap.add_argument("--prior_model", default="google/siglip-so400m-patch14-384")
    ap.add_argument("--scan_frames", type=int, default=256,
                    help="candidate frames scored for the topk_siglip arm")
    ap.add_argument("--shuffle_query", action="store_true",
                    help="PLACEBO: score frames with a DIFFERENT question's text while "
                         "answering the real one. Measures how much of a selector's gain "
                         "is actually query-driven. Swift Sampling (2605.22678) scores "
                         "with no query at all and matches AKS on easier benchmarks; "
                         "nobody has run this on LVBench, where query conditioning is "
                         "supposedly worth +7 to +12.")
    ap.add_argument("--curves", default="",
                    help="curve arm: npz of per-second score curves keyed "
                         "'lvbench:<key>>>><uid>' (t1_train.py eval_lvbench writes them)")
    ap.add_argument("--qframe_candidates", type=int, default=128,
                    help="qframe: uniform candidate frames (Q-Frame's protocol: 128); 0 = every second")
    ap.add_argument("--alloc", choices=["topk", "kelly", "its", "aks", "qframe", "kelly_cap1",
                                        "topk_stack", "topk_nms", "keyclip", "waterfill"],
                    default="kelly",
                    help="curve arm: how a curve becomes frame times (see curve_times); its = "
                         "BOLT's inverse transform sampling, aks = AKS's judge-and-split")
    ap.add_argument("--power", type=float, default=3.0, help="its: BOLT's exponent")
    ap.add_argument("--nms_radius", type=float, default=10.0,
                    help="topk_nms: chosen seconds lie at least this far apart (Study X5)")
    ap.add_argument("--clip_len", type=int, default=4,
                    help="keyclip: seconds in each clip around a peak (Study X5)")
    ap.add_argument("--timing", action="store_true",
                    help="add t_read and t_answer (seconds) to each row; use with --prefetch 0 (Study X1)")
    ap.add_argument("--stack", type=int, default=4,
                    help="topk_stack: frames placed inside each selected second (Study T)")
    ap.add_argument("--dose_k", type=int, default=-1,
                    help="--arm dose: frames placed inside the annotated evidence (Study V)")
    ap.add_argument("--dose_margin", type=float, default=30.0,
                    help="--arm dose: the other frames stay this many seconds away from the evidence")
    ap.add_argument("--aks_t1", type=float, default=0.8, help="aks: split threshold (released default)")
    ap.add_argument("--aks_t2", type=float, default=-100.0, help="aks: std threshold (released default)")
    ap.add_argument("--aks_depth", type=int, default=5, help="aks: maximum depth (released default)")
    ap.add_argument("--refill", choices=["start", "nearest"], default="start",
                    help="a placed time whose source frame is taken: restore the budget from the "
                         "start of the video (times_to_indices) or take the nearest unused frame")
    ap.add_argument("--topk_short", choices=["refill", "uniform"], default="refill",
                    help="topk on a video shorter than the frame budget: one frame per second plus "
                         "the refill, or the budget spread uniformly over the video")
    ap.add_argument("--tau", type=float, default=1.0, help="kelly: posterior temperature")
    ap.add_argument("--eps", type=float, default=0.5, help="kelly: uniform share of the budget")
    ap.add_argument("--curve_placebo", action="store_true",
                    help="curve arm PLACEBO: allocate with ANOTHER question's curve from the "
                         "SAME video -- same video, same allocator, wrong question")
    ap.add_argument("--bench", default="lvbench",
                    choices=["lvbench", "videomme", "lvb", "mlvu", "cgbench", "cgbench_fresh"],
                    help="question set; non-LVBench sets come through bench.py")
    ap.add_argument("--seed", type=int, default=0,
                    help="fixes the --shuffle_query rotation")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on the first N questions")
    ap.add_argument("--sample", type=int, default=0,
                    help="a seeded random subset of N questions (kept in video order, so the "
                         "per-video reference memo still applies); --limit takes the first N, "
                         "which covers only a handful of videos")
    ap.add_argument("--shard", default="",
                    help="K/N: answer only the videos whose order of first appearance is K mod N, "
                         "so N processes can split one arm; a video's questions stay together "
                         "and the per-question computation is unchanged")
    ap.add_argument("--prefetch", type=int, default=0,
                    help="read the frames of the next K questions in K threads while the GPU "
                         "answers; the reads are deterministic, so every output is unchanged")
    ap.add_argument("--cpu_threads", type=int, default=0,
                    help="cap torch's CPU threads and each video reader's decode threads at N; "
                         "0 keeps the library defaults, which start one thread per visible core "
                         "(224 on the pods) whatever the container's CPU quota. Outputs are "
                         "unchanged: 30 logged CG-Bench questions gave identical answers and "
                         "option probabilities with 4, alone and with --prefetch 4 (2026-11-10)")
    ap.add_argument("--skip_done", nargs="*", default=[],
                    help="other jsonl files whose questions count as done (a shard resuming "
                         "past the rows its arm wrote before it was split)")
    args = ap.parse_args()
    if args.cpu_threads:
        torch.set_num_threads(args.cpu_threads)
    if args.prefetch and (args.center == "prior" or args.arm == "topk_siglip"):
        ap.error("--prefetch reads on the CPU only; --center prior and --arm topk_siglip use the GPU there")

    if args.bench == "lvbench":
        items, census = load_questions(args.parquet, args.meta, args.video_dir)
        for it in items:
            it.update(rkey=f"{it['key']}>>>{it['uid']}", qkey=f"lvbench:{it['key']}>>>{it['uid']}",
                      n_opts=4, spans=[it["span"]])
    else:
        import bench as _bench
        items = []
        for q in _bench.load(args.bench):
            spans = [list(sp) for sp in q.spans]
            dur = float(q.duration or 0.0)
            items.append({"key": q.vid, "uid": q.key, "video_path": q.video_path,
                          "question": q.prompt_question(), "gold": q.gold, "duration": dur,
                          "center": (spans[0][0] + spans[0][1]) / 2 if spans else dur / 2,
                          "span": spans[0] if spans else None, "spans": spans,
                          "rkey": q.key, "qkey": q.key, "n_opts": len(q.options)})
        census = {"bench": args.bench, "scoreable": len(items),
                  "videos": len({it["key"] for it in items})}
    print(f"population: {census}", flush=True)
    if args.sample:
        import random as _rs
        keep = set(_rs.Random(args.seed).sample(range(len(items)), args.sample))
        items = [it for i, it in enumerate(items) if i in keep]
        print(f"sample: {len(items)} questions from {len({it['key'] for it in items})} videos "
              f"(seed {args.seed})", flush=True)
    if args.limit:
        items = items[: args.limit]

    # Resume: skip questions already written, so a killed run continues where it stopped.
    done: set[str] = set()
    for path in [args.out] + args.skip_done:
        if os.path.exists(path):
            with open(path, encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        done.add(json.loads(line)["key"])
    print(f"arm={args.arm} n_frames={args.n_frames} todo={len(items) - len(done)}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model, padding_side="left",
                                              trust_remote_code=True)
    # Pick the video contract by CAPABILITY, not by model name. transformers 4.57 routes
    # stock Qwen2_5_VLProcessor through `self.video_processor`, which takes videos=[frames]
    # with a separate video_metadata= kwarg. The legacy [(frames, metadata)] tuple form that
    # TimeLens-7B's bundled processor still accepts is flattened to an empty list there and
    # dies as `IndexError: list index out of range` inside convert_pil_frames_to_video --
    # once per item, which the per-item handler below swallows as a skip, so the run exits 0
    # having produced nothing. That is exactly what happened on the first E3 attempt.
    new_video_api = getattr(processor, "video_processor", None) is not None
    print(f"processor={type(processor).__name__} "
          f"contract={'video_metadata kwarg' if new_video_api else 'legacy tuple'}", flush=True)
    # Thinking-capable templates (Qwen3.5) open the reply with a reasoning block, but the reply
    # is capped at 16 new tokens and parsed for a letter, and p_letters is read at the first
    # generated token. Only a template that exposes the switch receives it, so every other
    # answerer's prompt is byte-identical to before.
    chat_kwargs = ({"enable_thinking": False}
                   if "enable_thinking" in str(getattr(processor, "chat_template", None) or "")
                   else {})
    if chat_kwargs:
        print(f"chat template kwargs: {chat_kwargs}", flush=True)
    # A timm vision tower (Perception-LM's) takes no attention switch from transformers and runs
    # timm's own fused attention, so only its language model is set to sdpa. Every other
    # answerer loads exactly as before.
    vision_type = getattr(getattr(AutoConfig.from_pretrained(args.model, trust_remote_code=True),
                                  "vision_config", None), "model_type", "")
    attn = ({"text_config": "sdpa", "vision_config": "eager"} if vision_type == "timm_wrapper"
            else "sdpa")
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation=attn,
        device_map="cuda", trust_remote_code=True).eval()

    # the placebo pairs each item with another item's question text, fixed by seed
    scoring_text = {}
    if args.shuffle_query:
        import random as _r
        qs = [it["question"] for it in items]
        rot = _r.Random(args.seed + 1).randrange(1, max(len(qs), 2))
        for i, it in enumerate(items):
            scoring_text[it["key"] + it["uid"]] = qs[(i + rot) % len(qs)]
        print(f"PLACEBO: scoring with another question's text (rotation {rot})", flush=True)

    curves, curve_of = None, {}
    if args.arm == "curve":
        import numpy as np
        curves = np.load(args.curves)
        by_video: dict[str, list[str]] = {}
        qkey_of = {it["rkey"]: it["qkey"] for it in items}
        for it in items:
            by_video.setdefault(it["key"], []).append(it["rkey"])
        for keys in by_video.values():
            if args.curve_placebo and len(keys) < 2:
                continue                         # no other question about this video
            for i, k in enumerate(keys):         # placebo: the next question, same video
                src = keys[(i + 1) % len(keys)] if args.curve_placebo else k
                curve_of[k] = qkey_of[src]
        if args.curve_placebo:
            # A video with a single question has no same-video partner, so the placebo is
            # undefined there (CG-Bench has such videos; LVBench does not). Those questions are
            # left out of this arm and compared on the questions both arms share.
            n_all = len(items)
            items = [it for it in items if it["rkey"] in curve_of]
            print(f"PLACEBO: skipped {n_all - len(items)} questions from single-question videos; "
                  f"{len(items)} remain", flush=True)
        print(f"curves: {args.curves} alloc={args.alloc} tau={args.tau} eps={args.eps}"
              f"{' PLACEBO (same video, other question)' if args.curve_placebo else ''}",
              flush=True)

    if args.shard:
        # After the placebo pairing, which needs every question of a video, and by whole video,
        # so the per-video reference read is still decoded once.
        shard_k, shard_n = (int(x) for x in args.shard.split("/"))
        vids = list(dict.fromkeys(it["key"] for it in items))
        keep = {v for i, v in enumerate(vids) if i % shard_n == shard_k}
        items = [it for it in items if it["key"] in keep]
        print(f"shard {shard_k}/{shard_n}: {len(items)} questions from {len(keep)} videos, "
              f"{sum(it['rkey'] not in done for it in items)} still to answer", flush=True)

    prior_model = prior_proc = None
    if args.center == "prior" or args.arm == "topk_siglip":
        from transformers import AutoModel
        prior_proc = AutoProcessor.from_pretrained(args.prior_model)
        prior_model = AutoModel.from_pretrained(
            args.prior_model, torch_dtype=torch.float16).to("cuda").eval()
        print(f"prior encoder: {args.prior_model}", flush=True)

    letter_ids = [processor.tokenizer.encode(c, add_special_tokens=False) for c in LETTERS]
    assert all(len(t) == 1 for t in letter_ids), letter_ids
    letter_ids = [t[0] for t in letter_ids]

    realized: list[float] = []
    n_done = skipped = correct = unparsed = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    # The reference call depends only on the video, and questions arrive grouped by
    # video (~15 each), so it is decoded once per video rather than once per question.
    # Same tensors, same metadata: a speedup, not a protocol change. With --prefetch each
    # reading thread keeps its own memo and reader, and the reads run ahead of the answers.
    local = threading.local()

    def read(item: dict) -> dict:
        """Steps 1 and 2 for one question: the reference read, the placement, and the frames at
        the chosen indices. Deterministic and CPU-only (bar --center prior and topk_siglip, which
        --prefetch refuses), so with --prefetch it runs ahead of the GPU in a thread."""
        if not hasattr(local, "ref_memo"):
            local.ref_memo, local.vr_memo = {}, {}
        ref_memo, vr_memo = local.ref_memo, local.vr_memo
        key = item["rkey"]
        # 1) Reference call at the frame budget: fixes N and the per-frame
        #    resolution. Both arms inherit them, so only placement differs.
        if item["video_path"] not in ref_memo:
            ref_video = {"type": "video", "video": item["video_path"],
                         "nframes": args.n_frames, "min_pixels": args.min_pixels}
            if args.total_pixels:
                ref_video["total_pixels"] = args.total_pixels
            if args.max_pixels:
                ref_video["max_pixels"] = args.max_pixels
            ref_msgs = [{"role": "user", "content": [ref_video, {"type": "text", "text": "x"}]}]
            _, ref_videos = process_vision_info(ref_msgs, return_video_metadata=True)
            ref_memo.clear()
            ref_memo[item["video_path"]] = ref_videos[0]
        ref_tensor, meta = ref_memo[item["video_path"]]
        n, _, H, W = ref_tensor.shape
        native_fps = float(meta["fps"])
        total = int(meta["total_num_frames"])
        dur = total / native_fps

        # 2) Place N frames by the kernel, then read exactly those indices.
        if args.center == "prior":
            center, prior_err = prior_center(
                ref_tensor, dur, item["question"], prior_model, prior_proc,
                item["center"])
        else:
            center = min(max(item["center"], 0.0), dur)
            prior_err = None
        if args.arm in ("uniform", "topk_siglip", "curve", "dose"):
            # topk_siglip only needs n and the frame grid here; the actual
            # indices are replaced by the SigLIP selection below.
            times = kernel_times("uniform", center, dur, n)
        else:
            times = kernel_times(args.arm, center, dur, n,
                                 offpeak_shift=args.offpeak_shift)
        if args.arm.startswith("offpeak"):
            realized.append(abs(min(dur, center + args.offpeak_shift) - center))
        if args.arm == "dose":
            times = dose_times(item["spans"], dur, n, args.dose_k, args.dose_margin)
        if args.arm == "curve":
            times = curve_times(curves[curve_of[key]], dur, n, args.alloc,
                                args.tau, args.eps, power=args.power,
                                aks=(args.aks_t1, args.aks_t2, args.aks_depth),
                                short_uniform=args.topk_short == "uniform",
                                seed=zlib.crc32(f"{args.seed}:{key}".encode()),
                                candidates=args.qframe_candidates, stack=args.stack,
                                nms_radius=args.nms_radius, clip_len=args.clip_len)
        idx = (times_to_indices_nearest if args.refill == "nearest" else times_to_indices)(
            times, native_fps, total, n)
        # frames the budget refill supplied: times that fell on an already-taken source
        # frame, plus any shortfall (from the start of the video, or with --refill nearest
        # from the nearest unused frames)
        n_refill = (n - len({min(total - 1, max(0, round(t * native_fps))) for t in times})
                    if args.arm == "curve" else None)
        if vr_memo.get("path") != item["video_path"]:   # one reader per video
            vr_memo.update(path=item["video_path"],
                           vr=VideoReader(item["video_path"], num_threads=args.cpu_threads))
        vr = vr_memo["vr"]

        if args.arm == "topk_siglip":
            # Score a denser uniform scan, keep the n best frames. This is the
            # field's dominant recipe: plain argmax top-k captures ~88% of the
            # reported gain, every published aggregation adding ~+0.9 on top.
            scan = [min(total - 1, int(total * (i + 0.5) / args.scan_frames))
                    for i in range(args.scan_frames)]
            sf = torch.from_numpy(vr.get_batch(scan).asnumpy()).permute(0, 3, 1, 2)
            sf = torch.nn.functional.interpolate(
                sf.float(), size=(384, 384), mode="bilinear",
                align_corners=False, antialias=True).clamp(0, 255)
            qtext = scoring_text.get(item["key"] + item["uid"], item["question"])
            stem = qtext.split("\n(A)")[0].strip() or qtext.strip()
            with torch.no_grad():
                bt = prior_proc(text=[stem],
                                images=[sf[i].permute(1, 2, 0).to(torch.uint8).cpu().numpy()
                                        for i in range(sf.shape[0])],
                                padding="max_length", truncation=True,
                                return_tensors="pt").to("cuda")
                sim = prior_model(**bt).logits_per_text[0].float()
            best = torch.topk(sim, min(n, len(scan))).indices.tolist()
            idx = sorted({scan[i] for i in best})

        frames = torch.from_numpy(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2)
        frames = torch.nn.functional.interpolate(
            frames.float(), size=(H, W), mode="bicubic",
            align_corners=False, antialias=True).clamp(0, 255).to(ref_tensor.dtype)
        vmeta = {"fps": native_fps, "frames_indices": idx,
                 "total_num_frames": total, "video_backend": "decord"}
        return {"center": center, "prior_err": prior_err, "dur": dur, "idx": idx,
                "n_refill": n_refill, "native_fps": native_fps, "frames": frames, "vmeta": vmeta}

    pending = [item for item in items if item["rkey"] not in done]
    pool = ThreadPoolExecutor(args.prefetch) if args.prefetch else None
    ahead: deque = deque()
    submitted = 0

    with open(args.out, "a", encoding="utf-8") as out_fh:
        for j, item in enumerate(pending):
            key = item["rkey"]
            try:
                t_read0 = time.perf_counter()
                if pool:   # keep the next --prefetch reads in flight, consume them in order
                    while submitted < len(pending) and submitted <= j + args.prefetch:
                        ahead.append(pool.submit(read, pending[submitted]))
                        submitted += 1
                    r = ahead.popleft().result()
                else:
                    r = read(item)
                t_read = time.perf_counter() - t_read0
                center, prior_err, dur, idx = r["center"], r["prior_err"], r["dur"], r["idx"]
                n_refill, native_fps = r["n_refill"], r["native_fps"]
                frames, vmeta = r["frames"], r["vmeta"]

                # 3) Ask the multiple-choice question over exactly those frames.
                messages = [{"role": "user", "content": [
                    {"type": "video", "video": item["video_path"]},
                    {"type": "text", "text": PROMPT.format(item["question"],
                                                          letter_list(item["n_opts"]))}]}]
                text = processor.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True, **chat_kwargs)
                if new_video_api:
                    inputs = processor(text=[text], images=None, videos=[frames],
                                       video_metadata=[vmeta], do_sample_frames=False,
                                       padding=True, return_tensors="pt").to("cuda")
                else:
                    inputs = processor(text=[text], images=None, videos=[(frames, vmeta)],
                                       padding=True, return_tensors="pt").to("cuda")
                t_gen0 = time.perf_counter()
                gen = model.generate(**inputs, do_sample=False, temperature=None,
                                     top_p=None, top_k=None, max_new_tokens=16,
                                     output_scores=True, return_dict_in_generate=True)
                if args.timing:
                    torch.cuda.synchronize()
                timing_fields = ({"t_read": round(t_read, 4),
                                  "t_answer": round(time.perf_counter() - t_gen0, 4)}
                                 if args.timing else {})
                out_ids = gen.sequences
                reply = processor.batch_decode(
                    [out_ids[0][inputs.input_ids.shape[-1]:]], skip_special_tokens=True)[0]
                # The answer distribution at the first generated token: the answerer's own
                # confidence, logged so it can serve as a sensor (greedy output unchanged).
                p_letters = torch.softmax(
                    gen.scores[0][0].float()[letter_ids[:item["n_opts"]]], 0).tolist()
            except Exception as ex:  # one bad video must not kill the run
                skipped += 1
                print(f"  [skip {skipped}] {item['key']} | {ex!r}", flush=True)
                # ... but EVERY video failing is a broken harness, not bad data. Without
                # this the run churns through all 1,534 items, writes nothing and exits 0,
                # which reads downstream as "this arm scored nothing" rather than "this arm
                # never ran". Fail loudly while the cause is still on screen.
                if n_done == 0 and skipped >= FAIL_FAST_SKIPS:
                    raise RuntimeError(
                        f"{skipped} consecutive failures with no successful item; "
                        f"last error: {ex!r}. The harness is broken, not the data."
                    ) from ex
                continue

            pred = parse_letter(reply, item["n_opts"])
            if pred is None:
                unparsed += 1
            is_correct = pred is not None and pred == item["gold"]
            correct += int(is_correct)
            n_done += 1
            out_fh.write(json.dumps({
                "key": key, "arm": args.arm, "pred_letter": pred, "gold_letter": item["gold"],
                "correct": is_correct, "raw": reply.strip()[:80], "center_s": center,
                "prior_abs_err_s": prior_err,
                "span": item["span"], "duration_s": dur, "n_frames": len(idx), "n_refill": n_refill,
                "in_span": (sum(any(a - 0.5 <= i / native_fps <= b + 0.5 for a, b in item["spans"])
                                for i in idx) if item["spans"] else None),
                "p_letters": [round(p, 5) for p in p_letters], **timing_fields}) + "\n")
            out_fh.flush()
            if n_done % 50 == 0:
                print(f"  [{n_done}/{len(items)}] acc {correct / n_done:.4f} "
                      f"(skipped {skipped}, unparsed {unparsed})", flush=True)

    if pool:
        pool.shutdown(wait=False, cancel_futures=True)
    acc = correct / n_done if n_done else float("nan")
    print(f"DONE {args.arm}: n={n_done} acc={acc:.4f} skipped={skipped} "
          f"unparsed={unparsed} -> {args.out}", flush=True)
    if realized:
        # A placebo clamped back onto the evidence would look like a successful null,
        # so report the displacement actually achieved before anyone interprets it.
        got = sorted(realized)
        print(f"  offpeak displacement: requested {args.offpeak_shift:.0f}s | "
              f"median {got[len(got) // 2]:.0f}s | min {got[0]:.0f}s | max {got[-1]:.0f}s",
              flush=True)


if __name__ == "__main__":
    main()
