#!/usr/bin/env python3
"""CPU-only dry run of F2C's selection (nsp5_f2c.py) as eval_lvbench_foveate.py --res_alloc f2c runs
it, on a few questions per benchmark. No video is read and no model runs: each question's duration is
its curve length, and the 28-pixel grid is checked at assumed native frame sizes.

For each question it prints the clips (first second-last second, l), every chosen second, the tokens of
each frame and the question's total, and checks the total against the budget for:
  ours      F2C at the harness budget: K = 64 full frames of 14,336 / 64 = 224 tokens
  k32       F2C at the harness budget with F2C's largest K, 32 (448 tokens a full frame)
  released  F2C's setting: K = 32 full frames at the video's own resolution (lmms-eval's limits)
  host      CaliFrame's placement (maxopt, kelly, tau 3.58, eps 0) as the anchors of F2C's clip rule

    env CUDA_VISIBLE_DEVICES= HF_HUB_OFFLINE=1 PYTHONPATH=. .venv/bin/python -u nsp5_f2c_dryrun.py
"""
from __future__ import annotations

import argparse
import collections
import os
import time

import numpy as np

import nsp5_f2c as f2c

PRIORS = "data/t1/priors/qwen3vl_emb_8b"
BENCHES = ["lvbench", "videomme", "lvb", "mlvu", "cgbench"]
BUDGET = 14336.0
SIZES = [(720, 1280), (360, 640)]           # assumed native frames (H, W) for the grid check
LMMS_MIN, LMMS_MAX = 256 * 28 * 28, 1605632  # lmms-eval's Qwen2.5-VL pixel limits


def vid_of(bench: str, keys: list[str]) -> dict[str, str]:
    if bench == "lvbench":                     # lvbench:<video>>>><uid>
        return {k: k.split(":", 1)[1].split(">>>")[0] for k in keys}
    import bench as _bench
    return {q.key: q.vid for q in _bench.load(bench)}


def pick(keys: list[str], vids: dict[str, str], lengths: dict[str, int], n: int) -> list[str]:
    """n questions on n different videos, at evenly spaced quantiles of curve length."""
    by_len = sorted((k for k in keys if k in vids), key=lambda k: (lengths[k], k))
    out, seen = [], set()
    for q in np.linspace(0.02, 0.98, n):
        for k in by_len[int(q * (len(by_len) - 1)):] + by_len:
            if vids[k] not in seen:
                out.append(k)
                seen.add(vids[k])
                break
    return out


def ranges(clips: list) -> str:
    return " ".join(f"{c.start}-{c.end}(l{c.length}" + (f",b{c.budget:g}" if c.budget != 1 else "")
                    + ")" for c in clips)


def report(tag: str, clips: list, n_anchor: int, full: float, budget: float,
           sizes: list[tuple[int, int]], verbose: bool) -> dict:
    secs, scales = f2c.frame_plan(clips)
    shares = full * np.asarray(scales)
    total = float(shares.sum())
    lens = collections.Counter(c.length for c in clips)
    shared = sum(c.frames for c in clips) - len(secs)          # seconds that two clips claim
    grid = {(hw, mode): sum(h * w // 784 for h, w in f2c.grid_sizes(hw[0], hw[1], shares, budget, mode))
            for hw in sizes for mode in ("nearest", "floor")}
    ok = total <= budget + 1e-6 and all(g <= budget for g in grid.values())
    per_l = sorted({(c.length, round(full * c.scale, 1)) for c in clips})
    print(f"  {tag}: {n_anchor} anchors -> {len(clips)} clips, {len(secs)} frames ({shared} seconds "
          f"shared by two clips), l {dict(sorted(lens.items()))}")
    print(f"    tokens a frame (l: share) {', '.join(f'{ln}: {t}' for ln, t in per_l)}; total "
          f"{total:,.1f} of {budget:,.0f}; grid nearest/floor "
          + ", ".join(f"{h}x{w} {grid[((h, w), 'nearest')]:,}/{grid[((h, w), 'floor')]:,}"
                      for h, w in sizes)
          + f" -> {'OK' if ok else 'OVER BUDGET'}")
    if verbose:
        print(f"    clips: {ranges(clips)}")
        print(f"    seconds: {secs}")
    return {"ok": ok, "lens": lens, "frames": len(secs), "shared": shared, "total": total,
            "near": grid[(sizes[0], "nearest")], "floor": grid[(sizes[0], "floor")],
            "anchors": n_anchor, "budget": budget}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--benches", nargs="*", default=BENCHES)
    ap.add_argument("--n", type=int, default=5, help="questions per benchmark")
    ap.add_argument("--quiet", action="store_true", help="omit the per-second lists")
    args = ap.parse_args()
    from eval_lvbench_foveate import curve_times      # the harness's own placement code

    agg: dict[str, list[dict]] = collections.defaultdict(list)
    cf = collections.defaultdict(collections.Counter)   # R(1) counterfactuals, "ours" setting
    secs_per_q = []
    for bench in args.benches:
        tag = "" if bench == "lvbench" else f"{bench}_"
        concat = np.load(os.path.join(PRIORS, f"{tag}curves_concat.npz"))
        maxopt = np.load(os.path.join(PRIORS, f"{tag}curves_maxopt.npz"))
        img = np.load(os.path.join(PRIORS, f"{bench}_img.npz"))
        keys = sorted(concat.files)
        vids = vid_of(bench, keys)
        lengths = {k: len(concat[k]) for k in keys}
        have = set(img.files)
        chosen = pick([k for k in keys if vids.get(k) in have], vids, lengths, args.n)
        print(f"\n===== {bench}: {len(keys)} curves; {len(chosen)} questions =====", flush=True)
        for k in chosen:
            z = concat[k].astype(np.float64)
            T = len(z)
            emb = img[vids[k]]
            r = z / 100.0
            print(f"\n[{bench}] {k}  video {vids[k]}  {T} s  (relevance cosine "
                  f"{r.min():.3f}-{r.max():.3f}; {len(f2c.candidates(r))} basin peaks)")
            t0 = time.perf_counter()
            clips, na = f2c.select(r, emb, 64)
            secs_per_q.append(time.perf_counter() - t0)
            agg["ours"].append(report("ours (K 64, 224 tokens a full frame)", clips, na,
                                      BUDGET / 64, BUDGET, SIZES, not args.quiet))
            clips, na = f2c.select(r, emb, 32)
            agg["k32"].append(report("k32 (K 32, 448 tokens a full frame)", clips, na,
                                     BUDGET / 32, BUDGET, SIZES, False))
            for (h, w) in SIZES:                         # released: native full frames, K = 32
                full = f2c.native_tokens(h, w, LMMS_MIN, LMMS_MAX)
                agg[f"released {h}p"].append(report(
                    f"released (K 32, native {h}x{w} = {full:.0f} tokens a full frame)", clips, na,
                    full, 32 * full, [(h, w)], False))
            zm = maxopt[k].astype(np.float64)
            times = curve_times(zm, float(T), 64, "kelly", 3.58, 0.0)
            hclips, hna = f2c.select(zm / 100.0, emb, 64, placed=times)
            agg["host"].append(report("host (CaliFrame kelly tau 3.58 eps 0, maxopt)", hclips,
                                      hna, BUDGET / 64, BUDGET, SIZES, not args.quiet))
            for name, r1 in (("R(1)=0", 0.0), ("R(1)=1", 1.0)):
                c0, _ = f2c.select(r, emb, 64, r_single=r1)
                cf[name].update(c.length for c in c0)

    print("\n===== summary over all questions =====")
    for name, rows in agg.items():
        lens = sum((x["lens"] for x in rows), collections.Counter())
        tot = sum(lens.values())
        print(f"{name:<14} {len(rows)} q, all within budget: {all(x['ok'] for x in rows)}; "
              f"anchors {np.mean([x['anchors'] for x in rows]):.1f}, frames "
              f"{np.mean([x['frames'] for x in rows]):.1f}, shared seconds "
              f"{np.mean([x['shared'] for x in rows]):.1f}; tokens: shares "
              f"{np.mean([x['total'] / x['budget'] for x in rows]):.1%} of budget, grid nearest "
              f"{np.mean([x['near'] / x['budget'] for x in rows]):.1%}, floor "
              f"{np.mean([x['floor'] / x['budget'] for x in rows]):.1%}; l share "
              + " ".join(f"{ln}:{c / tot:.0%}" for ln, c in sorted(lens.items())))
    for name, c in cf.items():
        tot = sum(c.values())
        print(f"ours with {name}: l share "
              + " ".join(f"{ln}:{v / tot:.0%}" for ln, v in sorted(c.items())))
    print(f"selection time per question (ours): median {np.median(secs_per_q) * 1e3:.0f} ms, "
          f"max {max(secs_per_q) * 1e3:.0f} ms")


if __name__ == "__main__":
    main()
