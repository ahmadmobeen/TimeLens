"""Amendment (METHODS-LOG 2026-10-30): seed the corrected reference arms of Studies A and B.

times_to_indices restores the frame budget from the start of the video when two placed times
fall on one source frame (crowding) or a placement returns fewer times than frames (top-k on a
video shorter than the budget). The amendment reruns M (q8_kelly) and top-k (q8opt_topk) with
--refill nearest and --topk_short uniform, which change frames only where that refill happened.
This script finds those questions without answering anything: it reads each video's frame count
and rate as the harness does (decord: len(vr), vr.get_avg_fps()), re-places every logged question
with curve_times, and keeps a logged row only where the old rule refilled nothing. The kept rows
seed <arm>_nr.jsonl; the harness, resuming from that file with the new flags, answers the rest.

    .venv/bin/python nsp5_refill.py
"""
import json
import os
from concurrent.futures import ProcessPoolExecutor

import numpy as np

import bench
from eval_lvbench_foveate import curve_times

L, P = "logs/nsp5", "data/t1/priors/qwen3vl_emb_8b"
BENCH = ("lvbench", "videomme", "lvb", "mlvu", "cgbench")
META = f"{L}/video_meta.json"
ARMS = {"q8_kelly": ("kelly", "q8_kelly"), "q8opt_topk": ("topk", "ge1_opt_topk")}  # alloc, LVBench file


def video_meta(path):
    from decord import VideoReader
    vr = VideoReader(path, num_threads=1)
    return path, len(vr), float(vr.get_avg_fps())


def scan(paths):
    meta = json.load(open(META)) if os.path.exists(META) else {}
    todo = sorted(set(paths) - set(meta))
    with ProcessPoolExecutor(16) as ex:
        for path, total, fps in ex.map(video_meta, todo, chunksize=4):
            meta[path] = [total, fps]
    with open(META + ".tmp", "w", encoding="utf-8") as fh:
        json.dump(meta, fh)
    os.replace(META + ".tmp", META)
    return meta


def main():
    items = {b: bench.load(b) for b in BENCH}
    meta = scan([it.video_path for b in BENCH for it in items[b]])
    for b in BENCH:
        curves = np.load(f"{P}/{'' if b == 'lvbench' else b + '_'}curves_maxopt.npz")
        cut = len("lvbench:") if b == "lvbench" else 0
        path_of = {it.key[cut:]: (it.key, it.video_path) for it in items[b]}
        for arm, (alloc, lvb_file) in ARMS.items():
            src = f"{L}/{lvb_file}.jsonl" if b == "lvbench" else f"{L}/x1_{b}_{arm}.jsonl"
            dst = f"{L}/{arm}_nr.jsonl" if b == "lvbench" else f"{L}/x1_{b}_{arm}_nr.jsonl"
            if os.path.exists(dst):
                raise SystemExit(f"{dst} exists; not overwriting a seed")
            kept, rerun, off = [], 0, 0
            for line in open(src, encoding="utf-8"):
                r = json.loads(line)
                ckey, path = path_of[r["key"]]
                total, fps = meta[path]
                n, dur = r["n_frames"], r["duration_s"]
                if abs(total / fps - dur) > 1e-6:       # not the video the harness read
                    off += 1
                    rerun += 1
                    continue
                times = curve_times(curves[ckey], dur, n, alloc, 3.58, 0.5)
                if len(times) == n and len({min(total - 1, max(0, round(t * fps))) for t in times}) == n:
                    kept.append(line if line.endswith("\n") else line + "\n")
                else:
                    rerun += 1
            with open(dst, "w", encoding="utf-8") as fh:
                fh.writelines(kept)
            print(f"{b:<9} {arm:<11} kept {len(kept):5d}  to rerun {rerun:4d}"
                  f"{f'  ({off} with a different duration)' if off else ''}", flush=True)


if __name__ == "__main__":
    main()
