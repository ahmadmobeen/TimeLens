#!/usr/bin/env python3
"""Study X6: visual tokens per question for dense uniform sampling (up to 768 frames under
qwen_vl_utils' default video budget) and for M2 (64 frames under the harness's shared budget).

No decoding: each video's native frame size comes from its container header (PyAV), and the size
each frame is resized to follows qwen_vl_utils.fetch_video exactly: the per-frame pixel cap is
max(min(768 * 28^2, total_pixels / n * 2), 1.05 * min_pixels), then smart_resize to multiples of
28. Qwen2.5-VL merges two frames per temporal patch and 2 x 2 patches of 14 px per token, so a
question's visual tokens are n / 2 * (h / 28) * (w / 28). The frame count n of each X6 question is
the one its row records (videos shorter than 768 frames are read whole).

    .venv/bin/python nsp5_x6_tokens.py          # from research/TimeLens
"""
import json
import os
import sys
from statistics import mean, median

import av

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv_saved, sys.argv = sys.argv, sys.argv[:1]      # the eval module parses no args at import
import eval_lvbench_foveate as ev                          # noqa: E402
import bench                                               # noqa: E402
from qwen_vl_utils import vision_process as vp             # noqa: E402
sys.argv = sys.argv_saved

FACTOR = 28
FRAME_CAP = vp.VIDEO_MAX_TOKEN_NUM * FACTOR * FACTOR
LVB = "/home/mobeen/codes/active_repos/NeedleScan-vtg-landscape/research/TimeLens/data/lvbench"
# arm -> (min_pixels, total_pixels) as the reference read receives them
SETTINGS = {"x6_uniform_dense": (3136, vp.MODEL_SEQ_LEN * FACTOR * FACTOR * 0.9),
            "q8_kelly_eps0": (64 * FACTOR * FACTOR, 14336 * FACTOR * FACTOR)}
FILES = {"lvbench": "logs/nsp5/{arm}.jsonl", "videomme": "logs/nsp5/x1_videomme_{arm}.jsonl"}


def video_paths(b: str) -> dict:
    """Row key -> video path. Rows are keyed as the harness keys them: 'vid>>>uid' on LVBench, the
    question's own key (e.g. 'videomme:0') elsewhere."""
    if b == "lvbench":
        items, _ = ev.load_questions(f"{LVB}/data/train-00000-of-00001.parquet",
                                     f"{LVB}/video_info.meta.jsonl", f"{LVB}/videos")
        return {f"{it['key']}>>>{it['uid']}": it["video_path"] for it in items}
    return {q.key: q.video_path for q in bench.load(b)}


def native_hw(path: str) -> tuple:
    with av.open(path) as container:
        ctx = container.streams.video[0].codec_context
        return ctx.height, ctx.width


def tokens(hw: tuple, n: int, min_pixels: int, total_pixels: float) -> int:
    cap = max(min(FRAME_CAP, total_pixels / n * vp.FRAME_FACTOR), int(min_pixels * 1.05))
    h, w = vp.smart_resize(hw[0], hw[1], factor=FACTOR, min_pixels=min_pixels, max_pixels=cap)
    return n // vp.FRAME_FACTOR * (h // FACTOR) * (w // FACTOR)


OUT = "logs/nsp5/x6_tokens.json"     # the summary the paper's check_numbers.py reads


def main() -> None:
    summary = {}
    for b in FILES:
        paths, sizes = video_paths(b), {}
        print(f"== {b}")
        for arm, (min_px, total_px) in SETTINGS.items():
            counts, frames = [], []
            for line in open(FILES[b].format(arm=arm), encoding="utf-8"):
                if not line.strip():
                    continue
                r = json.loads(line)
                path = paths[r["key"]]
                if path not in sizes:
                    sizes[path] = native_hw(path)
                counts.append(tokens(sizes[path], int(r["n_frames"]), min_px, total_px))
                frames.append(int(r["n_frames"]))
            summary.setdefault(b, {})[arm] = {
                "questions": len(counts), "frames_mean": mean(frames), "frames_min": min(frames),
                "tokens_mean": mean(counts), "tokens_median": median(counts), "tokens_max": max(counts)}
            print(f"  {arm:18s} questions {len(counts):5d}  frames mean {mean(frames):6.1f} "
                  f"(min {min(frames)})  visual tokens mean {mean(counts):9.0f}  median "
                  f"{median(counts):9.0f}  max {max(counts):7d}")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(summary, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
