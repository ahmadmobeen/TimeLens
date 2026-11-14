#!/usr/bin/env python3
"""Fill the reference-read cache (data/t1/refmeta) without decoding whole videos into memory.

For videos where decord fails, qwen_vl_utils falls back to torchvision.io.read_video, which decodes
the WHOLE video into memory (100-590 GiB for CG-Bench). eval_lvbench_foveate.py keeps what the harness
uses from that read (the shape and dtype of the sampled, resized tensor, the frame rate and the frame
count) in data/t1/refmeta, but only once a read completes; the runner's memory guard often stops the
eval first, so the read repeats on every visit. This script computes the same values with PyAV,
keeping only frame timestamps, and writes them with the harness's own key and writer.

  python3 nsp5_refmeta_warm.py check [--mode packets|decode] [--workers N]
      recompute every existing entry and compare (all must match before warming)
  python3 nsp5_refmeta_warm.py warm  [--mode packets|decode] [--workers N]
      write the missing entries: videos whose decord read failed in any eval log, for every frame
      setting of the arms still planned on any runner

The frame count follows torchvision's _read_from_stream for start 0 and no end: frames keyed by pts
(a repeated pts counts once), kept when pts >= 0. 'decode' decodes every frame as torchvision does;
'packets' counts demuxed packets instead (much faster) and is used only if check shows it agrees.
Run with research/TimeLens/.venv/bin/python from research/TimeLens.
"""
import argparse
import glob
import json
import os
import re
import sys
from multiprocessing import Pool

import av
import torch

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.argv_saved, sys.argv = sys.argv, sys.argv[:1]      # the eval module parses no args at import
import eval_lvbench_foveate as ev                          # noqa: E402  (its cache key and writer)
from qwen_vl_utils import vision_process as vp             # noqa: E402
sys.argv = sys.argv_saved

DEFAULTS = {"n_frames": 64, "min_pixels": 64 * 28 * 28, "total_pixels": 14336 * 28 * 28, "max_pixels": 0}
BENCHES = ("cgbench", "cgbench_fresh", "mlvu", "lvb")       # the benchmarks with decord failures


def light_read(path: str, mode: str) -> tuple:
    """(total_frames, video_fps, height, width) as torchvision.io.read_video would report them."""
    with av.open(path, metadata_errors="ignore") as c:
        s = c.streams.video[0]
        fps = float(s.average_rate) if s.average_rate is not None else None
        c.seek(0, any_frame=False, backward=True, stream=s)
        pts, h, w = set(), None, None
        try:
            if mode == "packets":
                h, w = s.codec_context.height, s.codec_context.width
                for p in c.demux(s):
                    if p.pts is not None and p.size > 0:
                        pts.add(p.pts)
            else:
                for f in c.decode(video=0):
                    pts.add(f.pts)
                    if h is None:
                        h, w = f.height, f.width
        except av.error.FFmpegError:
            pass
    return len([p for p in pts if p is not None and p >= 0]), fps, h, w


def entry_for(path: str, settings: dict, read: tuple) -> dict:
    """What fetch_video returns for the torchvision read, reduced to what the cache keeps."""
    total, fps, h, w = read
    ele = {"type": "video", "video": path, **settings}
    nframes = vp.smart_nframes(ele, total_frames=total, video_fps=fps)
    factor = 14 * vp.SPATIAL_MERGE_SIZE
    min_pixels = ele.get("min_pixels", vp.VIDEO_MIN_TOKEN_NUM * factor * factor)
    total_pixels = ele.get("total_pixels", vp.MODEL_SEQ_LEN * factor * factor * 0.9)
    max_pixels = max(min(vp.VIDEO_MAX_TOKEN_NUM * factor * factor, total_pixels / nframes * vp.FRAME_FACTOR),
                     int(min_pixels * 1.05))
    max_pixels = min(ele.get("max_pixels", max_pixels), max_pixels)
    rh, rw = vp.smart_resize(h, w, factor=factor, min_pixels=min_pixels, max_pixels=max_pixels)
    return {"shape": [nframes, 3, rh, rw], "dtype": "torch.float32", "fps": fps, "total_num_frames": total}


def ref_video(path: str, s: dict) -> dict:
    """The eval's reference call for these settings (same keys, same order of construction)."""
    rv = {"type": "video", "video": path, "nframes": s["n_frames"], "min_pixels": s["min_pixels"]}
    if s["total_pixels"]:
        rv["total_pixels"] = s["total_pixels"]
    if s["max_pixels"]:
        rv["max_pixels"] = s["max_pixels"]
    return rv


def check_one(args: tuple) -> tuple:
    f, mode = args
    e = json.load(open(f))
    try:
        got = entry_for(e["video"], e["settings"] | {}, light_read(e["video"], mode))
    except Exception as ex:                                  # report, never hide
        return f, False, repr(ex)
    want = {k: e[k] for k in ("shape", "dtype", "fps", "total_num_frames")}
    return f, got == want, f"want {want} got {got}"


def warm_one(args: tuple) -> tuple:
    path, settings_list, mode = args
    todo = [s for s in settings_list if not os.path.exists(ev._ref_cache_path(ref_video(path, s)))]
    if not todo:
        return path, 0, ""
    try:
        read = light_read(path, mode)
    except Exception as ex:
        return path, 0, repr(ex)
    for s in todo:
        rv = ref_video(path, s)
        try:
            e = entry_for(path, {k: v for k, v in rv.items() if k not in ("type", "video")}, read)
        except ValueError:          # an impossible setting for this video: the real read raises too
            continue
        meta = {"fps": e["fps"], "total_num_frames": e["total_num_frames"], "video_backend": "torchvision"}
        ev.ref_cache_put(rv, (torch.empty(e["shape"], dtype=torch.float32, device="meta"), meta))
    return path, len(todo), ""


SCAN = os.path.join("logs", "nsp5", "cal", "decord_scan.json")
SCAN_BENCHES = ("lvbench", "videomme", "lvb", "mlvu", "cgbench", "cgbench_cal")   # cgbench_fresh = cgbench's videos


def decord_fails(path: str) -> tuple:
    """qwen_vl_utils' own decord read (16 frames, always including the last frame, as every
    reference read does); a failure here is what sends the eval to the whole-video fallback."""
    try:
        vp.VIDEO_READER_BACKENDS["decord"]({"type": "video", "video": path, "nframes": 16})
        return path, False, ""
    except Exception as ex:                                    # report the reason with the video
        return path, True, repr(ex)[:160]


def failing_videos() -> set:
    """Every video whose decord read failed in any eval log, plus every failure of the last scan."""
    out = set()
    pat = re.compile(r"video_reader_backend decord error.*?\[(/[^\]]+\.(?:mp4|mkv|webm|avi|mov))\]")
    for f in glob.glob("logs/nsp5/*.log"):
        out.update(pat.findall(open(f, errors="ignore").read()))
    if os.path.exists(SCAN):
        out.update(v for v, r in json.load(open(SCAN)).items() if r["fails"])
    return out


def decord_one(args: tuple) -> tuple:
    """Cache entries for a video decord reads, from decord's own frame count, frame rate and frame
    size: the values qwen_vl_utils' decord reference read returns (its tensor is resized the same way).
    Decord's failures are load-dependent, and on the 840 videos where the fallback ran, decord's count
    and rate equal torchvision's on 839 (one frame apart on the last), so caching every video removes
    the fallback without changing any reference."""
    import decord
    path, settings_list = args
    todo = [s for s in settings_list if not os.path.exists(ev._ref_cache_path(ref_video(path, s)))]
    if not todo:
        return path, 0, ""
    try:
        vr = decord.VideoReader(path)                      # builds the index; decodes no frame
        with av.open(path, metadata_errors="ignore") as c:  # frame size from the stream header
            cc = c.streams.video[0].codec_context
            h, w = cc.height, cc.width
        read = (len(vr), float(vr.get_avg_fps()), int(h), int(w))
    except Exception as ex:
        return path, 0, repr(ex)[:160]
    for s in todo:
        rv = ref_video(path, s)
        try:
            e = entry_for(path, {k: v for k, v in rv.items() if k not in ("type", "video")}, read)
        except ValueError:          # e.g. 256 frames from a 240-frame clip: the real read raises too
            continue
        entry = {"video": path, "settings": {k: v for k, v in rv.items() if k != "video"},
                 "shape": e["shape"], "dtype": e["dtype"], "fps": e["fps"],
                 "total_num_frames": e["total_num_frames"], "video_backend": "decord"}
        p = ev._ref_cache_path(rv)
        tmp = f"{p}.{os.getpid()}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(entry, fh)
        os.replace(tmp, p)
    return path, len(todo), ""


def warm_all(workers: int) -> None:
    """Every video of the benchmarks, every planned setting, from decord's metadata (fallback videos
    keep their torchvision entries, which are written first)."""
    import bench
    settings = planned_settings()
    videos = sorted({q.video_path for b in SCAN_BENCHES for q in bench.load(b)})
    written, errors = 0, []
    with Pool(workers) as pool:
        for path, n, err in pool.imap_unordered(decord_one, [(v, settings) for v in videos]):
            written += n
            if err:
                errors.append((path, err))
    print(f"{len(videos)} videos x {len(settings)} settings: wrote {written} entries; {len(errors)} errors", flush=True)
    for path, err in errors[:10]:
        print("  ERROR", path, err, flush=True)


def scan(workers: int) -> None:
    """Test every video of the benchmarks once, so the cache can be filled before an eval meets one."""
    import bench
    videos = sorted({q.video_path for b in SCAN_BENCHES for q in bench.load(b)})
    done = json.load(open(SCAN)) if os.path.exists(SCAN) else {}
    todo = [v for v in videos if v not in done]
    print(f"{len(videos)} videos, {len(todo)} not scanned yet", flush=True)
    with Pool(workers) as pool:
        for i, (path, fails, why) in enumerate(pool.imap_unordered(decord_fails, todo), 1):
            done[path] = {"fails": fails, "why": why}
            if i % 200 == 0:
                json.dump(done, open(SCAN, "w"))
                print(f"  {i}/{len(todo)} scanned, {sum(r['fails'] for r in done.values())} fail", flush=True)
    json.dump(done, open(SCAN, "w"))
    print(f"scan done: {sum(r['fails'] for r in done.values())} of {len(done)} videos fail decord", flush=True)


def planned_settings() -> list:
    """Distinct frame settings of the arms on the benchmarks with decord failures, on every runner."""
    seen = []
    for p in glob.glob("logs/nsp5/runner*/plan.json"):
        for a in json.load(open(p))["arms"]:
            if a.get("bench") not in BENCHES:
                continue
            s = dict(DEFAULTS)
            for k in s:
                m = re.search(rf"--{k} (\d+)", a["args"])
                if m:
                    s[k] = int(m.group(1))
            if s not in seen:
                seen.append(s)
    return seen


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=["check", "warm", "scan", "all"])
    ap.add_argument("--mode", choices=["packets", "decode"], default="decode")
    ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args()
    if a.what == "check":
        files = sorted(glob.glob("data/t1/refmeta/*.json"))
        with Pool(a.workers) as pool:
            res = pool.map(check_one, [(f, a.mode) for f in files], chunksize=1)
        bad = [r for r in res if not r[1]]
        print(f"{a.mode}: {len(res) - len(bad)} of {len(res)} entries reproduced exactly")
        for f, _, msg in bad[:10]:
            print("  MISMATCH", f, msg)
        sys.exit(1 if bad else 0)
    if a.what == "scan":
        scan(a.workers)
        return
    if a.what == "all":
        warm_all(a.workers)
        return
    settings = planned_settings()
    videos = sorted(failing_videos())
    print(f"{len(videos)} videos with a decord failure; settings: {settings}", flush=True)
    with Pool(a.workers) as pool:
        written = 0
        for path, n, err in pool.imap_unordered(warm_one, [(v, settings, a.mode) for v in videos]):
            written += n
            if err:
                print("  ERROR", path, err, flush=True)
    print(f"wrote {written} entries")


if __name__ == "__main__":
    main()
