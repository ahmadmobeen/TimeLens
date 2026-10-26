"""T1 stage A: decode every source video ONCE into a 1 fps JPEG cache.

The T1 amendment (METHODS-LOG, 2026-10-25) trains on per-second features of three
corpora -- LVBench (the hour-long target), Charades-STA and ActivityNet Captions (the
synthetic-timeline sources). Decoding hour-long video is the expensive step, so it is
done once here and every encoder (SigLIP now, a fine-tuned tower in stage 1b) reads
the cache instead of the video.

Frame k of a video is the first decoded frame at or after second k, so feature index
and timestamp agree to within one source frame. Every frame is resized to a 360 px
short side (long side capped at 640) and stored as JPEG.

Per video, two files under data/t1/frames/<src>/:
  <vid>.bin    the JPEG bytes, concatenated
  <vid>.json   {"off": [...T+1 byte offsets], "t": [...T seconds], "size": [W, H],
                "duration": container seconds, "path": source}
The .json is written last, by rename, so a present .json means a complete video.

Run:
  .venv/bin/python -u t1_decode.py --src lvbench charades anet --workers 48
  .venv/bin/python -u t1_decode.py --src videomme lvb mlvu cgbench --workers 32

The evaluation benchmarks (Video-MME, LongVideoBench, MLVU, CG-Bench) come through
bench.py; their sources are read from /gpfs by exact path, never by listing.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import time
from multiprocessing import Pool

import av

LVB = ("/home/mobeen/codes/active_repos/NeedleScan-vtg-landscape/research/TimeLens/"
       "data/lvbench")
CHARADES = "data/charades_full/charades_train_natural.json"
ANET = "data/charades_full/anet_train_natural.json"
OUT = "data/t1/frames"
SHORT, LONG_CAP, QUALITY = 360, 640, 85

EVAL_BENCHES = ("videomme", "lvb", "mlvu", "cgbench")   # loaded through bench.py

# Abort when this many videos have failed and none has succeeded (broken setup).
FAIL_FAST = 20


def sources(names: list[str]) -> list[tuple[str, str, str]]:
    """(src, vid, path) for every unique video of the requested corpora."""
    jobs = []
    if "lvbench" in names:
        import pandas as pd
        df = pd.read_parquet(f"{LVB}/data/train-00000-of-00001.parquet")
        for key, vp in sorted(set(zip(df.key, df.video_path))):
            jobs.append(("lvbench", key, f"{LVB}/videos/{vp}"))
    for src, ann in (("charades", CHARADES), ("anet", ANET)):
        if src in names:
            with open(ann, encoding="utf-8") as fh:
                paths = sorted({r["video_path"] for r in json.load(fh)})
            jobs += [(src, os.path.splitext(os.path.basename(p))[0], p) for p in paths]
    for src in names:                     # evaluation benchmarks, via the shared loaders
        if src in EVAL_BENCHES:
            import bench
            first = {}
            for it in bench.load(src):
                first.setdefault(it.vid, it.video_path)
            jobs += [(src, v, p) for v, p in sorted(first.items())]
    return jobs


def target_size(w: int, h: int) -> tuple[int, int]:
    s = SHORT / min(w, h)
    if max(w, h) * s > LONG_CAP:
        s = LONG_CAP / max(w, h)
    return int(round(w * s / 2)) * 2, int(round(h * s / 2)) * 2


def decode(job: tuple[str, str, str]) -> tuple[str, str, int, str | None]:
    """Decode one video to the cache. Returns (src, vid, n_frames, error or None)."""
    src, vid, path = job
    d = os.path.join(OUT, src)
    if os.path.exists(os.path.join(d, vid + ".json")):
        return src, vid, -1, None                     # already done
    try:
        off, ts, size = [0], [], None
        tmp = os.path.join(d, vid + ".bin.tmp")
        with av.open(path) as c, open(tmp, "wb") as fh:
            s = c.streams.video[0]
            s.thread_type = "AUTO"
            s.codec_context.thread_count = 4
            tb, start = s.time_base, s.start_time or 0
            dur = float(c.duration / 1e6) if c.duration else float(s.duration * tb)
            nxt = 0.0
            for f in c.decode(s):
                if f.pts is None:
                    continue
                t = float((f.pts - start) * tb)
                if t + 1e-6 < nxt:
                    continue
                if size is None:
                    size = target_size(f.width, f.height)
                buf = io.BytesIO()
                f.to_image(width=size[0], height=size[1]).save(buf, format="JPEG",
                                                               quality=QUALITY)
                fh.write(buf.getbuffer())
                off.append(off[-1] + buf.tell())
                ts.append(round(t, 3))
                nxt = float(int(t)) + 1.0
        if not ts:
            raise RuntimeError("no frames decoded")
        os.replace(tmp, os.path.join(d, vid + ".bin"))
        meta = {"off": off, "t": ts, "size": list(size), "duration": dur, "path": path}
        with open(os.path.join(d, vid + ".json.tmp"), "w", encoding="utf-8") as fh:
            json.dump(meta, fh)
        os.replace(os.path.join(d, vid + ".json.tmp"), os.path.join(d, vid + ".json"))
        return src, vid, len(ts), None
    except Exception as ex:                           # reported and counted, never hidden
        return src, vid, 0, repr(ex)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", nargs="+",
                    choices=["lvbench", "charades", "anet", *EVAL_BENCHES], required=True)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()

    jobs = sources(args.src)
    for src in args.src:
        os.makedirs(os.path.join(OUT, src), exist_ok=True)
    # Longest first, so the hour-long videos do not straggle at the end.
    jobs.sort(key=lambda j: j[0] != "lvbench")
    print(f"{len(jobs)} videos: " + ", ".join(
        f"{s}={sum(j[0] == s for j in jobs)}" for s in args.src), flush=True)

    t0, done, cached, frames, failed = time.time(), 0, 0, 0, []
    with Pool(args.workers) as pool:
        for src, vid, n, err in pool.imap_unordered(decode, jobs, chunksize=1):
            if err:
                failed.append({"src": src, "vid": vid, "error": err})
                print(f"  [fail {len(failed)}] {src}/{vid} | {err}", flush=True)
                if done == 0 and len(failed) >= FAIL_FAST:
                    pool.terminate()
                    raise RuntimeError(f"{FAIL_FAST} failures, nothing decoded: {err}")
                continue
            done += 1
            cached += n < 0
            frames += max(n, 0)
            if done % 250 == 0 or done == len(jobs) - len(failed):
                el = time.time() - t0
                print(f"  [{done}/{len(jobs)}] {frames} frames new, {cached} cached, "
                      f"{len(failed)} failed, {el / 60:.1f} min", flush=True)

    with open(os.path.join(OUT, "failures.json"), "w", encoding="utf-8") as fh:
        json.dump(failed, fh, indent=1)
    print(f"\nDONE: {done} videos ok ({cached} were cached), {frames} new frames, "
          f"{len(failed)} failed -> {OUT}/failures.json", flush=True)


if __name__ == "__main__":
    main()
