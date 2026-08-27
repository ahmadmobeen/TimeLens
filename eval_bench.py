"""Generate TimeLens-7B grounding predictions on any TimeLens-Bench-format split.

Loads a ``{key: {duration, spans, queries}}`` JSON (e.g. qvhighlights-timelens.json) where the
video for ``key`` is ``{video_dir}/{key}.mp4`` (specific known paths — never listed/scanned), runs
greedy generation per (key, query), parses spans, and appends to a JSONL in the eval's ``>>>``-key
format so it scores through the existing ``read_timelens_eval`` + ``length_profile`` path (the GT
travels inside the key, so no benchmark-specific adapter is needed).

Run (TimeLens venv, from research/TimeLens, PYTHONPATH=.):
  env -u VIRTUAL_ENV CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. ./.venv/bin/python eval_bench.py \\
    --json data/TimeLens-Bench/qvhighlights-timelens.json \\
    --video_dir /gpfs/public/datasets/qvhighlights/videos \\
    --out logs/qvh_timelens/preds.shard0of2.jsonl --shard 0 --num_shards 2
"""
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import torch
from transformers import AutoModelForImageTextToText, AutoProcessor

from evaluation.utils import GroundingDataset
from timelens.utils import extract_time

MODEL = "TencentARC/TimeLens-7B"


def load_annos(json_path: str, video_dir: str) -> list[dict]:
    d = json.loads(open(json_path, encoding="utf-8").read())
    annos = []
    for key, rec in d.items():
        # Exact known paths, not discovered. .mp4 first so existing datasets are
        # unaffected; the fallbacks exist because 13 of native ActivityNet's canonical
        # 500 pairs are .mkv, and hardcoding .mp4 dropped them to 475 -- a silent
        # 5% population change of exactly the kind issue #86 was about.
        vp = os.path.join(video_dir, key + ".mp4")
        if not os.path.exists(vp):
            for ext in (".mkv", ".webm", ".avi"):
                alt = os.path.join(video_dir, key + ext)
                if os.path.exists(alt):
                    vp = alt
                    break
        spans, queries = rec["spans"], rec["queries"]
        for i in range(min(len(spans), len(queries))):
            annos.append({
                "video_path": vp, "query": queries[i],
                "span": [list(spans[i])], "duration": float(rec["duration"]),
            })
    return annos


def _anno_key(anno: dict) -> str:
    """The prediction-file key for an annotation. Must match what the writer emits.

    Uses the video's ACTUAL basename, extension included, rather than assuming .mp4.
    load_annos falls back to .mkv/.webm/.avi when .mp4 is absent, and hardcoding .mp4
    here made the key name a file that was never opened -- which silently prevented 25
    of native ActivityNet's 500 pairs from joining against the 7B reference, whose keys
    carry the real .mkv. Datasets whose videos are .mp4 are unaffected: same key as
    before. Flagged by an external review of this change.
    """
    span = anno["span"][0] if isinstance(anno["span"][0], (list, tuple)) else anno["span"]
    return f"{os.path.basename(anno['video_path'])}>>>{anno['query']}>>>{list(span)}"


def _identity(x):
    """Pass a sample through a DataLoader untouched. Module-level, not a lambda, so it
    is picklable under spawn/forkserver as well as fork."""
    return x


class _IndexedSlice(torch.utils.data.Dataset):
    """Yield (original index, prepared item) for a chosen subset of a dataset.

    Module-level for the same reason as _identity: a class defined inside a function is
    not picklable, so a spawn/forkserver DataLoader would fail on it. Linux defaults to
    fork, which is why that went unnoticed until review.

    A decode failure returns (i, None) instead of raising. Under workers > 0 an uncaught
    exception in a worker tears down the whole loader, so per-item containment is not a
    nicety here.
    """

    def __init__(self, ds, todo: list[int]) -> None:
        self.ds = ds
        self.todo = todo

    def __len__(self) -> int:
        return len(self.todo)

    def __getitem__(self, j: int):
        i = self.todo[j]
        try:
            return i, self.ds[i]
        except Exception as ex:  # noqa: BLE001 - one bad clip must not kill the run
            print(f"  [decode fail] idx={i} | {ex!r}", flush=True)
            return i, None


def _iter_prepared(ds, todo: list[int], workers: int):
    """Yield (index, prepared item) for the given indices.

    workers <= 0 decodes inline, which is byte-for-byte the previous behaviour.
    workers > 0 decodes in subprocesses so decode overlaps GPU work; order is then
    arbitrary, which is harmless because every record carries its own key.

    The caller must pass only NOT-yet-done indices. The inline path could afford to
    skip cheaply after indexing, but a prefetching loader decodes ahead of the
    consumer, so anything left in the list gets decoded whether or not it is wanted.
    """
    # A decode failure yields (i, None) rather than raising. The pre-DataLoader loop
    # called ds[i] INSIDE its try, so one unreadable clip was skipped and the shard
    # continued; decoding in the iterator moved that call outside the try and would
    # have made a single bad clip kill the run. Under workers > 0 it is worse: an
    # uncaught exception in a worker tears down the whole loader.
    slice_ds = _IndexedSlice(ds, todo)
    if workers <= 0:
        for j in range(len(slice_ds)):
            yield slice_ds[j]
        return

    # batch_size=None disables automatic batching; _identity keeps the sample exactly
    # as __getitem__ built it rather than letting default_convert reshape it.
    loader = torch.utils.data.DataLoader(slice_ds, batch_size=None, num_workers=workers,
                                         prefetch_factor=2, collate_fn=_identity)
    yield from loader


def _moment_len(anno: dict) -> float:
    """GT moment length in seconds. load_annos stores span as [[s, e]], but the
    zoom/foveate paths hand around a bare [s, e], so accept both shapes."""
    sp = anno["span"]
    sp = sp[0] if isinstance(sp[0], (list, tuple)) else sp
    return float(sp[1]) - float(sp[0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--video_dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--min_tokens", type=int, default=64)
    ap.add_argument("--total_tokens", type=int, default=14336)
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--max_moment_s", type=float, default=0.0,
                    help="keep only clips whose GT moment is SHORTER than this many seconds "
                         "(0 = every clip, the default). Same flag name and meaning as "
                         "eval_zoom.py's GT-length filter. This selects WHICH clips run; it "
                         "changes no per-clip setting, so the survivors stay comparable to a "
                         "full-split run of another model -- read the same stratum out of it.")
    ap.add_argument("--model", default=MODEL,
                    help="HF model id; same Qwen2.5-VL architecture (e.g. Qwen/Qwen2.5-VL-7B-Instruct for the zero-shot backbone control)")
    ap.add_argument("--attn", default="sdpa",
                    choices=["sdpa", "flash_attention_2", "eager"],
                    help="attention kernel. Default sdpa, which is what produced the 7B "
                         "splits and the 8B Charades baseline -- keep it for anything that "
                         "must stay comparable to those. flash_attention_2 computes the "
                         "same exact attention with a different reduction order, so it is "
                         "not guaranteed bitwise identical; verify with "
                         "verify_preds_identical.py before trusting a switched run.")
    ap.add_argument("--patch_embed", default="matmul", choices=["matmul", "conv3d"],
                    help="Qwen3-VL (8B) vision patch embedding. 'matmul' expresses the "
                         "kernel==stride Conv3d as the matrix multiply it already is, "
                         "which is ~1200x faster end-to-end on long video because the "
                         "shipped bf16 Conv3d hits a catastrophic kernel on this shape "
                         "(see fast_patch_embed.py). It differs from that Conv3d by ~1 ulp "
                         "of bf16. Use 'conv3d' to reproduce 8B runs made before "
                         "2026-08-14, or to re-check the numerics. No effect on 7B.")
    ap.add_argument("--decode_workers", type=int, default=0,
                    help="decode video in N subprocesses via a DataLoader, overlapping decode "
                         "with GPU work. 0 (default) decodes inline, exactly as before. "
                         "Measured on a 150 s clip: decode 1.77 s vs generate 1.52 s, so ~2 "
                         "workers saturate one GPU and more buys nothing -- the ceiling is "
                         "generate. Needs enough /dev/shm: the loader passes tensors through "
                         "shared memory, and the container default of 64 MB is far too small.")
    args = ap.parse_args()
    args.model_path = args.model
    args.split = "test"

    if args.patch_embed == "matmul":
        import fast_patch_embed
        fast_patch_embed.patch()

    annos = [a for a in load_annos(args.json, args.video_dir) if os.path.exists(a["video_path"])]
    # Filter BEFORE sharding so the shards partition the stratum instead of each
    # holding a sparse slice of the full split. Consequence worth stating: a given
    # --shard owns different clips under different --max_moment_s, so a filtered
    # run must never resume against an unfiltered run's --out. The launcher keeps
    # the two in separate directories for exactly that reason.
    if args.max_moment_s:
        annos = [a for a in annos if _moment_len(a) < args.max_moment_s]
    annos = annos[args.shard :: args.num_shards]
    if args.limit:
        annos = annos[: args.limit]
    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    done: set[str] = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    done.add(next(iter(json.loads(line))))
                except Exception:
                    pass
    filt = f", GT<{args.max_moment_s:g}s" if args.max_moment_s else ""
    print(f"[{os.path.basename(args.json)}] shard {args.shard}/{args.num_shards}: {len(annos)} clips{filt}, "
          f"attn={args.attn}, patch_embed={args.patch_embed}, resume {len(done)} -> {args.out}")

    model = AutoModelForImageTextToText.from_pretrained(
        args.model, dtype=torch.bfloat16, attn_implementation=args.attn, device_map="auto"
    ).eval()
    processor = AutoProcessor.from_pretrained(
        args.model, padding_side="left", do_resize=False, trust_remote_code=True
    )
    ds = GroundingDataset(annos, processor, args)

    # Resolve resume BEFORE iterating so decode workers never prefetch a clip we would
    # discard. Same set of survivors as the old in-loop `if key in done: continue`.
    todo = [i for i in range(len(ds)) if _anno_key(ds.annos[i]) not in done]
    if len(todo) != len(ds):
        print(f"  resume: {len(ds) - len(todo)} already done, {len(todo)} to run", flush=True)

    n = skipped = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for i, item in _iter_prepared(ds, todo, args.decode_workers):
            a = ds.annos[i]
            key = _anno_key(a)
            if item is None:          # decode failed; same accounting as before
                skipped += 1
                continue
            # Guard against two annotations mapping to the same key (duplicate entries in
            # the source JSON, or two files differing only by extension). The pre-change
            # loop did NOT do this -- its `done` set was read once and never updated -- so
            # duplicates were written twice there too. This is an improvement over both
            # versions rather than a regression fix, suggested by an external review.
            if key in done:
                continue
            done.add(key)
            try:
                inputs = item["inputs"].to("cuda")
                out_ids = model.generate(
                    **inputs, do_sample=False, temperature=None, top_p=None, top_k=None,
                    max_new_tokens=args.max_new_tokens,
                )
                trimmed = [o[len(j):] for j, o in zip(inputs.input_ids, out_ids)]
                answer = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
                ts = extract_time(answer)
            except Exception as ex:  # one bad clip must not kill the run
                skipped += 1
                # `key` -- not `vid`, which this scope never binds. The old name raised
                # NameError from inside the guard and killed the shard the guard exists
                # to protect. `key` is bound above for every clip that reaches the try,
                # and it names the clip exactly (basename>>>query>>>span).
                print(f"  [skip {skipped}] {key} | {ex!r}", flush=True)
                continue
            f.write(json.dumps({key: {"timestamps": ts, "answers": answer, "duration": a["duration"]}}) + "\n")
            f.flush()
            n += 1
            if n % 50 == 0:
                print(f"  [{n}/{len(annos)}] cached (skipped {skipped})", flush=True)
    print(f"DONE shard {args.shard}: {n} preds, {skipped} skipped -> {args.out}")


if __name__ == "__main__":
    main()
