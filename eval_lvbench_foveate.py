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
import json
import os
import re

import torch
from decord import VideoReader
from qwen_vl_utils import process_vision_info
from transformers import AutoModelForImageTextToText, AutoProcessor

# Same kernel code as E1a/E1b/E2 -- imported, never copied.
from eval_foveate import kernel_times, times_to_indices

MODEL = "Qwen/Qwen2.5-VL-7B-Instruct"

PROMPT = (
    "{}\n\n"
    "Answer with the single letter of the correct option (A, B, C or D) and nothing else."
)

_TR = re.compile(r"^(\d+):(\d+)(?::(\d+))?-(\d+):(\d+)(?::(\d+))?$")
_LETTER = re.compile(r"\b([ABCD])\b")


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


def parse_letter(answer: str) -> str | None:
    """First standalone A/B/C/D in the model's reply, else None (counted, not guessed)."""
    m = _LETTER.search(answer.upper())
    return m.group(1) if m else None


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
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on the first N questions")
    args = ap.parse_args()

    items, census = load_questions(args.parquet, args.meta, args.video_dir)
    print(f"population: {census}", flush=True)
    if args.limit:
        items = items[: args.limit]

    # Resume: skip questions already written, so a killed run continues where it stopped.
    done: set[str] = set()
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    done.add(json.loads(line)["key"])
    print(f"arm={args.arm} n_frames={args.n_frames} todo={len(items) - len(done)}", flush=True)

    processor = AutoProcessor.from_pretrained(args.model, padding_side="left",
                                              trust_remote_code=True)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda", trust_remote_code=True).eval()

    realized: list[float] = []
    n_done = skipped = correct = unparsed = 0
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)

    with open(args.out, "a", encoding="utf-8") as out_fh:
        for item in items:
            key = f"{item['key']}>>>{item['uid']}"
            if key in done:
                continue
            try:
                # 1) Reference call at the frame budget: fixes N and the per-frame
                #    resolution. Both arms inherit them, so only placement differs.
                ref_msgs = [{"role": "user", "content": [
                    {"type": "video", "video": item["video_path"],
                     "nframes": args.n_frames, "min_pixels": 64 * 28 * 28,
                     "total_pixels": 14336 * 28 * 28},
                    {"type": "text", "text": "x"}]}]
                _, ref_videos = process_vision_info(ref_msgs, return_video_metadata=True)
                ref_tensor, meta = ref_videos[0]
                n, _, H, W = ref_tensor.shape
                native_fps = float(meta["fps"])
                total = int(meta["total_num_frames"])
                dur = total / native_fps

                # 2) Place N frames by the kernel, then read exactly those indices.
                center = min(max(item["center"], 0.0), dur)
                if args.arm == "uniform":
                    times = kernel_times("uniform", center, dur, n)
                else:
                    times = kernel_times(args.arm, center, dur, n,
                                         offpeak_shift=args.offpeak_shift)
                if args.arm.startswith("offpeak"):
                    realized.append(abs(min(dur, center + args.offpeak_shift) - center))
                idx = times_to_indices(times, native_fps, total, n)

                vr = VideoReader(item["video_path"])
                frames = torch.from_numpy(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2)
                frames = torch.nn.functional.interpolate(
                    frames.float(), size=(H, W), mode="bicubic",
                    align_corners=False, antialias=True).clamp(0, 255).to(ref_tensor.dtype)
                vmeta = {"fps": native_fps, "frames_indices": idx,
                         "total_num_frames": total, "video_backend": "decord"}

                # 3) Ask the multiple-choice question over exactly those frames.
                messages = [{"role": "user", "content": [
                    {"type": "video", "video": item["video_path"]},
                    {"type": "text", "text": PROMPT.format(item["question"])}]}]
                text = processor.apply_chat_template(messages, tokenize=False,
                                                     add_generation_prompt=True)
                inputs = processor(text=[text], images=None, videos=[(frames, vmeta)],
                                   padding=True, return_tensors="pt").to("cuda")
                out_ids = model.generate(**inputs, do_sample=False, temperature=None,
                                         top_p=None, top_k=None, max_new_tokens=16)
                reply = processor.batch_decode(
                    [out_ids[0][inputs.input_ids.shape[-1]:]], skip_special_tokens=True)[0]
            except Exception as ex:  # one bad video must not kill the run
                skipped += 1
                print(f"  [skip {skipped}] {item['key']} | {ex!r}", flush=True)
                continue

            pred = parse_letter(reply)
            if pred is None:
                unparsed += 1
            is_correct = pred is not None and pred == item["gold"]
            correct += int(is_correct)
            n_done += 1
            out_fh.write(json.dumps({
                "key": key, "arm": args.arm, "pred_letter": pred, "gold_letter": item["gold"],
                "correct": is_correct, "raw": reply.strip()[:80], "center_s": center,
                "span": item["span"], "duration_s": dur, "n_frames": len(idx)}) + "\n")
            out_fh.flush()
            if n_done % 50 == 0:
                print(f"  [{n_done}/{len(items)}] acc {correct / n_done:.4f} "
                      f"(skipped {skipped}, unparsed {unparsed})", flush=True)

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
