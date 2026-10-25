"""The answerability gate: is "can this interval answer the question?" a usable signal?

Both independent reviews (fable, codex) converged on the same reframing after the
similarity prior failed: stop asking *where does the video resemble the query* and ask
*does this interval contain enough evidence to answer it*. They also set the same stop
condition, which is what this script tests.

Ranking candidate intervals by a VLM's zero-shot yes/no logit is what FRAG already
does. So:

  * if zero-shot answerability is ALREADY strong (AUROC >= 0.85 against hard
    same-video negatives), the area belongs to FRAG and there is no paper here;
  * if it is weak (~0.6-0.75), a *calibrated* predictor trained on verified negatives
    has something to beat, and the framing survives to the next stage;
  * if it is at chance, the primitive does not exist at this model scale and the
    direction dies.

Any of the three is a result. The point is deciding it in hours, before a system is
built on top of a predicate that may not exist.

Candidates per question, following codex's design -- the hard negatives are the ones
that matter, because neighbouring windows are what a real scan must reject:
  gt         the true evidence window
  hard_neg   same video, displaced one window + 120 s, non-overlapping
  placebo    displaced 600 s, the E3 placebo condition
  uniform    evenly spread fillers

Run:
  env -u VIRTUAL_ENV CUDA_VISIBLE_DEVICES=0 PYTHONPATH=. \\
    .venv/bin/python -u probe_answerability.py --n_questions 120 --out logs/nsp5/gate.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import random

import torch
from decord import VideoReader
from transformers import AutoModelForImageTextToText, AutoProcessor

from eval_lvbench_foveate import load_questions

B = ("/home/mobeen/codes/active_repos/NeedleScan-vtg-landscape/research/TimeLens/"
     "data/lvbench")
WINDOW_S = 240.0      # the sigma=60 support: ~80% of E3's headroom sits at this scale
FRAMES_PER_WINDOW = 8
PROMPT = ("Can this video clip answer the following question? "
          "Answer with exactly one word, yes or no.\n\nQuestion: {}")


def windows_for(item: dict, n_uniform: int = 20) -> list[dict]:
    """Candidate intervals for one question, labelled by how they were constructed."""
    dur = float(item["duration"])
    half = WINDOW_S / 2
    centre = min(max(item["center"], half), max(half, dur - half))
    cands = [{"cand": "gt", "centre": centre}]

    for sign in (1, -1):                      # nearest non-overlapping neighbour
        c = centre + sign * (WINDOW_S + 120.0)
        if half <= c <= dur - half:
            cands.append({"cand": "hard_neg", "centre": c})
            break
    for sign in (1, -1):
        c = centre + sign * 600.0
        if half <= c <= dur - half:
            cands.append({"cand": "placebo", "centre": c})
            break

    taken = [c["centre"] for c in cands]
    for i in range(n_uniform):
        c = half + max(0.0, dur - WINDOW_S) * (i + 0.5) / n_uniform
        if all(abs(c - t) > WINDOW_S for t in taken):
            cands.append({"cand": "uniform", "centre": c})
            taken.append(c)
    for c in cands:
        c["start"] = max(0.0, c["centre"] - half)
        c["end"] = min(dur, c["centre"] + half)
        c["is_positive"] = c["cand"] == "gt"
    return cands


def clip_frames(vr, fps: float, total: int, start: float, end: float):
    idx = [min(total - 1, max(0, int((start + (end - start) * (i + 0.5) / FRAMES_PER_WINDOW) * fps)))
           for i in range(FRAMES_PER_WINDOW)]
    arr = torch.from_numpy(vr.get_batch(idx).asnumpy()).permute(0, 3, 1, 2)
    out = torch.nn.functional.interpolate(
        arr.float(), size=(308, 420), mode="bilinear",
        align_corners=False, antialias=True).clamp(0, 255).to(torch.uint8)
    return out, idx


@torch.no_grad()
def p_yes(model, proc, frames, idx, fps, total, question, yes_id, no_id, new_api) -> float:
    """P(yes) from the first generated token -- a logit, not a sampled word."""
    text = proc.apply_chat_template(
        [{"role": "user", "content": [{"type": "video"},
                                      {"type": "text", "text": PROMPT.format(question)}]}],
        tokenize=False, add_generation_prompt=True)
    meta = {"fps": fps, "frames_indices": idx, "total_num_frames": total,
            "video_backend": "decord"}
    if new_api:
        inp = proc(text=[text], images=None, videos=[frames], video_metadata=[meta],
                   do_sample_frames=False, padding=True, return_tensors="pt").to("cuda")
    else:
        inp = proc(text=[text], images=None, videos=[(frames, meta)],
                   padding=True, return_tensors="pt").to("cuda")
    logits = model(**inp).logits[0, -1].float()
    pair = torch.softmax(torch.stack([logits[yes_id], logits[no_id]]), dim=0)
    return float(pair[0])


def auroc(pos: list[float], neg: list[float]) -> float:
    """Mann-Whitney AUROC; 0.5 is chance, ties count half."""
    if not pos or not neg:
        return float("nan")
    wins = sum((p > n) + 0.5 * (p == n) for p in pos for n in neg)
    return wins / (len(pos) * len(neg))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default="Qwen/Qwen2.5-VL-7B-Instruct")
    ap.add_argument("--n_questions", type=int, default=120)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    items, census = load_questions(f"{B}/data/train-00000-of-00001.parquet",
                                   f"{B}/video_info.meta.jsonl", f"{B}/videos")
    print(f"population: {census}", flush=True)
    rng = random.Random(args.seed)
    rng.shuffle(items)
    items = items[: args.n_questions]

    proc = AutoProcessor.from_pretrained(args.model, padding_side="left",
                                         trust_remote_code=True)
    new_api = getattr(proc, "video_processor", None) is not None
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="cuda", trust_remote_code=True).eval()
    tok = proc.tokenizer
    yes_id, no_id = tok.encode("yes")[-1], tok.encode("no")[-1]
    print(f"scoring {len(items)} questions, window {WINDOW_S:.0f}s, "
          f"{FRAMES_PER_WINDOW} frames/clip, contract="
          f"{'video_metadata' if new_api else 'legacy tuple'}", flush=True)

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    scores: dict[str, list[float]] = {}
    ranks: list[tuple[int, int]] = []
    done = skipped = 0
    with open(args.out, "w", encoding="utf-8") as fh:
        for item in items:
            try:
                vr = VideoReader(item["video_path"])
                total, fps = len(vr), vr.get_avg_fps()
                cands = windows_for(item)
                for c in cands:
                    frames, idx = clip_frames(vr, fps, total, c["start"], c["end"])
                    c["p_yes"] = p_yes(model, proc, frames, idx, fps, total,
                                       item["question"], yes_id, no_id, new_api)
            except Exception as ex:
                skipped += 1
                print(f"  [skip {skipped}] {item['key']} | {ex!r}", flush=True)
                if done == 0 and skipped >= 10:
                    raise RuntimeError(f"10 failures, nothing scored: {ex!r}") from ex
                continue

            for c in cands:
                scores.setdefault(c["cand"], []).append(c["p_yes"])
                fh.write(json.dumps({"key": f"{item['key']}>>>{item['uid']}", **c}) + "\n")
            fh.flush()
            order = sorted(cands, key=lambda c: -c["p_yes"])
            ranks.append((next(i for i, c in enumerate(order) if c["is_positive"]),
                          len(cands)))
            done += 1
            if done % 20 == 0:
                print(f"  [{done}/{len(items)}] scored", flush=True)

    pos = scores.get("gt", [])
    hard = scores.get("hard_neg", [])
    allneg = [s for k, v in scores.items() if k != "gt" for s in v]
    r1 = sum(r == 0 for r, _ in ranks) / max(len(ranks), 1)
    r3 = sum(r < 3 for r, _ in ranks) / max(len(ranks), 1)
    print("\n=== ANSWERABILITY GATE ===", flush=True)
    print(f"  questions scored      {done}  (skipped {skipped})")
    print(f"  mean candidates/q     {sum(n for _, n in ranks)/max(len(ranks),1):.1f}")
    print(f"  AUROC vs ALL negs     {auroc(pos, allneg):.3f}")
    print(f"  AUROC vs HARD negs    {auroc(pos, hard):.3f}   <- the gate")
    print(f"  AUROC vs placebo      {auroc(pos, scores.get('placebo', [])):.3f}")
    print(f"  Recall@1              {r1*100:.1f}%   <- the gate")
    print(f"  Recall@3              {r3*100:.1f}%")
    print(f"  mean P(yes): gt {sum(pos)/max(len(pos),1):.3f}  "
          f"hard {sum(hard)/max(len(hard),1):.3f}  "
          f"all-neg {sum(allneg)/max(len(allneg),1):.3f}")
    # The two reviews set thresholds on DIFFERENT arms; conflating them makes the
    # result unreadable. This run measures the ZERO-SHOT arm only, so only the first
    # block applies. The calibrated-head thresholds are recorded for when/if MMJev's
    # weights become available.
    print("\n  READING THIS RUN (zero-shot arm):")
    print("    AUROC(hard) >= 0.85  -> zero-shot already solves it; FRAG owns the")
    print("                            area. STOP, no paper.")
    print("    AUROC(hard) ~0.60-0.75 -> LIVE: the predicate exists but is weak, so a")
    print("                            calibrated predictor has real headroom to beat.")
    print("    AUROC(hard) ~0.50    -> the primitive does not exist at 7B. Direction dies.")
    print("  NOT measured here (needs MMJev weights): the calibrated head, which would")
    print("  have to reach AUROC(hard) >= 0.85 and Recall@1 >= 40% to justify building.",
          flush=True)


if __name__ == "__main__":
    main()
