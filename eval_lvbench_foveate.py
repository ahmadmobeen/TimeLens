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


def curve_times(z, dur: float, n: int, alloc: str, tau: float, eps: float) -> list[float]:
    """Frame times from a per-second score curve (second k covers [k, k+1)).

    topk   the n highest-scoring seconds.
    kelly  inverse-CDF quantiles of (1 - eps) * softmax(z / tau) + eps * uniform: frames
           in proportion to posterior mass. When accuracy grows with log(frames on the
           evidence), as E3's ladder shows, proportional allocation maximises expected
           accuracy; eps of the budget stays uniform as a hedge against a confidently
           wrong posterior (E3's placebo: concentration in the wrong place costs 5 pts).
    """
    import numpy as np
    z = np.asarray(z, dtype=np.float64)[:max(1, int(np.ceil(dur)))]
    T = len(z)
    if alloc == "topk":
        return sorted(float(k) + 0.5 for k in np.argsort(-z)[:n])
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
    ap.add_argument("--alloc", choices=["topk", "kelly"], default="kelly",
                    help="curve arm: how a curve becomes frame times (see curve_times)")
    ap.add_argument("--tau", type=float, default=1.0, help="kelly: posterior temperature")
    ap.add_argument("--eps", type=float, default=0.5, help="kelly: uniform share of the budget")
    ap.add_argument("--curve_placebo", action="store_true",
                    help="curve arm PLACEBO: allocate with ANOTHER question's curve from the "
                         "SAME video -- same video, same allocator, wrong question")
    ap.add_argument("--bench", default="lvbench",
                    choices=["lvbench", "videomme", "lvb", "mlvu", "cgbench"],
                    help="question set; non-LVBench sets come through bench.py")
    ap.add_argument("--seed", type=int, default=0,
                    help="fixes the --shuffle_query rotation")
    ap.add_argument("--limit", type=int, default=0, help="smoke-test on the first N questions")
    ap.add_argument("--sample", type=int, default=0,
                    help="a seeded random subset of N questions (kept in video order, so the "
                         "per-video reference memo still applies); --limit takes the first N, "
                         "which covers only a handful of videos")
    args = ap.parse_args()

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
    if os.path.exists(args.out):
        with open(args.out, encoding="utf-8") as fh:
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
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, attn_implementation="sdpa",
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
            assert len(keys) >= 2 or not args.curve_placebo, keys
            for i, k in enumerate(keys):         # placebo: the next question, same video
                src = keys[(i + 1) % len(keys)] if args.curve_placebo else k
                curve_of[k] = qkey_of[src]
        print(f"curves: {args.curves} alloc={args.alloc} tau={args.tau} eps={args.eps}"
              f"{' PLACEBO (same video, other question)' if args.curve_placebo else ''}",
              flush=True)

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
    # Same tensors, same metadata: a speedup, not a protocol change.
    ref_memo: dict[str, tuple] = {}
    vr_memo: dict = {}

    with open(args.out, "a", encoding="utf-8") as out_fh:
        for item in items:
            key = item["rkey"]
            if key in done:
                continue
            try:
                # 1) Reference call at the frame budget: fixes N and the per-frame
                #    resolution. Both arms inherit them, so only placement differs.
                if item["video_path"] not in ref_memo:
                    ref_msgs = [{"role": "user", "content": [
                        {"type": "video", "video": item["video_path"],
                         "nframes": args.n_frames, "min_pixels": 64 * 28 * 28,
                         "total_pixels": 14336 * 28 * 28},
                        {"type": "text", "text": "x"}]}]
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
                if args.arm in ("uniform", "topk_siglip", "curve"):
                    # topk_siglip only needs n and the frame grid here; the actual
                    # indices are replaced by the SigLIP selection below.
                    times = kernel_times("uniform", center, dur, n)
                else:
                    times = kernel_times(args.arm, center, dur, n,
                                         offpeak_shift=args.offpeak_shift)
                if args.arm.startswith("offpeak"):
                    realized.append(abs(min(dur, center + args.offpeak_shift) - center))
                if args.arm == "curve":
                    times = curve_times(curves[curve_of[key]], dur, n, args.alloc,
                                        args.tau, args.eps)
                idx = times_to_indices(times, native_fps, total, n)
                if vr_memo.get("path") != item["video_path"]:   # one reader per video
                    vr_memo.update(path=item["video_path"], vr=VideoReader(item["video_path"]))
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
                gen = model.generate(**inputs, do_sample=False, temperature=None,
                                     top_p=None, top_k=None, max_new_tokens=16,
                                     output_scores=True, return_dict_in_generate=True)
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
                "span": item["span"], "duration_s": dur, "n_frames": len(idx),
                "in_span": (sum(any(a - 0.5 <= i / native_fps <= b + 0.5 for a, b in item["spans"])
                                for i in idx) if item["spans"] else None),
                "p_letters": [round(p, 5) for p in p_letters]}) + "\n")
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
