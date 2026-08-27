"""NS-P1 confidence probe: is the grounder confidently wrong on short moments?

Two measurements per clip, both under the OFFICIAL eval inputs (GroundingDataset):

  1. Greedy decode with scores. Per generated token we keep the log-probability of
     the chosen token and the entropy of the full next-token distribution. Tokens
     are tagged as digit / non-digit, because the answer template ("The event
     happens in X - Y seconds.") is boilerplate and would be low-entropy for any
     model. The claim under test is about the DIGITS.

  2. Teacher-forced scoring of the ground-truth answer. We score the string the
     model *should* have produced under the same visual prefix and compare it to
     the log-probability of the answer it *did* produce. If the wrong answer
     outscores the true one by a wide margin, the model is not uncertain between
     the two -- it is confident and wrong.

Run (TimeLens venv, from research/TimeLens), one shard per GPU:
  env -u VIRTUAL_ENV CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 PYTHONPATH=. \\
    .venv/bin/python -u eval_confidence.py --shard 0 --nshards 2 \\
    --out logs/confidence/probe_s0.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from transformers import AutoModelForImageTextToText, AutoProcessor

from timelens.dataset.timelens_data import DATASET_DICT
from evaluation.utils import GroundingDataset
from timelens.utils import extract_time

MODEL = "TencentARC/TimeLens-7B"
S0_MAX_LEN = 2.0
NONS0_SAMPLE_SEED = 20260807
SUBSAMPLE_SEED = 20260808


def span_of(anno: dict) -> tuple[float, float]:
    sp = anno["span"][0] if isinstance(anno["span"][0], (list, tuple)) else anno["span"]
    return float(sp[0]), float(sp[1])


def iou_of(pred: tuple[float, float], gt: tuple[float, float]) -> float:
    inter = max(0.0, min(pred[1], gt[1]) - max(pred[0], gt[0]))
    union = max(pred[1], gt[1]) - min(pred[0], gt[0])
    return inter / union if union > 0 else 0.0


def gt_answer_text(gt: tuple[float, float]) -> str:
    """The answer string the model should have produced, in its own format."""
    return f"The event happens in {gt[0]:.1f} - {gt[1]:.1f} seconds."


def control_span(gt: tuple[float, float], duration: float) -> tuple[float, float]:
    """A deterministic decoy: same length as the truth, elsewhere in the video.

    Matching the length keeps token count and digit count comparable, so the
    only thing that differs from the ground-truth answer is WHERE it points.
    If a finetune raises the decoy's likelihood as much as the truth's, the arm
    merely flattened its distribution rather than learning where the moment is.
    """
    L = gt[1] - gt[0]
    D = max(float(duration), L + 0.2)
    c2 = D - (gt[0] + gt[1]) / 2.0          # reflect about the video midpoint
    lo, hi = c2 - L / 2.0, c2 + L / 2.0
    overlap = max(0.0, min(hi, gt[1]) - max(lo, gt[0]))
    if overlap > 0:                          # centred moment: fall back to the roomier end
        lo, hi = (0.0, L) if gt[0] > D - gt[1] else (D - L, D)
    lo = max(0.0, min(lo, D - L))
    return (round(lo, 1), round(lo + L, 1))


def build_split(limit: int, nons0_per_s0: float, subsample: int = 0,
                subsample_seed: int = SUBSAMPLE_SEED) -> list[dict]:
    """All S0 clips plus a seeded random comparison sample of longer moments.

    ``subsample`` restricts the S0 half to a deterministic sample, so every
    finetune arm is probed on an identical clip set and can be paired offline
    by key. Selection sorts by a stable key first, so it does not depend on the
    order the annotations happen to load in.
    """
    import random

    raw = DATASET_DICT["charades-timelens"].load_annos(split="test")
    avail = [a for a in raw if os.path.exists(a["video_path"])]
    s0, nons0 = [], []
    for a in avail:
        lo, hi = span_of(a)
        (s0 if hi - lo < S0_MAX_LEN else nons0).append(a)
    for a in s0:
        a["_stratum"] = "s0"
    for a in nons0:
        a["_stratum"] = "nons0"
    if subsample and subsample < len(s0):
        s0.sort(key=lambda a: (os.path.basename(a["video_path"]), a["query"],
                               str(list(span_of(a)))))
        s0 = random.Random(subsample_seed).sample(s0, subsample)
    rng = random.Random(NONS0_SAMPLE_SEED)
    rng.shuffle(nons0)
    keep = s0 + nons0[: int(len(s0) * nons0_per_s0)]
    if limit:
        keep = keep[:limit]
    return keep


@torch.no_grad()
def score_forced(model, inputs, target_ids: torch.Tensor) -> float:
    """Sum log p(target | visual prefix) under teacher forcing."""
    plen = inputs["input_ids"].shape[-1]
    fwd = {k: v for k, v in inputs.items()}
    fwd["input_ids"] = torch.cat([inputs["input_ids"], target_ids], dim=1)
    fwd["attention_mask"] = torch.cat(
        [inputs["attention_mask"], torch.ones_like(target_ids)], dim=1)
    logits = model(**fwd).logits[:, plen - 1 : -1, :].float()
    lp = F.log_softmax(logits, dim=-1)
    return float(lp.gather(-1, target_ids.unsqueeze(-1)).sum())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--nons0_per_s0", type=float, default=1.0)
    ap.add_argument("--model", default=MODEL,
                    help="checkpoint to probe; defaults to the frozen grounder")
    ap.add_argument("--arm", default="frozen", help="label stored in each record")
    ap.add_argument("--adapter", default=None,
                    help="optional PEFT/LoRA adapter applied on top of --model")
    ap.add_argument("--subsample", type=int, default=0,
                    help="restrict S0 to a deterministic sample shared across arms")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    cli = ap.parse_args()

    clips = build_split(cli.limit, cli.nons0_per_s0, cli.subsample)
    os.makedirs(os.path.dirname(os.path.abspath(cli.out)), exist_ok=True)

    done: set[str] = set()
    if os.path.exists(cli.out):
        for line in open(cli.out, encoding="utf-8"):
            if line.strip():
                try:
                    done.add(next(iter(json.loads(line))))
                except Exception:
                    pass

    processor = AutoProcessor.from_pretrained(
        cli.model, padding_side="left", do_resize=False, trust_remote_code=True)
    tok = processor.tokenizer
    ds_args = SimpleNamespace(model_path=cli.model, split="test", min_tokens=64,
                              total_tokens=14336, fps=2, max_new_tokens=512)
    ds = GroundingDataset(clips, processor, ds_args)
    model = AutoModelForImageTextToText.from_pretrained(
        cli.model, dtype=torch.bfloat16, attn_implementation="sdpa",
        device_map="auto").eval()
    if cli.adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, cli.adapter).eval()
        # PEFT attaches LoRA by SUFFIX match but loads weights by FULL PATH, and it
        # reports success either way. When the checkpoint was saved under an older
        # transformers the language model sat at model.layers.*, where this version
        # nests it at model.language_model.layers.* -- so every key silently missed,
        # lora_B stayed at its zero init, and dW was exactly 0. On 2026-08-08 that
        # produced a full day of "the adapter changes nothing" results that were
        # really the base model. A non-zero lora_B is the only proof it loaded.
        nz = sum(1 for n, q in model.named_parameters()
                 if "lora_B" in n and float(q.abs().sum()) > 0)
        if nz == 0:
            raise SystemExit(
                f"ABORT: adapter {cli.adapter} attached but every lora_B is zero, so it\n"
                "  has no effect. The checkpoint keys almost certainly do not match this\n"
                "  model's module paths. Remap them before trusting any number."
            )
        print(f"[confidence] applied adapter {cli.adapter} "
              f"({nz} non-zero lora_B tensors)", flush=True)

    mine = [i for i in range(len(clips)) if i % cli.nshards == cli.shard]
    print(f"[confidence] shard {cli.shard}/{cli.nshards}: {len(mine)} clips "
          f"(resume {len(done)}) -> {cli.out}", flush=True)

    n_done = skipped = 0
    with open(cli.out, "a", encoding="utf-8") as f:
        for i in mine:
            a = clips[i]
            gt = span_of(a)
            vid = os.path.basename(a["video_path"])
            key = f"{vid}>>>{a['query']}>>>{list(gt)}"
            if key in done:
                continue
            try:
                inputs = ds[i]["inputs"].to("cuda")
                out = model.generate(**inputs, do_sample=False, max_new_tokens=512,
                                     return_dict_in_generate=True, output_scores=True)
                plen = inputs["input_ids"].shape[-1]
                gen = out.sequences[0][plen:]
                answer = processor.decode(gen, skip_special_tokens=True)

                tok_lp, tok_ent, is_digit = [], [], []
                for step, logit in enumerate(out.scores):
                    if step >= gen.shape[0]:
                        break
                    lp = F.log_softmax(logit[0].float(), dim=-1)
                    tid = int(gen[step])
                    tok_lp.append(float(lp[tid]))
                    tok_ent.append(float(-(lp.exp() * lp).sum()))
                    piece = tok.decode([tid])
                    is_digit.append(any(c.isdigit() for c in piece))

                pred_lp = float(sum(tok_lp))
                tgt_ids = tok(gt_answer_text(gt), add_special_tokens=False,
                              return_tensors="pt").input_ids.to(inputs["input_ids"].device)
                gt_lp = score_forced(model, inputs, tgt_ids)

                cs = control_span(gt, a["duration"])
                ctrl_ids = tok(gt_answer_text(cs), add_special_tokens=False,
                               return_tensors="pt").input_ids.to(inputs["input_ids"].device)
                ctrl_lp = score_forced(model, inputs, ctrl_ids)

                spans = extract_time(answer)
                pred = tuple(spans[0]) if spans else (0.0, 0.0)
                rec = {
                    "pred_span": list(pred), "gt_span": list(gt),
                    "iou": iou_of(pred, gt), "answer": answer,
                    "tok_logprob": tok_lp, "tok_entropy": tok_ent,
                    "is_digit": is_digit,
                    "pred_answer_logprob": pred_lp,
                    "gt_answer_logprob": gt_lp,
                    "ctrl_span": list(cs),
                    "ctrl_answer_logprob": ctrl_lp,
                    "ctrl_iou_gt": iou_of(cs, gt),
                    "n_ctrl_tokens": int(ctrl_ids.shape[-1]),
                    "n_pred_tokens": len(tok_lp),
                    "n_gt_tokens": int(tgt_ids.shape[-1]),
                    "duration": a["duration"], "stratum": a["_stratum"],
                    "arm": cli.arm,
                }
            except Exception as ex:
                skipped += 1
                print(f"  [skip {skipped}] {vid} | {ex!r}", flush=True)
                continue
            f.write(json.dumps({key: rec}) + "\n")
            f.flush()
            n_done += 1
            if n_done % 25 == 0:
                print(f"  [{n_done}/{len(mine)}] done (skipped {skipped})", flush=True)
    print(f"DONE: {n_done} clips, {skipped} skipped -> {cli.out}", flush=True)


if __name__ == "__main__":
    main()
