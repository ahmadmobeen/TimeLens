"""Evaluate stronger reward-1 LoRA adapter on a targeted subset of Charades-TimeLens test.

Targeted subset = all <2s clips + a capped random sample of mid/long bins (see --indices_json).
Identical decode recipe to eval_lora.py (fps=2, min64, total14336, greedy, >>>-key JSONL).

Usage (2 shards, one per GPU):
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-111 \\
      env -u VIRTUAL_ENV PYTHONPATH=. .venv/bin/python eval_lora_targeted.py \\
      --adapter /path/to/adapter \\
      --indices_json /tmp/stronger_target_indices.json \\
      --out logs/lora_eval/stronger_final_shard0.jsonl \\
      --shard 0 --num_shards 2

  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 112-223 \\
      env -u VIRTUAL_ENV PYTHONPATH=. .venv/bin/python eval_lora_targeted.py \\
      --adapter /path/to/adapter \\
      --indices_json /tmp/stronger_target_indices.json \\
      --out logs/lora_eval/stronger_final_shard1.jsonl \\
      --shard 1 --num_shards 2
"""
from __future__ import annotations

import argparse
import json
import os
from types import SimpleNamespace

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

from c_head import config as C
from timelens.dataset.timelens_data import DATASET_DICT
from timelens.utils import extract_time
from evaluation.utils import GroundingDataset


def _span(a):
    sp = a["span"][0] if isinstance(a["span"][0], (list, tuple)) else a["span"]
    return float(sp[0]), float(sp[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Path to the PEFT LoRA adapter directory")
    ap.add_argument("--indices_json", required=True, help="JSON file with list of anno indices to decode")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--max_new_tokens", type=int, default=512)
    args = ap.parse_args()

    # Load targeted indices
    with open(args.indices_json) as f:
        target_indices = json.load(f)

    # Load full test set then filter to targeted indices
    raw = DATASET_DICT["charades-timelens"].load_annos(split="test")
    all_annos = [a for a in raw if os.path.exists(a["video_path"])]
    target_set = set(target_indices)
    annos_all = [all_annos[i] for i in target_indices if i < len(all_annos)]

    # Shard
    annos = annos_all[args.shard :: args.num_shards]
    print(f"[shard {args.shard}/{args.num_shards}] {len(annos)} clips -> {args.out}", flush=True)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    # Resume support
    done: set[str] = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    done.add(next(iter(json.loads(line))))
                except Exception:
                    pass
    print(f"  resuming from {len(done)} done keys", flush=True)

    # Flash attn fallback
    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except Exception as _fa_err:
        attn_impl = "sdpa"
        print(f"[warn] flash_attn failed ({_fa_err!r}), using sdpa", flush=True)

    print(f"Loading base model {C.MODEL_PATH} (attn={attn_impl}) ...", flush=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        C.MODEL_PATH,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
    )
    print(f"Applying LoRA adapter from {args.adapter} ...", flush=True)
    model = PeftModel.from_pretrained(base_model, args.adapter)
    # An adapter whose keys miss attaches cleanly, raises nothing, and decodes
    # bit-identically to the frozen model, because lora_B is zero-initialised.
    # See NS-P1 ledger #80; repair with remap_adapter_keys.py.
    _nz = sum(1 for n, p in model.named_parameters()
              if "lora_B" in n and float(p.abs().sum()) > 0)
    if _nz == 0:
        raise SystemExit(f"ABORT: every lora_B in {args.adapter} is zero; this run "
                         f"would measure the frozen base model, not the adapter.")
    print(f"  adapter verified: {_nz} non-zero lora_B tensors", flush=True)
    model.eval()

    processor = AutoProcessor.from_pretrained(
        C.MODEL_PATH, padding_side="left", do_resize=False, trust_remote_code=True
    )
    ds_args = SimpleNamespace(
        model_path=C.MODEL_PATH,
        min_tokens=C.MIN_TOKENS,
        total_tokens=C.TOTAL_TOKENS,
        fps=args.fps,
        split="test",
    )
    ds = GroundingDataset(annos, processor, ds_args)

    n = skipped = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for i in range(len(ds)):
            a = ds.annos[i]
            s, e = _span(a)
            vid = os.path.splitext(os.path.basename(a["video_path"]))[0]
            key = f"{vid}.mp4>>>{a['query']}>>>{[s, e]}"
            if key in done:
                continue
            try:
                inputs = ds[i]["inputs"].to("cuda")
                with torch.no_grad():
                    out = model.generate(
                        **inputs,
                        do_sample=False,
                        temperature=None,
                        top_p=None,
                        top_k=None,
                        max_new_tokens=args.max_new_tokens,
                    )
                trimmed = [o[len(j):] for j, o in zip(inputs.input_ids, out)]
                ans = processor.batch_decode(trimmed, skip_special_tokens=True)[0]
                ts = extract_time(ans)
            except Exception as ex:
                skipped += 1
                print(f"  [skip {skipped}] {vid} | {ex!r}", flush=True)
                continue
            f.write(json.dumps({key: {"timestamps": ts, "answers": ans, "duration": float(a["duration"])}}) + "\n")
            f.flush()
            n += 1
            if n % 50 == 0:
                print(f"  [{n}/{len(annos)}] decoded (skipped {skipped})", flush=True)

    print(f"DONE shard {args.shard}: {n} preds, {skipped} skipped -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
