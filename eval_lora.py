"""Evaluate reward-1 LoRA adapter on Charades-TimeLens test set.

Mirrors c_head/fps_sweep.py decode recipe exactly:
  - AutoModelForImageTextToText, bfloat16, flash_attention_2 (sdpa fallback), device_map
  - AutoProcessor, padding_side=left, do_resize=False, trust_remote_code=True
  - GroundingDataset, fps=2, min_tokens=64, total_tokens=14336, split=test
  - greedy decode (do_sample=False)
  - extract_time to parse timestamps
  - >>>-key JSONL output

The ONLY difference from fps_sweep: base model is wrapped with PeftModel.

Usage (from research/TimeLens/, run each shard on its own GPU):
  # GPU 0 — shard 0 of 2
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-111 \\
      env -u VIRTUAL_ENV PYTHONPATH=. \\
      /path/to/venv/bin/python eval_lora.py \\
      --adapter /path/to/adapter \\
      --out logs/lora_eval/reward1_final_shard0.jsonl \\
      --shard 0 --num_shards 2

  # GPU 1 — shard 1 of 2
  CUDA_VISIBLE_DEVICES=1 OMP_NUM_THREADS=8 taskset -c 112-223 \\
      env -u VIRTUAL_ENV PYTHONPATH=. \\
      /path/to/venv/bin/python eval_lora.py \\
      --adapter /path/to/adapter \\
      --out logs/lora_eval/reward1_final_shard1.jsonl \\
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

import re

# NS-P1 loss-1 Stage 1 (reparam-SFT): when the adapter was trained center-first (REPARAM_CENTER=1),
# the model emits "The event is centered at C seconds and lasts W seconds" instead of "X - Y
# seconds". Parse (C, W) and reconstruct the symmetric span [C-W/2, C+W/2] so all downstream
# scoring (IoU, |dc|, length profile) is byte-identical to the span-format path.
_REPARAM_CENTER = os.environ.get("REPARAM_CENTER", "0").lower() not in ("0", "", "false", "no")
_CENTER_RE = re.compile(
    r"centered\s+at\s*([-+]?\d*\.?\d+)\s*seconds?.*?lasts?\s*([-+]?\d*\.?\d+)\s*seconds?",
    re.IGNORECASE | re.DOTALL,
)


def _parse_center_span(ans):
    """Parse center-first output and reconstruct [s, e] pairs (extract_time's shape)."""
    out = []
    for m in _CENTER_RE.finditer(ans):
        c = float(m.group(1))
        w = abs(float(m.group(2)))
        out.append([max(0.0, c - w / 2.0), c + w / 2.0])
    return out


def _span(a):
    sp = a["span"][0] if isinstance(a["span"][0], (list, tuple)) else a["span"]
    return float(sp[0]), float(sp[1])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="Path to the PEFT LoRA adapter directory")
    ap.add_argument("--out", required=True, help="Output JSONL path")
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--fps", type=int, default=2, help="Frame rate (must be 2 for apples-to-apples)")
    ap.add_argument("--max_new_tokens", type=int, default=512)
    ap.add_argument("--dataset", default="charades-timelens",
                    help="DATASET_DICT key (charades-timelens | activitynet-omniembed | ...); "
                         "for the paper's stratified ANet subset set TIMELENS_ANET_PER_BUCKET=100")
    ap.add_argument("--merge_lora", action="store_true",
                    help="Merge LoRA weights into base before inference (slightly faster, same results)")
    # no --max_moment_s: decode full test set for regression guard
    args = ap.parse_args()

    # Qwen3-VL (TimeLens-8B) hits the PyTorch 2.9 Conv3d regression in its vision patch
    # embedding, ~240,000x slower on this shape (issue #82, pytorch/pytorch#174051). This
    # was fixed in eval_bench.py, eval_zoom.py and the SFT trainer but not here, and the
    # trainer omission alone burned 59 GPU-hours before it was noticed. A no-op for
    # TimeLens-7B, which never instantiates the patched class.
    try:
        import fast_patch_embed
        fast_patch_embed.patch()
    except ImportError:
        print("[warn] fast_patch_embed not importable; 8B eval will be far slower", flush=True)

    if args.fps != 2:
        print(f"[warn] fps={args.fps} deviates from baseline fps=2; results won't be apples-to-apples")

    # Load full test set (no span filter) — gives both <2s stratum AND regression guard
    raw = DATASET_DICT[args.dataset].load_annos(split="test")
    annos = [a for a in raw if os.path.exists(a["video_path"])]
    annos = annos[args.shard :: args.num_shards]
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

    # Model: base + LoRA adapter — match fps_sweep exactly except for the PeftModel wrap
    # Test actual import (is_flash_attn_2_available can return True but still fail on import)
    attn_impl = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401 — confirm the .so actually loads
    except Exception as _fa_err:
        attn_impl = "sdpa"
        print(f"[warn] flash_attn import failed ({_fa_err!r}), falling back to sdpa", flush=True)

    # Take the base model from the ADAPTER'S OWN CONFIG rather than the module-level
    # C.MODEL_PATH default. C.MODEL_PATH is TimeLens-7B, so an 8B adapter was being applied
    # to a 7B base: the checkpoint's lora_B is [4096, 64] (8B hidden 4096) against a model
    # expecting [3584, 64] (7B hidden 3584). PEFT raised a size mismatch, which is the good
    # outcome -- but the mismatch should be impossible, not merely caught. The adapter
    # declares its own base, so read it and there is nothing to remember to pass.
    base_path = C.MODEL_PATH
    _cfg = os.path.join(args.adapter, "adapter_config.json")
    if os.path.exists(_cfg):
        with open(_cfg, encoding="utf-8") as _fh:
            _declared = json.load(_fh).get("base_model_name_or_path")
        if _declared and _declared != base_path:
            print(f"[base] adapter declares {_declared}; using it instead of the "
                  f"C.MODEL_PATH default {base_path}", flush=True)
            base_path = _declared

    print(f"Loading base model {base_path} (attn={attn_impl}) ...", flush=True)
    base_model = AutoModelForImageTextToText.from_pretrained(
        base_path,
        torch_dtype=torch.bfloat16,
        attn_implementation=attn_impl,
        device_map="auto",
    )
    print(f"Applying LoRA adapter from {args.adapter} ...", flush=True)
    model = PeftModel.from_pretrained(base_model, args.adapter)
    # PEFT injects adapters by suffix match on target_modules but restores weights by
    # full parameter path, and reports a total mismatch only in a return value nobody
    # reads. Because lora_B is zero-initialised, a failed restore leaves
    # dW = (alpha/r).B.A at exactly zero: the adapter attaches, nothing raises, and the
    # decode comes out bit-identical to the frozen model. That produced a month of
    # base-model numbers for the GRPO arm (NS-P1 ledger #80). Never decode without
    # checking. Modules frozen at train time are legitimately zero -- the vision tower
    # usually is -- so the bar is "some lora_B moved", not "all of them".
    _nz = sum(1 for n, p in model.named_parameters()
              if "lora_B" in n and float(p.abs().sum()) > 0)
    if _nz == 0:
        raise SystemExit(
            f"ABORT: adapter {args.adapter} attached but every lora_B is zero, so it is "
            f"a no-op and this run would measure the frozen base model. The usual cause "
            f"is a checkpoint saved under a different module tree; repair it with "
            f"research/TimeLens/remap_adapter_keys.py and re-run.")
    print(f"  adapter verified: {_nz} non-zero lora_B tensors", flush=True)
    if args.merge_lora:
        print("  merging LoRA weights ...", flush=True)
        model = model.merge_and_unload()
    model.eval()

    processor = AutoProcessor.from_pretrained(
        base_path, padding_side="left", do_resize=False, trust_remote_code=True
    )
    ds_args = SimpleNamespace(
        model_path=base_path,   # must match the model actually loaded: GroundingDataset
                                # switches its pixel/token budget on the model family
                                # (downsample_rate 32 for Qwen3-VL vs 28 for Qwen2.5-VL),
                                # so a 7B path here would silently mis-size 8B inputs
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
                ts = _parse_center_span(ans) if _REPARAM_CENTER else extract_time(ans)
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
