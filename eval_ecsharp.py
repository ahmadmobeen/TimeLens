"""Evaluate an EC-Sharp adapter+head on Charades-TimeLens test.

The DEPLOYED predictor is the head's soft-argmax expected center (NOT the text decode): one prefill
forward per clip (no autoregressive generation) -> per-frame states + last-prompt-token query ->
ECSharpHead -> (center, width) -> span in seconds. Writes the SAME >>>-key JSONL as eval_lora.py so
`score_lora.py --lora_from_timestamps` scores it apples-to-apples vs the 1.05s baseline.

The head rides in the adapter's non_lora_state_dict.bin (get_peft_state_non_lora_maybe_zero_3 saved
all non-LoRA trainable params, i.e. ec_head.*, at train end).

Run per-shard per-GPU from research/TimeLens/ (main-model venv):
  CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=8 taskset -c 0-111 env -u VIRTUAL_ENV PYTHONPATH=. \\
    .venv/bin/python eval_ecsharp.py --adapter <lora_dir> \\
    --out logs/lora_eval/ecsharp_shard0.jsonl --shard 0 --num_shards 2
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
from evaluation.utils import GroundingDataset
from training.ecsharp import ECSharpConfig, ECSharpHead, build_perframe_query, vis_token_ids


def _span(a):
    sp = a["span"][0] if isinstance(a["span"][0], (list, tuple)) else a["span"]
    return float(sp[0]), float(sp[1])


def _load_head(adapter_dir: str, hidden: int, device: str) -> ECSharpHead:
    p = os.path.join(adapter_dir, "non_lora_state_dict.bin")
    if not os.path.exists(p):
        raise FileNotFoundError(f"non_lora_state_dict.bin (carries ec_head.*) not found in {adapter_dir}")
    sd = torch.load(p, map_location="cpu")
    hsd = {k.split("ec_head.", 1)[1]: v for k, v in sd.items() if "ec_head." in k}
    if not hsd:
        raise ValueError(f"no ec_head.* keys in {p}; sample keys: {list(sd)[:6]}")
    head = ECSharpHead(hidden)
    missing, unexpected = head.load_state_dict(hsd, strict=False)
    print(f"  loaded ec_head ({len(hsd)} tensors) missing={list(missing)} unexpected={list(unexpected)}", flush=True)
    return head.to(device).eval()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="PEFT LoRA adapter dir (also holds non_lora_state_dict.bin)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num_shards", type=int, default=1)
    ap.add_argument("--fps", type=int, default=2)
    ap.add_argument("--merge_lora", action="store_true")
    args = ap.parse_args()

    cfg = ECSharpConfig()
    raw = DATASET_DICT["charades-timelens"].load_annos(split="test")
    annos = [a for a in raw if os.path.exists(a["video_path"])][args.shard :: args.num_shards]
    print(f"[shard {args.shard}/{args.num_shards}] {len(annos)} clips -> {args.out}  (layer={cfg.layer})", flush=True)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    done: set[str] = set()
    if os.path.exists(args.out):
        for line in open(args.out, encoding="utf-8"):
            line = line.strip()
            if line:
                try:
                    done.add(next(iter(json.loads(line))))
                except Exception:
                    pass
    print(f"  resuming from {len(done)} done", flush=True)

    attn = "flash_attention_2"
    try:
        import flash_attn  # noqa: F401
    except Exception as e:
        attn = "sdpa"
        print(f"[warn] flash_attn import failed ({e!r}) -> sdpa", flush=True)

    print(f"Loading base {C.MODEL_PATH} (attn={attn}) + adapter {args.adapter}", flush=True)
    base = AutoModelForImageTextToText.from_pretrained(
        C.MODEL_PATH, torch_dtype=torch.bfloat16, attn_implementation=attn, device_map="auto"
    )
    model = PeftModel.from_pretrained(base, args.adapter)
    # An adapter whose keys miss attaches cleanly, raises nothing, and decodes
    # bit-identically to the frozen model, because lora_B is zero-initialised.
    # See NS-P1 ledger #80; repair with remap_adapter_keys.py.
    _nz = sum(1 for n, p in model.named_parameters()
              if "lora_B" in n and float(p.abs().sum()) > 0)
    if _nz == 0:
        raise SystemExit(f"ABORT: every lora_B in {args.adapter} is zero; this run "
                         f"would measure the frozen base model, not the adapter.")
    print(f"  adapter verified: {_nz} non-zero lora_B tensors", flush=True)
    if args.merge_lora:
        model = model.merge_and_unload()
    model.eval()

    hidden = int(getattr(model.config, "hidden_size", 3584) or 3584)
    head = _load_head(args.adapter, hidden, "cuda")
    vis_ids = torch.tensor(vis_token_ids(model.config), dtype=torch.long, device="cuda")

    processor = AutoProcessor.from_pretrained(
        C.MODEL_PATH, padding_side="left", do_resize=False, trust_remote_code=True
    )
    ds_args = SimpleNamespace(
        model_path=C.MODEL_PATH, min_tokens=C.MIN_TOKENS, total_tokens=C.TOTAL_TOKENS,
        fps=args.fps, split="test",
    )
    ds = GroundingDataset(annos, processor, ds_args)

    n = skipped = 0
    with open(args.out, "a", encoding="utf-8") as f:
        for i in range(len(ds)):
            a = ds.annos[i]
            s, e = _span(a)
            dur = float(a["duration"])
            vid = os.path.splitext(os.path.basename(a["video_path"]))[0]
            key = f"{vid}.mp4>>>{a['query']}>>>{[s, e]}"
            if key in done:
                continue
            try:
                inputs = ds[i]["inputs"].to("cuda")
                with torch.no_grad():
                    out = model(**inputs, output_hidden_states=True, use_cache=False)
                hs = out.hidden_states[cfg.layer]
                grid = inputs.get("video_grid_thw", inputs.get("image_grid_thw"))
                built = build_perframe_query(hs, inputs["input_ids"], grid, vis_ids, labels=None)
                if built is None:
                    raise ValueError("degenerate frame/token layout")
                V, mask, pos, hq = built
                center, w, _ = head(V, mask, pos, hq)
                c_sec = float(center[0]) * dur
                w_sec = float(w[0]) * dur
                ps = max(0.0, c_sec - w_sec / 2.0)
                pe = min(dur, c_sec + w_sec / 2.0)
                if pe <= ps:
                    pe = min(dur, ps + 0.1)
            except Exception as ex:
                skipped += 1
                print(f"  [skip {skipped}] {vid} | {ex!r}", flush=True)
                continue
            f.write(json.dumps({key: {
                "timestamps": [[ps, pe]],
                "answers": f"ec_center={c_sec:.2f} w={w_sec:.2f}",
                "duration": dur,
            }}) + "\n")
            f.flush()
            n += 1
            if n % 50 == 0:
                print(f"  [{n}/{len(annos)}] (skipped {skipped})", flush=True)

    print(f"DONE shard {args.shard}: {n} preds, {skipped} skipped -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
