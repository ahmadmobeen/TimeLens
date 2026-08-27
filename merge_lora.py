"""Merge a TimeLens-7B LoRA adapter into the base weights -> a standalone HF model dir.

Since the D-lever pilot froze llm/merger/vision (LoRA-only), `merge_and_unload()` captures
every trained parameter. The output dir name MUST contain "timelens-7b" so the eval harness
(`evaluation/utils.py`) takes the Qwen2.5-VL processor branch. Processor/code files from the
adapter dir are copied alongside so the merged dir is self-contained for eval.

Usage:
  python merge_lora.py --adapter <lora_dir> --out <merged_dir>
"""
import argparse
import glob
import os
import shutil

import torch
from peft import PeftModel
from transformers import AutoModelForImageTextToText, AutoProcessor

BASE = "TencentARC/TimeLens-7B"
# processor/runtime files the eval needs that LoRA save copies into the adapter dir
COPY_GLOBS = ("*.json", "*.jinja", "*.txt", "*.py")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="LoRA adapter dir (contains adapter_config.json)")
    ap.add_argument("--out", required=True, help="Output merged model dir (must contain 'timelens-7b')")
    ap.add_argument("--base", default=BASE)
    a = ap.parse_args()

    assert "timelens-7b" in a.out.lower(), "out dir must contain 'timelens-7b' for the eval 7B branch"

    print(f"[merge] base={a.base}  adapter={a.adapter}")
    model = AutoModelForImageTextToText.from_pretrained(
        a.base, torch_dtype=torch.bfloat16, attn_implementation="sdpa", trust_remote_code=True
    )
    model = PeftModel.from_pretrained(model, a.adapter)
    # A silently-unloaded adapter has lora_B at its zero init, so merging it writes out
    # a byte-for-byte copy of the base model under a finetuned name -- and once merged
    # there is no adapter left to inspect, so the mistake becomes permanent and
    # invisible. See NS-P1 ledger #80; repair with remap_adapter_keys.py.
    _nz = sum(1 for n, p in model.named_parameters()
              if "lora_B" in n and float(p.abs().sum()) > 0)
    if _nz == 0:
        raise SystemExit(f"ABORT: every lora_B in {a.adapter} is zero; merging would "
                         f"write out the base model as if it were finetuned.")
    print(f"[merge] adapter verified: {_nz} non-zero lora_B tensors")
    model = model.merge_and_unload()
    os.makedirs(a.out, exist_ok=True)
    model.save_pretrained(a.out, safe_serialization=True)

    # processor + custom code: prefer the adapter dir's copies, fall back to base
    proc = AutoProcessor.from_pretrained(a.adapter, trust_remote_code=True)
    proc.save_pretrained(a.out)
    for pat in COPY_GLOBS:
        for f in glob.glob(os.path.join(a.adapter, pat)):
            bn = os.path.basename(f)
            if bn.startswith("adapter_") or bn == "non_lora_state_dict.bin":
                continue
            dst = os.path.join(a.out, bn)
            if not os.path.exists(dst):
                shutil.copy2(f, dst)
    print(f"[merge] saved merged model -> {a.out}")
    print("[merge] files:", sorted(os.listdir(a.out)))


if __name__ == "__main__":
    main()
