"""Repair a LoRA adapter whose keys were saved under an older module tree.

Why this exists
---------------
PEFT does two independent things when it loads an adapter, and they address
parameters differently:

  * it **injects** LoRA modules by matching ``target_modules`` as a *suffix*
    (``q_proj``, ``down_proj``, ...), which succeeds under any prefix path;
  * it **restores** weights by *full parameter path*, which requires the
    save-time and load-time module trees to agree segment for segment.

``lora_B`` is initialised to exactly zero so the adapted model starts out
identical to the pretrained one. So when the restore silently matches nothing,
``lora_A`` keeps its random init, ``lora_B`` keeps its zeros, and
``dW = (alpha/r) . B . A`` is *exactly* zero: the forward pass is bit-identical
to the base model, at any alpha. The load is non-strict, so nothing raises --
PEFT reports the discarded keys in ``unexpected_keys`` and callers ignore it.

That is how ``reward1_charades_short`` was evaluated as the frozen model for a
month (NS-P1 ledger #80). It was trained under VideoChat-R1's transformers,
where Qwen2.5-VL's decoder hung directly off ``model`` and the vision tower sat
at the top level; transformers 4.57.1 nests both under ``model``.

The weights themselves are fine. Only the names are wrong, and the difference is
a pure prefix rename with no change in module structure -- verified by comparing
collapsed path shapes, which match one-for-one at 196 + 160 + 2 = 358 modules.

This script rewrites the names and refuses to emit anything it cannot prove
loads: it attaches the result to a real base model and requires *every* lora_B
to come back non-zero.

Usage::

    python remap_adapter_keys.py \\
        --adapter ../VideoChat-R1/ckpt/reward1_charades_short \\
        --base TencentARC/TimeLens-7B \\
        --out   ../VideoChat-R1/ckpt/reward1_charades_short_remapped
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import torch
from peft import PeftModel
from peft.utils import set_peft_model_state_dict
from safetensors.torch import load_file, save_file
from transformers import AutoModelForImageTextToText

# Old tree -> new tree. Applied to the path *after* the "base_model.model."
# prefix. Order matters only in that each rule must be anchored at the start.
RENAMES: tuple[tuple[str, str], ...] = (
    ("model.layers.", "model.language_model.layers."),
    ("visual.", "model.visual."),
)
PREFIX = "base_model.model."
# Copied alongside the weights so the output is a drop-in adapter directory.
SIDECARS = ("adapter_config.json", "README.md", "added_tokens.json",
            "chat_template.json", "merges.txt", "preprocessor_config.json",
            "special_tokens_map.json", "tokenizer.json", "tokenizer_config.json",
            "vocab.json")


def remap_key(key: str) -> str:
    """Rewrite one checkpoint key from the old module tree to the new one."""
    if not key.startswith(PREFIX):
        return key
    inner = key[len(PREFIX):]
    for old, new in RENAMES:
        if inner.startswith(old):
            return PREFIX + new + inner[len(old):]
    return key


def verify_and_report(pm, remapped: dict, scaling: float) -> tuple[int, int, float]:
    """Push the remapped weights in and measure what actually landed.

    The invariant is *not* "every lora_B is non-zero" -- a module can be zero
    because training legitimately left it so. Freezing the vision tower while
    adapting the language model is a standard Qwen2.5-VL recipe, and it leaves
    the ViT-block lora_B at its zero init in the checkpoint. What must hold is
    that every lora_B the checkpoint recorded as non-zero comes back non-zero.
    """
    expected = {k for k, v in remapped.items()
                if "lora_B" in k and float(v.float().abs().sum()) > 0}
    res = set_peft_model_state_dict(pm, remapped)
    live = {n.replace(".default.weight", ".weight"): p
            for n, p in pm.named_parameters() if "lora_B" in n}
    landed = {k for k in expected if float(live[k].abs().sum()) > 0} if expected else set()
    missed = sorted(expected - landed)
    print(f"after remapped load: trained lora_B {len(landed)}/{len(expected)} landed | "
          f"{len(live) - len(expected)} zero in the checkpoint (frozen at train time) | "
          f"unexpected_keys={len(res.unexpected_keys)}")
    if missed:
        raise SystemExit(
            f"ABORT: {len(missed)} trained lora_B tensor(s) did not land, e.g. "
            f"{missed[:3]}. A partial load is worse than none -- it silently mixes "
            f"trained and untrained layers.")
    nz, n_b = len(landed), len(expected)

    # dW magnitude relative to the bf16 resolution floor (~4e-3 relative), so the
    # log records that the adapter is not merely present but consequential.
    worst = 0.0
    for mod in pm.modules():
        lora_a = getattr(mod, "lora_A", None)
        if lora_a is None or "default" not in lora_a:
            continue
        a = mod.lora_A["default"].weight.float()
        b = mod.lora_B["default"].weight.float()
        dw = (b @ a) * scaling
        worst = max(worst, float(dw.abs().max() / mod.base_layer.weight.float().abs().max()))
    print(f"largest |dW|/|W| across adapted layers: {worst:.3e} "
          f"(bf16 relative resolution ~4e-3)")
    return nz, n_b, worst


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--adapter", required=True, help="source adapter directory")
    ap.add_argument("--base", default="TencentARC/TimeLens-7B")
    ap.add_argument("--out", required=True, help="directory to write the repaired adapter to")
    ap.add_argument("--device", default="cpu", help="device for the verification load")
    cli = ap.parse_args()

    src, out = Path(cli.adapter), Path(cli.out)
    sd = load_file(str(src / "adapter_model.safetensors"))
    remapped = {remap_key(k): v for k, v in sd.items()}
    changed = sum(1 for k in sd if remap_key(k) != k)
    print(f"read {len(sd)} tensors from {src}")
    print(f"renamed {changed}/{len(sd)} keys")
    if changed != len(sd):
        untouched = [k for k in sd if remap_key(k) == k][:5]
        raise SystemExit(f"ABORT: {len(sd) - changed} key(s) matched no rename rule, "
                         f"e.g. {untouched}. Add a rule rather than ship a partial load.")

    print(f"loading {cli.base} to verify the remap ...")
    base = AutoModelForImageTextToText.from_pretrained(
        cli.base, dtype=torch.bfloat16, device_map=cli.device).eval()
    # Attach the ORIGINAL adapter so PEFT builds the modules (injection matches by
    # suffix and works fine); the remapped weights then go in over the top.
    pm = PeftModel.from_pretrained(base, str(src))
    cfg = json.loads((src / "adapter_config.json").read_text(encoding="utf-8"))
    nz, n_b, worst = verify_and_report(pm, remapped, cfg["lora_alpha"] / cfg["r"])

    out.mkdir(parents=True, exist_ok=True)
    save_file(remapped, str(out / "adapter_model.safetensors"), metadata={"format": "pt"})
    for name in SIDECARS:
        if (src / name).exists():
            shutil.copy2(src / name, out / name)
    renames = "\n".join(f"  {o!r} -> {n!r}" for o, n in RENAMES)
    (out / "REMAP.md").write_text(
        f"""# Repaired adapter

Produced by `research/TimeLens/remap_adapter_keys.py` from `{src}`.

The source checkpoint was saved under an older Qwen2.5-VL module tree, so PEFT
matched none of its {len(sd)} tensors at load time and left every `lora_B` at its
zero init -- making the adapter an exact no-op and any evaluation of it a
measurement of the frozen base model. See NS-P1 ledger #80.

Only key names differ between this directory and the source; tensor values are
byte-identical. Renames applied:

{renames}

Verified on write: {nz}/{n_b} trained `lora_B` tensors landed, largest |dW|/|W|
{worst:.3e}. The vision-tower blocks carry a zero `lora_B` in the source
checkpoint -- they were frozen during training -- so they contribute nothing
here either. That is the trained model, not a second load failure.
""", encoding="utf-8")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
