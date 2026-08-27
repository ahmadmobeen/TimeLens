"""Make TimeLens-8B's vision patch embedding stop taking 99.9% of the forward pass.

MEASURED CAUSE. `Qwen3VLVisionPatchEmbed` (transformers 4.57.1) embeds patches with
`nn.Conv3d(3, 1152, kernel_size=(2,16,16), stride=(2,16,16), bias=True)`. In bfloat16 on
a B200 that convolution hits a catastrophic kernel: 6,534 ms for 1,456 patches, which is
99.9% of the whole vision forward. Everything else is noise by comparison -- all 27
transformer blocks together add ~225 ms, and one block on a real sequence is 0.55 ms.

The give-away is that the same convolution is FAST in fp32:

    F.conv3d  bf16   6533.90 ms      <- what ships
    F.conv3d  fp32      0.29 ms      22,000x faster, same operation
    F.linear  bf16      0.0255 ms    240,000x faster

So this is bf16 kernel selection for this particular shape, not the cost of the
convolution. `torch.backends.cudnn.benchmark = True` does not rescue it (0.95x).
TimeLens-7B is unaffected: its patch size is 14, not 16, and its tower runs the whole
150 s clip in 2.9 s. A one-step change in patch size crosses a kernel-selection cliff.

WHY A MATMUL IS THE RIGHT REPLACEMENT. When a convolution's kernel equals its stride,
each output element depends on exactly one non-overlapping input block, and here the
input is pre-arranged so that one block is one row: `hidden_states` arrives already
shaped (num_patches, in_channels*temporal_patch*patch*patch). Flattening the weight to
(embed_dim, in_channels*temporal_patch*patch*patch) makes the convolution literally a
matrix multiply plus bias -- identical arithmetic, no approximation, no reordering of
which inputs meet which weights.

HONEST ACCOUNT OF NUMERICS. This is NOT bitwise identical to the shipped bf16
convolution: max|diff| = 1.56e-2, a relative 4.8e-3, which is about one unit in the last
place of bfloat16. It cannot be made bitwise identical, and the reason matters: every
sane implementation -- bf16 addmm, bf16 linear, fp32 matmul rounded to bf16, and fp32
conv3d rounded to bf16 -- agrees with the others and disagrees with the bf16 convolution
by that same 1 ulp. The bf16 convolution is the outlier, so this path is at least as
accurate, not less. But "at least as accurate" is not "unchanged", and on greedy decoding
a 1-ulp shift can flip a token near a tie. That is an empirical question, so it is settled
empirically: see `verify_preds_identical.py` and the verification recorded alongside this
module. Do not assume; re-verify if the model, dtype, or transformers version changes.

Usage::

    import fast_patch_embed
    fast_patch_embed.patch()      # returns the number of classes patched
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def _fast_forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
    """Conv3d with kernel == stride, expressed as the matmul it already is."""
    weight = self.proj.weight
    # (embed_dim, C, T, P, P) -> (embed_dim, C*T*P*P). The weight is contiguous, so this
    # is a view; the row order matches the input's flat layout element for element.
    w2d = weight.reshape(self.embed_dim, -1)
    hidden_states = hidden_states.view(-1, w2d.shape[1]).to(dtype=weight.dtype)
    return F.linear(hidden_states, w2d, self.proj.bias)


_fast_forward._fast_patch_embed = True  # type: ignore[attr-defined]

# Only Qwen3-VL. Qwen2.5-VL (TimeLens-7B) is NOT patched on purpose: it does not have the
# problem, and its numbers are the published baseline for several tables, so there is no
# reason to perturb its arithmetic by even one ulp.
_TARGETS = (
    ("transformers.models.qwen3_vl.modeling_qwen3_vl", "Qwen3VLVisionPatchEmbed"),
    ("transformers.models.qwen3_vl_moe.modeling_qwen3_vl_moe", "Qwen3VLMoeVisionPatchEmbed"),
)


def patch(targets=_TARGETS, verbose: bool = True) -> int:
    """Rebind the listed patch-embed forwards. Idempotent; returns how many were changed."""
    n = 0
    for module_path, cls_name in targets:
        try:
            mod = __import__(module_path, fromlist=[cls_name])
            cls = getattr(mod, cls_name)
        except (ImportError, AttributeError):
            continue
        if getattr(cls.forward, "_fast_patch_embed", False):
            continue
        cls.forward = _fast_forward
        n += 1
        if verbose:
            print(f"[fast_patch_embed] {cls_name}.forward -> matmul "
                  f"(bf16 Conv3d kernel is ~240,000x slower on this shape)", flush=True)
    if verbose and n == 0:
        print("[fast_patch_embed] nothing patched (already applied, or classes absent)", flush=True)
    return n


__all__ = ["patch"]
