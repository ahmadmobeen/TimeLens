"""Model/config/processor loader for TimeLens (Qwen3-VL-8B and Qwen2.5-VL-7B).

NS-P1: the D-lever finetune targets the reproduced TimeLens-7B (Qwen2.5-VL). The
underlying Auto* classes load Qwen2.5-VL transparently, so the allowlist below was
widened to accept the 7B (qwen2.5 / timelens-7b) in addition to the original 8B path.
"""

from transformers import AutoConfig, AutoModelForImageTextToText, AutoProcessor

# 2026-08-19 (issue #90 Exp 2): widened again, for the same reason and on the same kind
# of evidence as the 7B widening recorded in the docstring above. VideoChat-R1_7B is
# Qwen2_5_VLForConditionalGeneration, 28 layers, visual.merger present, and the 7B
# recipe's freezes leave exactly the trainable set they leave on TimeLens-7B -- checked
# on a meta device, the same audit that cleared the 8B. This allowlist only decides
# whether to hand back the Auto* classes, and every branch hands back the same three,
# so the guard was rejecting a NAME rather than an architecture.
_SUPPORTED_MARKERS = ("qwen3", "timelens-8b", "qwen2.5", "qwen2_5", "timelens-7b",
                      "videochat-r1")


def _validate_model_path(model_path: str) -> None:
    model_path_lower = model_path.lower()
    if not any(marker in model_path_lower for marker in _SUPPORTED_MARKERS):
        raise ValueError(
            "Only Qwen3-VL/TimeLens-8B or Qwen2.5-VL/TimeLens-7B is supported, "
            f"got model_path={model_path!r}."
        )


def get_model_class(model_path: str):
    _validate_model_path(model_path)
    return AutoModelForImageTextToText


def get_config_class(model_path: str):
    _validate_model_path(model_path)
    return AutoConfig


def get_processor_class(model_path: str):
    _validate_model_path(model_path)
    return AutoProcessor
