# Copyright (c) 2025 Jun Zhang. Licensed under the BSD-3-Clause License.

import copy
import os

from qwen_vl_utils import process_vision_info
from torch.utils.data import Dataset

GROUNDER_PROMPT = (
    "Please find the visual event described by the sentence '{}', determining its starting and ending times. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)

# NS-P1 length-prior probe (env TIMELENS_PROMPT_VARIANT=short). A FAIR, query-agnostic clause that
# only licenses short/precise spans in general -- it does NOT reveal this query's GT length. Tests
# whether the model's ~3-4s output-length floor (emits <2s only ~4.6% of the time vs 17.5% of GT;
# METHODS-LOG mechanism) is a shallow decoding prior (promptable) or baked into the weights.
GROUNDER_PROMPT_SHORT = (
    "Please find the visual event described by the sentence '{}', determining its precise starting and ending times. "
    "Events vary widely in length; some last only one or two seconds. "
    "Report tight boundaries that match the event's true duration -- do not widen a brief event to a typical interval. "
    "The format should be: 'The event happens in <start time> - <end time> seconds'."
)

# prompt for TimeLens-7B (based on Qwen2.5-VL) with interleaved textual timestamps
_TS_PREFIX = (
    "You are given a video with multiple frames. "
    "The numbers before each video frame indicate its sampling timestamp (in seconds). "
)
GROUNDER_PROMPT_TEXT_TIMESTAMP = _TS_PREFIX + GROUNDER_PROMPT
GROUNDER_PROMPT_TEXT_TIMESTAMP_SHORT = _TS_PREFIX + GROUNDER_PROMPT_SHORT

# NS-P1 loss-1 Stage 1 (reparam-SFT): env REPARAM_CENTER=1 -> ask for the center-first
# (center, duration) reparameterization, matching training/data/grounding.py's target. Must be
# paired with eval_lora.py's center parser (which reconstructs the [c-w/2, c+w/2] span).
GROUNDER_PROMPT_CENTER = (
    "Please find the visual event described by the sentence '{}', determining its center time and duration. "
    "The format should be: 'The event is centered at <center time> seconds and lasts <duration> seconds'."
)
GROUNDER_PROMPT_TEXT_TIMESTAMP_CENTER = _TS_PREFIX + GROUNDER_PROMPT_CENTER


class GroundingDataset(Dataset):
    def __init__(self, annos, processor, args):
        super().__init__()
        self.annos = annos
        self.processor = processor
        self.args = args
        short = os.environ.get("TIMELENS_PROMPT_VARIANT", "").lower() == "short"
        reparam_center = os.environ.get("REPARAM_CENTER", "0").lower() not in ("0", "", "false", "no")
        is_7b = "timelens-7b" in args.model_path.lower()
        match_train = os.environ.get("ECS_MATCH_TRAIN_PROMPT", "0").lower() not in ("0", "", "false", "no")
        if reparam_center:
            # NS-P1 loss-1 Stage 1: center-first (center,duration) prompt, matched to the SFT target
            self.prompt = GROUNDER_PROMPT_TEXT_TIMESTAMP_CENTER if is_7b else GROUNDER_PROMPT_CENTER
            print("REPARAM_CENTER=1: using the center-first (center,duration) grounding prompt (NS-P1 loss-1)")
        elif match_train:
            # NS-P1 EC-Sharp de-confound: eval with the BARE prompt (no _TS_PREFIX) that training used
            # (training/data/grounding.py GROUNDING_PROMPT), so the head's last-prompt-token query `hq`
            # is drawn from the SAME distribution it was trained on.
            self.prompt = GROUNDER_PROMPT_SHORT if short else GROUNDER_PROMPT
            print("ECS_MATCH_TRAIN_PROMPT=1: eval uses the bare (no-TS-prefix) training prompt")
        elif is_7b:
            # prompt for TimeLens-7B (based on Qwen2.5-VL) with interleaved textual timestamps
            self.prompt = GROUNDER_PROMPT_TEXT_TIMESTAMP_SHORT if short else GROUNDER_PROMPT_TEXT_TIMESTAMP
        else:
            self.prompt = GROUNDER_PROMPT_SHORT if short else GROUNDER_PROMPT
        if short and not reparam_center:
            print("TIMELENS_PROMPT_VARIANT=short: using the length-prior-probe prompt (NS-P1)")
        # NS-P1 A1 oracle length-hint probe (issue #80 / R2-Q5): append a per-query hint naming the
        # GT length bin. Requires annos that carry the GT "span" (eval_bench.py annos do).
        self.oracle_len_hint = os.environ.get(
            "TIMELENS_ORACLE_LEN_HINT", "0").lower() not in ("0", "", "false", "no")
        if self.oracle_len_hint:
            print("TIMELENS_ORACLE_LEN_HINT=1: appending the per-query GT-length-bin hint (NS-P1 A1)")

    def __len__(self):
        return len(self.annos)

    def __getitem__(self, index):
        anno = copy.deepcopy(self.annos[index])

        video_path = anno["video_path"]
        query = anno["query"]

        if "qwen3" in self.args.model_path.lower() or "timelens-8b" in self.args.model_path.lower():
            # for TimeLens-8B(based on Qwen3-VL) and Qwen3-VL models
            downsample_rate = 32
        elif "qwen2" in self.args.model_path.lower() or "timelens-7b" in self.args.model_path.lower() or "videochat-r1" in self.args.model_path.lower():
            # for TimeLens-7B (based on Qwen2.5-VL), Qwen2.5-VL, and VideoChat-R1 (Qwen2.5-VL arch) models
            downsample_rate = 28
        else:
            raise NotImplementedError(
                f"Model {self.args.model_path} not supported yet."
            )

        video_content = {
            "type": "video",
            "video": video_path,
            "min_pixels": self.args.min_tokens * downsample_rate * downsample_rate,
            "total_pixels": self.args.total_tokens * downsample_rate * downsample_rate,
            "fps": self.args.fps,
        }
        # NS-P1 temporal-zoom: if the anno carries a [video_start, video_end] window, qwen_vl_utils
        # samples frames ONLY within it (cropped decode) — used by eval_zoom.py. Timestamps the model
        # sees/emits stay ABSOLUTE (frame_idx/fps). Absent -> full-video decode, unchanged.
        if anno.get("video_start") is not None:
            video_content["video_start"] = float(anno["video_start"])
        if anno.get("video_end") is not None:
            video_content["video_end"] = float(anno["video_end"])
        text_prompt = self.prompt.format(query)
        # NS-P1 A1 oracle length-hint probe: name this query's GT length bin (strata as in the
        # paper: [0,2)/[2,5)/[5,10)/[10,30)/[30,inf)). Tests whether the length-compression prior
        # is conditionable by an oracle signal short of the answer itself.
        if self.oracle_len_hint and anno.get("span"):
            sp = anno["span"][0] if isinstance(anno["span"][0], (list, tuple)) else anno["span"]
            lg = float(sp[1]) - float(sp[0])
            if lg < 2:
                bin_txt = "under 2 seconds"
            elif lg < 5:
                bin_txt = "between 2 and 5 seconds"
            elif lg < 10:
                bin_txt = "between 5 and 10 seconds"
            elif lg < 30:
                bin_txt = "between 10 and 30 seconds"
            else:
                bin_txt = "more than 30 seconds"
            text_prompt += f" Note: the target event lasts {bin_txt}."
        messages = [
            {
                "role": "user",
                "content": [
                    video_content,
                    {"type": "text", "text": text_prompt},
                ],
            }
        ]

        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        if "timelens-7b" in self.args.model_path.lower():
            # for TimeLens-7B (based on Qwen2.5-VL) with interleaved textual timestamps
            images, videos = process_vision_info(messages, return_video_metadata=True)
            inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
            )
        elif (
            "qwen3" in self.args.model_path.lower()
            or "timelens-8b" in self.args.model_path.lower()
        ):
            # for TimeLens-8B(based on Qwen3-VL) and Qwen3-VL models
            images, videos, video_kwargs = process_vision_info(
                messages,
                image_patch_size=16,
                return_video_kwargs=True,
                return_video_metadata=True,
            )
            videos, video_metadatas = zip(*videos)
            videos, video_metadatas = list(videos), list(video_metadatas)
            inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                video_metadata=video_metadatas,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
        elif "qwen2" in self.args.model_path.lower() or "videochat-r1" in self.args.model_path.lower():
            # for Qwen2.5-VL-architecture models (incl. VideoChat-R1)
            images, videos, video_kwargs = process_vision_info(
                messages, return_video_kwargs=True
            )
            inputs = self.processor(
                text=[text],
                images=images,
                videos=videos,
                padding=True,
                return_tensors="pt",
                **video_kwargs,
            )
        else:
            raise NotImplementedError(
                f"Model {self.args.model_path} not supported yet."
            )

        return {"inputs": inputs, "anno": anno}
