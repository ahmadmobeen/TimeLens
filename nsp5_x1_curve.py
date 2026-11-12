"""Study X1, CPU part (NeedleScan METHODS-LOG 2026-11-12, wave 2): accuracy against compute per
question as the questions asked about each video go from 1 to 15, on LVBench.

Compute per question, in seconds of one B200 (nsp5_x1_cost.sh measured every part):

    index of the video / questions per video  +  scoring  +  placing and decoding  +  answering

- index: the mean LVBench video's seconds over the scorer's frames per second (x1_cost_index.json);
  uniform has none. AKS as released keeps no index: it runs BLIP's vision model over every second
  for every question, so its cost does not fall with more questions.
- scoring: the mean per question (x1_cost_score.json); placing, decoding and answering: the means of
  t_read and t_answer in the --timing rows (x1cost_<arm>.jsonl).
- accuracy: plain LVBench accuracy of the full arm in the paper's selector table.

    .venv/bin/python nsp5_x1_curve.py      -> prints the table, writes logs/nsp5/x1_cost_curve.json
"""
from __future__ import annotations

import json
import statistics

L = "logs/nsp5"
FRAMES = "data/t1/frames/lvbench"
# selector: (full LVBench arm, --timing arm, scorer index or None, scoring key or None)
SELECTORS = {
    "Uniform": ("e3_uniform", "uniform", None, None),
    "BOLT (as released)": ("bolt_own", "bolt_own", "clip_l14", "bolt_clip_question"),
    "AKS (as released)": ("aks_own", "aks_own", "blip_itm_l", "aks_blip_matching_question"),
    "GMM-EVA selection": ("f2c_topk_nr", "f2c_topk_nr", "qwen3vl_emb_8b", "gmm_eva_selection_8b_concat"),
    "CaliFrame": ("q8_kelly_eps0", "q8_kelly_eps0", "qwen3vl_emb_8b", "califrame_8b_per_option"),
}
PER_VIDEO = (1, 2, 5, 10, 15)


def rows(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return [json.loads(line) for line in fh]


def mean_video_seconds() -> float:
    from t1_data import lvbench_items
    vids = sorted({it["key"] for it in lvbench_items()})
    lens = []
    for v in vids:
        with open(f"{FRAMES}/{v}.json", encoding="utf-8") as fh:
            lens.append(len(json.load(fh)["t"]))
    return statistics.fmean(lens)


def main() -> None:
    with open(f"{L}/x1_cost_index.json", encoding="utf-8") as fh:
        index = json.load(fh)
    with open(f"{L}/x1_cost_score.json", encoding="utf-8") as fh:
        score = json.load(fh)
    sec = mean_video_seconds()
    report = {"mean_video_seconds": sec, "gpu": index.get("gpu"), "selectors": {}}
    print(f"mean LVBench video: {sec / 60:.1f} min; seconds of one B200 per question")
    print(f"{'selector':22s}{'acc':>7s}" + "".join(f"{f'q={q}':>9s}" for q in PER_VIDEO)
          + f"{'index/video':>13s}{'score':>8s}{'read':>7s}{'answer':>8s}")
    for name, (full, timing, scorer, skey) in SELECTORS.items():
        acc = 100 * statistics.fmean(r["correct"] for r in rows(f"{L}/{full}.jsonl"))
        t = rows(f"{L}/x1cost_{timing}.jsonl")
        read = statistics.fmean(r["t_read"] for r in t)
        answer = statistics.fmean(r["t_answer"] for r in t)
        scoring = score[skey]["mean_s"] if skey else 0.0
        video = sec / index[scorer]["frames_per_s"] if scorer else 0.0
        fixed_per_question = name.startswith("AKS")              # no index kept between questions
        cost = {q: (video if fixed_per_question else video / q) + scoring + read + answer
                for q in range(1, 16)}
        report["selectors"][name] = {"accuracy": acc, "index_s_per_video": video,
                                     "scoring_s": scoring, "read_s": read, "answer_s": answer,
                                     "index_per_question": not fixed_per_question,
                                     "seconds_per_question": cost}
        print(f"{name:22s}{acc:7.2f}" + "".join(f"{cost[q]:9.2f}" for q in PER_VIDEO)
              + f"{video:13.1f}{scoring:8.3f}{read:7.2f}{answer:8.2f}")
    with open(f"{L}/x1_cost_curve.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print(f"wrote {L}/x1_cost_curve.json")


if __name__ == "__main__":
    main()
