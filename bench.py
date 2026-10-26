"""One question schema for every long-video benchmark NS-P5 evaluates on.

Each loader maps a benchmark's own annotation format onto `QItem`, so the rest of the
pipeline -- 1 fps decode, prior scoring, curves, the answering harness -- never needs
to know which benchmark it is running. Formats were read from the files on disk
(2026-10-26), not assumed:

  lvbench   parquet; question text carries "(A) ..." options; time_reference "MM:SS-MM:SS"
  videomme  formatted_dataset.json; candidates "A. ..."; answer letter; no spans
  lvb       LongVideoBench lvb_val.json; candidates list + correct_choice index; no
            usable spans (`position` is not in seconds)
  mlvu      MLVU dev json/<task>.json, multiple-choice tasks 1-7; answer given as text
  cgbench   CG-Bench cgbench(_mini).json; 5-8 choices; clue_intervals in seconds

Paths are exact, from the approved listing; nothing here lists a directory.
"""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field

LETTERS = "ABCDEFGH"
GPFS = "/gpfs/public/datasets"
LVB_ROOT = ("/home/mobeen/codes/active_repos/NeedleScan-vtg-landscape/research/TimeLens/"
            "data/lvbench")
MLVU_MC_TASKS = ("1_plotQA", "2_needle", "3_ego", "4_count", "5_order", "6_anomaly_reco",
                 "7_topic_reasoning")


@dataclass(frozen=True)
class QItem:
    bench: str
    key: str                 # unique across benchmarks: "<bench>:<id>"
    vid: str                 # video id, the frame-cache file name
    video_path: str
    stem: str                # the question without its options
    options: tuple[str, ...]
    gold: str                # "A".."H"
    duration: float | None   # seconds, when the annotation gives it
    spans: tuple[tuple[float, float], ...] = ()   # evidence, seconds; () if none
    meta: dict = field(default_factory=dict, compare=False, hash=False)

    def prompt_question(self) -> str:
        """Stem + lettered options, in LVBench's own "(A) ..." style for every benchmark."""
        return self.stem + "".join(f"\n({LETTERS[i]}) {o}" for i, o in enumerate(self.options))


def _strip_letter(opt: str) -> str:
    return re.sub(r"^\s*\(?[A-H][.)]\s*", "", opt).strip()


def load_lvbench() -> list[QItem]:
    from eval_lvbench_foveate import load_questions
    items, _ = load_questions(f"{LVB_ROOT}/data/train-00000-of-00001.parquet",
                              f"{LVB_ROOT}/video_info.meta.jsonl", f"{LVB_ROOT}/videos")
    out = []
    for it in items:
        stem = it["question"].split("\n(A)")[0].strip()
        opts = tuple(o.strip() for _, o in re.findall(
            r"\(([A-H])\)\s*(.*?)(?=\n\([A-H]\)|$)", it["question"], flags=re.S))
        out.append(QItem("lvbench", f"lvbench:{it['key']}>>>{it['uid']}", it["key"],
                         it["video_path"], stem, opts, it["gold"], it["duration"],
                         ((it["span"][0], it["span"][1]),)))
    return out


def load_videomme() -> list[QItem]:
    with open(f"{GPFS}/Video-MME/formatted_dataset.json", encoding="utf-8") as fh:
        rows = json.load(fh)
    return [QItem("videomme", f"videomme:{r['id']}", os.path.splitext(r["video"])[0],
                  f"{GPFS}/Video-MME/data/{r['video']}", r["question"].strip(),
                  tuple(_strip_letter(c) for c in r["candidates"]), r["answer"].strip(), None,
                  (), {"split": r["question_type"], "task": r["task_type"],
                       "domain": r["domain"]})
            for r in rows]


def load_lvb() -> list[QItem]:
    with open(f"{GPFS}/LongVideoBench/lvb_val.json", encoding="utf-8") as fh:
        rows = json.load(fh)
    return [QItem("lvb", f"lvb:{r['id']}", os.path.splitext(r["video_path"])[0],
                  f"{GPFS}/LongVideoBench/videos/{r['video_path']}", r["question"].strip(),
                  tuple(str(c).strip() for c in r["candidates"]), LETTERS[r["correct_choice"]],
                  float(r["duration"]), (),
                  {"duration_group": r["duration_group"], "category": r["question_category"],
                   "level": r["level"]})
            for r in rows]


def load_mlvu() -> list[QItem]:
    out = []
    for task in MLVU_MC_TASKS:
        with open(f"{GPFS}/MLVU/MLVU/json/{task}.json", encoding="utf-8") as fh:
            rows = json.load(fh)
        for i, r in enumerate(rows):
            opts = tuple(str(c).strip() for c in r["candidates"])
            out.append(QItem("mlvu", f"mlvu:{task}:{i}", f"{task}__{os.path.splitext(r['video'])[0]}",
                             f"{GPFS}/MLVU/MLVU/video/{task}/{r['video']}", r["question"].strip(),
                             opts, LETTERS[opts.index(str(r["answer"]).strip())],
                             float(r["duration"]), (), {"task": task}))
    return out


def load_cgbench(mini: bool = True) -> list[QItem]:
    name = "cgbench_mini.json" if mini else "cgbench.json"
    with open(f"{GPFS}/CG-Bench/{name}", encoding="utf-8") as fh:
        rows = json.load(fh)
    return [QItem("cgbench", f"cgbench:{r['qid']}", r["video_uid"],
                  f"{GPFS}/CG-Bench/videos/{r['video_uid']}.mp4", r["question"].strip(),
                  tuple(str(c).strip() for c in r["choices"]), r["right_answer"].strip(),
                  float(r["duration"]), tuple((float(a), float(b)) for a, b in r["clue_intervals"]),
                  {"category": r["sub_category"], "domain": r["domain"]})
            for r in rows]


LOADERS = {"lvbench": load_lvbench, "videomme": load_videomme, "lvb": load_lvb,
           "mlvu": load_mlvu, "cgbench": load_cgbench}


def load(bench: str) -> list[QItem]:
    items = LOADERS[bench]()
    keys = [it.key for it in items]
    assert len(set(keys)) == len(keys), f"{bench}: duplicate question keys"
    for it in items:
        assert 2 <= len(it.options) <= len(LETTERS), (it.key, it.options)
        assert it.gold in LETTERS[:len(it.options)], (it.key, it.gold, len(it.options))
    return items


if __name__ == "__main__":
    import collections
    for b in LOADERS:
        items = load(b)
        vids = {it.vid: it.video_path for it in items}
        missing = sum(not os.path.exists(p) for p in vids.values())
        opts = collections.Counter(len(it.options) for it in items)
        spans = sum(bool(it.spans) for it in items)
        print(f"{b:<9} {len(items):>6} questions  {len(vids):>5} videos ({missing} missing)  "
              f"options {dict(sorted(opts.items()))}  with spans {spans}")
    print("\nexample prompt (cgbench):\n" + load("cgbench")[0].prompt_question())
