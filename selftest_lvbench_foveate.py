"""Self-check for eval_lvbench_foveate's data handling. Run before any E3 sweep.

No pytest exists in this venv, so this is a plain script: it exits non-zero on the
first failure and prints the population census it verified.

Three defects in LVBench's time_reference field cost real data when missed, so each
has a case here:
  * 250 references are an instant ("01:01-01:01"). Rejecting those as malformed
    silently discards 16 % of the benchmark.
  * 13 are reversed ("33:52-33:46"). They are dropped, never silently swapped.
  * 2 do not parse at all ("08:17-None").
The expected counts below are the measured truth for this release of the parquet; if
the data is ever re-pulled and they change, that is a finding to investigate, not a
number to edit.

Run:
  env -u VIRTUAL_ENV PYTHONPATH=. .venv/bin/python selftest_lvbench_foveate.py
"""
from __future__ import annotations

import sys

from eval_lvbench_foveate import load_questions, parse_letter, parse_time_reference

PARQUET = ("/home/mobeen/codes/active_repos/NeedleScan-vtg-landscape/research/TimeLens/"
           "data/lvbench/data/train-00000-of-00001.parquet")
META = ("/home/mobeen/codes/active_repos/NeedleScan-vtg-landscape/research/TimeLens/"
        "data/lvbench/video_info.meta.jsonl")
VIDEOS = ("/home/mobeen/codes/active_repos/NeedleScan-vtg-landscape/research/TimeLens/"
          "data/lvbench/videos")

EXPECTED = {"total": 1549, "unparseable_span": 2, "reversed_span": 13,
            "no_duration": 0, "no_video": 0, "span_past_end": 0,
            "scoreable": 1534, "videos": 103}

failures: list[str] = []


def check(name: str, got, want) -> None:
    if got != want:
        failures.append(f"{name}: got {got!r}, want {want!r}")


# --- time_reference parsing -------------------------------------------------
check("MM:SS span", parse_time_reference("01:30-02:00"), (90.0, 120.0))
check("HH:MM:SS span", parse_time_reference("01:00:30-01:01:00"), (3630.0, 3660.0))
check("instant kept", parse_time_reference("01:01-01:01"), (61.0, 61.0))
check("reversed parses (caller drops it)", parse_time_reference("33:52-33:46"), (2032.0, 2026.0))
check("None half", parse_time_reference("08:17-None"), None)
check("both None", parse_time_reference("None-None"), None)
check("empty", parse_time_reference(""), None)

# --- answer-letter parsing --------------------------------------------------
check("bare letter", parse_letter("B"), "B")
check("letter with stop", parse_letter("C."), "C")
check("lowercase", parse_letter("d"), "D")
check("prefixed", parse_letter("The answer is A"), "A")
check("parenthesised", parse_letter("(D)"), "D")
check("no commitment", parse_letter("I cannot tell"), None)
check("empty reply", parse_letter(""), None)

# --- population census on the real release ----------------------------------
try:
    _, census = load_questions(PARQUET, META, VIDEOS)
    for key, want in EXPECTED.items():
        check(f"census.{key}", census.get(key), want)
    print(f"census: {census}")
except FileNotFoundError as exc:
    failures.append(f"census skipped, missing input: {exc}")

if failures:
    print("\nFAILED:")
    for line in failures:
        print(f"  {line}")
    sys.exit(1)
print(f"\nOK: {len(EXPECTED) + 14} checks passed")
