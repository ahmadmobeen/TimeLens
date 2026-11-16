"""Study AI (METHODS-LOG 2026-11-16): F2C with the budget its clips leave unspent filled by full frames at the
uniform arm's times (nsp5_f2c.uniform_fill; harness --f2c_fill uniform). A question whose clips leave less than
one full frame unspent gets exactly F2C's frames and sizes, so its answer is F2C's (Study AF's f2c_full); only the
other questions run again, as the arm <stem>_f2c_fillrun with --skip_done on the list this script writes.

    .venv/bin/python nsp5_f2c_fill.py skip <bench>     logs/nsp5/f2c_fill/skip_<bench>.jsonl: the questions the fill
                                                       leaves untouched, from F2C's complete log
    .venv/bin/python nsp5_f2c_fill.py splice <bench>   the arm f2c_fill: the new answers, and F2C's for the rest;
                                                       checks that the rerun rebuilt F2C's clips and filled exactly
                                                       the questions the log predicts
"""
import json
import math
import os
import sys

import nsp5_f2c as f2c

L = "logs/nsp5"
SIZE = {"lvbench": 1534, "videomme": 2700, "lvb": 1337, "mlvu": 2174, "cgbench": 3000}
K = 64


def stem(arm: str, bench: str) -> str:
    return f"{L}/{arm}.jsonl" if bench == "lvbench" else f"{L}/x1_{bench}_{arm}.jsonl"


def rows(path: str) -> dict:
    with open(path, encoding="utf-8") as fh:
        return {r["key"]: r for r in (json.loads(line) for line in fh if line.strip())}


def fill_of(row: dict) -> int:
    """Full frames the fill adds to this F2C question, from its logged clips (nsp5_f2c.frame_plan's scales)."""
    clips = [f2c.Clip(s, s, fr, ln, b) for s, fr, ln, b in row["f2c_clips"]]
    _, scales = f2c.frame_plan(clips)
    return max(0, int(math.floor(K - sum(scales) + f2c.EPS)))


def skip(bench: str) -> None:
    base = rows(stem("f2c_full", bench))
    if len(base) != SIZE[bench]:
        raise SystemExit(f"{bench}: F2C's log holds {len(base)} of {SIZE[bench]} questions; wait for it to merge")
    keep = [k for k, r in base.items() if fill_of(r) == 0]
    os.makedirs(f"{L}/f2c_fill", exist_ok=True)
    out = f"{L}/f2c_fill/skip_{bench}.jsonl"
    with open(out, "w", encoding="utf-8") as fh:
        fh.writelines(json.dumps({"key": k}) + "\n" for k in keep)
    print(f"{bench}: {SIZE[bench] - len(keep)} questions to run with the fill, {len(keep)} keep F2C's answer -> {out}")


def splice(bench: str) -> None:
    base, new = rows(stem("f2c_full", bench)), rows(stem("f2c_fillrun", bench))
    want = {k for k, r in base.items() if fill_of(r) > 0}
    if set(new) != want:
        raise SystemExit(f"{bench}: the rerun holds {len(new)} questions, the log predicts {len(want)} "
                         f"({len(set(new) - want)} unexpected, {len(want - set(new))} missing)")
    bad = [k for k in want if new[k]["f2c_clips"] != base[k]["f2c_clips"] or new[k]["f2c_fill"] != fill_of(base[k])]
    if bad:
        raise SystemExit(f"{bench}: {len(bad)} reruns did not rebuild F2C's clips or fill, e.g. {bad[0]}")
    out = stem("f2c_fill", bench)
    with open(out, "w", encoding="utf-8") as fh:
        for k, r in base.items():
            fh.write(json.dumps(new[k] if k in new else {**r, "f2c_fill": 0}) + "\n")
    same = sum(new[k]["correct"] == base[k]["correct"] for k in want)
    print(f"{bench}: {len(base)} questions -> {out}; {len(want)} filled, {same} of them answered as F2C did")


if __name__ == "__main__":
    {"skip": skip, "splice": splice}[sys.argv[1]](sys.argv[2])
