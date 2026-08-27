"""Are two prediction files the same model output, byte for byte?

The gate for any 8B speed work: an optimisation may only be accepted if it leaves the
predictions unchanged. This compares a REFERENCE run against a CANDIDATE run on the keys
they share and reports exact-match rates, failing loudly on any divergence.

  env -u VIRTUAL_ENV PYTHONPATH=. .venv/bin/python verify_preds_identical.py \\
    --ref 'logs/scale8b_qvh_short/preds_s*of6.jsonl' \\
    --cand 'logs/scale8b_qvh_fa2/preds_s*of6.jsonl'

Exit status is the verdict: 0 = identical on every shared key, 1 = divergence, 2 = the
comparison itself was not valid (no overlap, or one side empty).

WHY ANSWER STRINGS ARE THE PRIMARY TEST. The `answers` field is the model's literal
decoded text. The `timestamps` field is a *derived* value, and harnesses disagree about
whether to store round(parse(answers)) or full precision -- a difference that made one
comparison in this project read 1.22% agreement when the underlying decode was in fact
100% identical. Answer-string equality is convention-free and cannot be faked by a
storage choice, so it decides the verdict. Timestamp agreement is reported alongside,
both raw and rounded, purely as a cross-check.

WHAT COUNTS AS UNCHANGED. Exact string equality, not approximate. Attention kernels are
mathematically equivalent but not bitwise identical -- they reduce in a different order --
so a kernel swap CAN flip a greedily-decoded token near a tie. That is precisely what
this script exists to detect rather than assume, and a single flipped clip is a real
finding to report, not noise to round away.
"""
from __future__ import annotations

import argparse
import glob
import json
import sys


def load(patterns: str) -> dict[str, dict]:
    out: dict[str, dict] = {}
    paths = sorted(p for pat in patterns.split(",") for p in glob.glob(pat.strip()))
    if not paths:
        print(f"ABORT: no files matched {patterns!r}")
        sys.exit(2)
    for path in paths:
        with open(path, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if line:
                    out.update(json.loads(line))
    print(f"  loaded {len(out):5d} records from {len(paths)} file(s): {patterns}")
    return out


def span(rec: dict):
    ts = rec.get("timestamps") or []
    return tuple(float(x) for x in ts[0]) if ts else None


def rnd(s):
    return None if s is None else (float(round(s[0])), float(round(s[1])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True, help="glob(s) for the reference run")
    ap.add_argument("--cand", required=True, help="glob(s) for the candidate run")
    ap.add_argument("--show", type=int, default=5, help="how many divergences to print")
    cli = ap.parse_args()

    R, C = load(cli.ref), load(cli.cand)
    keys = sorted(set(R) & set(C))
    print(f"  shared keys: {len(keys)}  (ref-only {len(set(R)-set(C))}, cand-only {len(set(C)-set(R))})")
    if not keys:
        print("ABORT: no shared keys -- the two runs are not comparable")
        sys.exit(2)

    ans = [k for k in keys if (R[k].get("answers") or "") == (C[k].get("answers") or "")]
    sp = [k for k in keys if span(R[k]) == span(C[k])]
    spr = [k for k in keys if rnd(span(R[k])) == rnd(span(C[k]))]
    n = len(keys)
    print(f"  answers  identical : {len(ans):5d}/{n} = {100*len(ans)/n:6.2f}%   <- the verdict")
    print(f"  spans    identical : {len(sp):5d}/{n} = {100*len(sp)/n:6.2f}%")
    print(f"  spans    identical after rounding : {len(spr):5d}/{n} = {100*len(spr)/n:6.2f}%")

    bad = [k for k in keys if k not in set(ans)]
    for k in bad[: cli.show]:
        print(f"  DIVERGENCE {k[:70]}")
        print(f"     ref : {(R[k].get('answers') or '')[:90]!r}")
        print(f"     cand: {(C[k].get('answers') or '')[:90]!r}")

    if bad:
        print(f"\nFAIL: {len(bad)}/{n} record(s) diverge. The optimisation changes model output.")
        sys.exit(1)
    print(f"\nPASS: all {n} shared records are byte-identical.")
    if len(keys) < len(R):
        print(f"  NOTE: the candidate covers {len(keys)} of the reference's {len(R)} records, "
              f"so this is a partial verification -- it proves nothing about the other "
              f"{len(R)-len(keys)}.")


if __name__ == "__main__":
    main()
