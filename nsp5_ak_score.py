#!/usr/bin/env python3
"""Study AK (METHODS-LOG 2026-11-18, descriptive): GMM-EVA in full logged on CG-Bench's full split (AK1) and
at 32 frames on LVBench and Video-MME under three total budgets of visual tokens (AK2). Accuracy comes from
exact counts over the merged per-question files in logs/nsp5, rounded half up to one decimal as in the
paper's tables, and is compared with the submitted cells under the registered matching rule (both
benchmarks within 0.1 points).
    .venv/bin/python nsp5_ak_score.py
"""
import os
from decimal import ROUND_HALF_UP, Decimal

from nsp5_pooled import load, path

SIZE = {"lvbench": 1534, "videomme": 2700, "cgbench": 3000, "cgbench_fresh": 8256}
FULL = ("cgbench", "cgbench_fresh")              # the mini split's 3,000 questions and the 8,256 others
SUBMITTED_FULL = Decimal("45.8")
SUBMITTED_32 = {"lvbench": Decimal("53.3"), "videomme": Decimal("62.5")}
BUDGETS = (4032, 8192, 14336)                    # visual tokens; 8,192 = 32 x 256, GMM-EVA's reference


def counts(arm: str, benches: tuple) -> tuple:
    right = n = 0
    for b in benches:
        rows = load(path(arm, b))
        assert len(rows) == SIZE[b], f"{arm} on {b}: {len(rows)} of {SIZE[b]} questions"
        right += int(round(sum(rows.values())))
        n += len(rows)
    return right, n


def pct(right: int, n: int) -> Decimal:
    return (Decimal(100 * right) / Decimal(n)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)


def main() -> None:
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    r, n = counts("aa_gmmeva_full", FULL)
    rm, nm = counts("aa_gmmeva_full", ("cgbench",))
    rf, nf = counts("aa_gmmeva_full", ("cgbench_fresh",))
    print(f"AK1 GMM-EVA in full, CG-Bench full split: {pct(r, n)} ({r}/{n}; mini {pct(rm, nm)}, "
          f"other questions {pct(rf, nf)}); submitted {SUBMITTED_FULL}")
    print("AK2 GMM-EVA in full at 32 frames (submitted 53.3 / 62.5 / 59.1):")
    matches = []
    for bt in BUDGETS:
        arm = f"ak32_t{bt}_gmmeva_full"
        cells, ok = [], True
        for b in ("lvbench", "videomme"):
            rb, nb = counts(arm, (b,))
            cells.append(f"{pct(rb, nb)} ({rb}/{nb})")
            ok &= abs(pct(rb, nb) - SUBMITTED_32[b]) <= Decimal("0.1")
        rp, npool = counts(arm, ("lvbench", "videomme"))
        print(f"  {bt:>6} tokens: LVBench {cells[0]} | Video-MME {cells[1]} | pooled {pct(rp, npool)}"
              f"{'  MATCH' if ok else ''}")
        if ok:
            matches.append(bt)
    print("registered outcome:", f"budget {matches[0]} matches" if matches else
          "no budget matches; Table 3 reports the 8,192-token budget")


if __name__ == "__main__":
    main()
