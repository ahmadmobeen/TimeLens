"""Score Study AF (METHODS-LOG 2026-11-16, registered a970770e): F2C reimplemented (nsp5_f2c.py) against
CaliFrame under the same image input, Qwen2.5-VL-7B. Paired differences in plain accuracy, videos resampled
within each benchmark, 5,000 draws, seed 0 (nsp5_pooled.pooled).

    AF1 (primary)  aa_cal_img minus f2c_full, the five benchmarks pooled (per benchmark printed under it)
    descriptive    F2C minus GMM-EVA's selection under image input (LVBench, Video-MME, CG-Bench);
                   the host variant minus F2C (LVBench, CG-Bench)

And Study AI (registered 8e1d202b), once nsp5_f2c_fill.py has spliced f2c_fill on every benchmark:

    AI1 (primary)  aa_cal_img minus f2c_fill (F2C with the budget its clips leave unspent filled), pooled
    descriptive    f2c_fill minus f2c_full; the share of the visual-token budget each arm spends

    .venv/bin/python nsp5_af_score.py          # from research/TimeLens
"""
import json
import os

from nsp5_pooled import BENCH, SIZE, load, path
from nsp5_wave1_score import show

CAL, F2C, GMM_SEL, HOST, FILL = "aa_cal_img", "f2c_full", "aa_gmmeva_sel_img", "f2c_host_cal", "f2c_fill"
AA_BENCH = ("lvbench", "videomme", "cgbench")
HOST_BENCH = ("lvbench", "cgbench")


def check_complete() -> bool:
    """Every arm of the tests holds every question of its benchmarks."""
    ok = True
    for arm, benches in ((CAL, BENCH), (F2C, BENCH), (GMM_SEL, AA_BENCH), (HOST, HOST_BENCH)):
        for b in benches:
            n = len(load(path(arm, b))) if os.path.exists(path(arm, b)) else 0
            ok &= n == SIZE[b]
            print(f"  {arm:<20} {b:<9} {n:5d} of {SIZE[b]:5d} {'complete' if n == SIZE[b] else 'INCOMPLETE'}")
    return ok


def main() -> None:
    print("== completeness")
    complete = check_complete()
    print("== AF1 (primary): CaliFrame minus F2C, image input for both, five benchmarks")
    show("AF1", CAL, F2C, BENCH)
    print("== descriptive: F2C minus GMM-EVA's selection, image input")
    show("F2C - GMM-EVA sel", F2C, GMM_SEL, AA_BENCH)
    if complete:
        print("== descriptive: the host variant (CaliFrame's anchors, F2C's clips and resolution) minus F2C")
        show("host - F2C", HOST, F2C, HOST_BENCH)
    else:
        print("(an arm is incomplete: the registered tests need every arm merged)")
    study_ai()


def budget_share(arm: str, bench: str) -> float:
    """Mean share of the 14,336-token budget the arm's frames hold."""
    with open(path(arm, bench), encoding="utf-8") as fh:
        rows = [json.loads(line) for line in fh if line.strip()]
    return sum(r["visual_tokens"] for r in rows) / (14336 * len(rows))


def study_ai() -> None:
    """Study AI (registered 8e1d202b): F2C with the budget its clips leave unspent filled (nsp5_f2c_fill.py)."""
    missing = [b for b in BENCH if not os.path.exists(path(FILL, b)) or len(load(path(FILL, b))) != SIZE[b]]
    if missing:
        print(f"== Study AI: f2c_fill not spliced yet on {', '.join(missing)}")
        return
    print("== AI1 (primary): CaliFrame minus F2C with the fill, image input for both, five benchmarks")
    show("AI1", CAL, FILL, BENCH)
    print("== descriptive: F2C with the fill minus F2C as written")
    show("fill - F2C", FILL, F2C, BENCH)
    print("== descriptive: share of the visual-token budget spent")
    for b in BENCH:
        print(f"    {b:<9} F2C {budget_share(F2C, b):6.1%}  with the fill {budget_share(FILL, b):6.1%}  "
              f"CaliFrame {budget_share(CAL, b):6.1%}")


if __name__ == "__main__":
    main()
