"""Score Study AB (METHODS-LOG 2026-11-15, registered d28929cc): stacked top-k at the depth the
calibration split picks (m = 4), against CaliFrame (M2). Paired differences in plain accuracy, videos
resampled within each benchmark, 5,000 draws, seed 0 (nsp5_pooled.pooled).

    AB1 (primary)  q8_kelly_eps0 - t_stack4, the five benchmarks pooled
    AB2            the same on the 8,256 new CG-Bench questions
    descriptive    per benchmark; t_stack4 - q8opt_topk_nr (stacking added to top-k), five benchmarks

    .venv/bin/python nsp5_ab_score.py          # from research/TimeLens
"""
from nsp5_pooled import BENCH
from nsp5_wave1_score import show

FRESH = ("cgbench_fresh",)


def main() -> None:
    show("AB1 (primary): CaliFrame minus stacked top-k (m = 4), five benchmarks", "q8_kelly_eps0", "t_stack4", BENCH)
    show("AB2: the same on the new CG-Bench questions", "q8_kelly_eps0", "t_stack4", FRESH)
    show("descriptive: stacked top-k minus plain top-k, five benchmarks", "t_stack4", "q8opt_topk_nr", BENCH)
    show("descriptive: CaliFrame minus plain top-k, five benchmarks", "q8_kelly_eps0", "q8opt_topk_nr", BENCH)


if __name__ == "__main__":
    main()
