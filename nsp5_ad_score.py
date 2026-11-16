"""Score Study AD (METHODS-LOG 2026-11-15, registered d4086d43): CaliFrame (M2) against GMM-EVA's selection at
128 frames with Gemma-4-31B, Study K's settings. Paired differences in plain accuracy, videos resampled within
each benchmark, 5,000 draws, seed 0 (nsp5_pooled.pooled).

    AD1 (primary)  M2 minus GMM-EVA's selection at 128 frames, LVBench, Video-MME and CG-Bench mini pooled
    descriptive    per benchmark (printed under AD1); the change in the lead from 64 to 128 frames on LVBench
                   and Video-MME, paired per question as in Study S's S4

    .venv/bin/python nsp5_ad_score.py          # from research/TimeLens
"""
import numpy as np

from nsp5_compare import video_of
from nsp5_pooled import SIZE, load, path
from nsp5_wave1_score import show

LVC = ("lvbench", "videomme", "cgbench")
LV = ("lvbench", "videomme")
M2_128, GMM_128 = "k_n128_q8_kelly_eps0_gemma4_31b", "k_n128_f2c_topk_nr_gemma4_31b"
M2_64, GMM_64 = "k_q8_kelly_eps0_gemma4_31b", "k_f2c_topk_nr_gemma4_31b"
DRAWS, SEED = 5000, 0


def check_complete() -> None:
    """Every arm of the test holds every question of its benchmarks."""
    for arm, benches in ((M2_128, LVC), (GMM_128, LVC), (M2_64, LV), (GMM_64, LV)):
        for b in benches:
            n = len(load(path(arm, b)))
            status = "complete" if n == SIZE[b] else "INCOMPLETE"
            print(f"  {arm:<36} {b:<9} {n:5d} of {SIZE[b]:5d} {status}")


def lead_change(benches: tuple) -> None:
    """(M2 minus GMM-EVA's selection) at 128 frames minus the same at 64, per question; videos resampled within
    each benchmark, pooled over questions as nsp5_pooled.pooled pools them."""
    strata, diffs = [], []
    for b in benches:
        arms = [load(path(a, b)) for a in (M2_128, GMM_128, M2_64, GMM_64)]
        keys = sorted(set.intersection(*(set(a) for a in arms)))
        d = np.array([(arms[0][k] - arms[1][k]) - (arms[2][k] - arms[3][k]) for k in keys])
        _, inv = np.unique([video_of(k) for k in keys], return_inverse=True)
        strata.append((np.bincount(inv, weights=d), np.bincount(inv).astype(float)))
        diffs.append(d)
        print(f"    {b:<14} n={len(keys):5d}  {100 * d.mean():+6.2f}")
    rng = np.random.default_rng(SEED)
    means = []
    for _ in range(DRAWS):
        s = c = 0.0
        for sums, cnts in strata:
            idx = rng.integers(0, len(sums), len(sums))
            s += sums[idx].sum()
            c += cnts[idx].sum()
        means.append(s / c)
    d = np.concatenate(diffs)
    lo, hi = np.percentile(means, [2.5, 97.5])
    print(f"lead at 128 minus lead at 64 frames, pooled: {100 * d.mean():+.2f} [{100 * lo:+.2f}, {100 * hi:+.2f}] "
          f"(n={len(d)})")


def main() -> None:
    print("== completeness")
    check_complete()
    print("== AD1 (primary), 128 frames, Gemma-4-31B")
    show("AD1", M2_128, GMM_128, LVC)
    print("== descriptive: the same at 64 frames (Study AC), LVBench and Video-MME")
    show("64 frames", M2_64, GMM_64, LV)
    print("== descriptive: the change in the lead from 64 to 128 frames, LVBench and Video-MME")
    lead_change(LV)


if __name__ == "__main__":
    main()
