"""Score Study AE (METHODS-LOG 2026-11-16): CaliFrame (M2) against GMM-EVA's selection at 128 frames with
Gemma-4-31B on CG-Bench's full split without the calibration videos. Paired differences in plain accuracy,
videos resampled, 5,000 draws, seed 0.

    AE1 (primary)  M2 minus GMM-EVA's selection at 128 frames over 11,256 questions: the mini split (Study AD)
                   and the new questions, in one stratum with videos resampled, because the two parts share videos
    descriptive    the new questions alone; the same at 64 frames on the new questions (Study Z's M2 and
                   AE's GMM-EVA run); the change in the lead from 64 to 128 frames there, paired per question

    .venv/bin/python nsp5_ae_score.py          # from research/TimeLens
"""
import numpy as np

from nsp5_compare import video_of
from nsp5_pooled import load, path
from nsp5_wave1_score import show

MINI, FRESH = "cgbench", "cgbench_fresh"
SIZE = {MINI: 3000, FRESH: 8256}
M2_128, GMM_128 = "k_n128_q8_kelly_eps0_gemma4_31b", "k_n128_f2c_topk_nr_gemma4_31b"
M2_64, GMM_64 = "k_q8_kelly_eps0_gemma4_31b", "k_f2c_topk_nr_gemma4_31b"
DRAWS, SEED = 5000, 0


def check_complete() -> bool:
    """Every arm of the tests holds every question of its parts."""
    ok = True
    for arm, parts in ((M2_128, (MINI, FRESH)), (GMM_128, (MINI, FRESH)), (M2_64, (FRESH,)), (GMM_64, (FRESH,))):
        for b in parts:
            n = len(load(path(arm, b)))
            ok &= n == SIZE[b]
            print(f"  {arm:<36} {b:<14} {n:5d} of {SIZE[b]:5d} {'complete' if n == SIZE[b] else 'INCOMPLETE'}")
    return ok


def video_boot(d: np.ndarray, keys: list) -> tuple:
    """95% interval of the mean of d with videos resampled in one stratum."""
    _, inv = np.unique([video_of(k) for k in keys], return_inverse=True)
    sums, cnts = np.bincount(inv, weights=d), np.bincount(inv).astype(float)
    idx = np.random.default_rng(SEED).integers(0, len(sums), size=(DRAWS, len(sums)))
    lo, hi = np.percentile(sums[idx].sum(1) / cnts[idx].sum(1), [2.5, 97.5])
    return lo, hi, len(sums)


def ae1() -> None:
    """M2 minus GMM-EVA's selection at 128 frames, the mini split and the new questions in one stratum."""
    a = {**load(path(M2_128, MINI)), **load(path(M2_128, FRESH))}
    r = {**load(path(GMM_128, MINI)), **load(path(GMM_128, FRESH))}
    keys = sorted(set(a) & set(r))
    d = np.array([a[k] - r[k] for k in keys])
    lo, hi, n_vid = video_boot(d, keys)
    acc_a = 100 * np.mean([a[k] for k in keys])
    acc_r = 100 * np.mean([r[k] for k in keys])
    print(f"AE1: {M2_128} minus {GMM_128}: {acc_a:.2f} vs {acc_r:.2f}, {100 * d.mean():+.2f} "
          f"[{100 * lo:+.2f}, {100 * hi:+.2f}] (n={len(keys)}, {n_vid} videos)")


def lead_change() -> None:
    """(M2 minus GMM-EVA's selection) at 128 frames minus the same at 64, per question, on the new questions."""
    arms = [load(path(x, FRESH)) for x in (M2_128, GMM_128, M2_64, GMM_64)]
    keys = sorted(set.intersection(*(set(x) for x in arms)))
    d = np.array([(arms[0][k] - arms[1][k]) - (arms[2][k] - arms[3][k]) for k in keys])
    lo, hi, _ = video_boot(d, keys)
    print(f"lead at 128 minus lead at 64 frames, new questions: {100 * d.mean():+.2f} "
          f"[{100 * lo:+.2f}, {100 * hi:+.2f}] (n={len(keys)})")


def main() -> None:
    print("== completeness")
    if not check_complete():
        print("(scoring the questions present; the registered test needs every arm complete)")
    print("== AE1 (primary), 128 frames, Gemma-4-31B, CG-Bench mini and new questions in one stratum")
    ae1()
    print("== descriptive: the new questions alone, 128 frames")
    show("new questions", M2_128, GMM_128, (FRESH,))
    print("== descriptive: the new questions at 64 frames")
    show("64 frames", M2_64, GMM_64, (FRESH,))
    print("== descriptive: the change in the lead from 64 to 128 frames on the new questions")
    lead_change()


if __name__ == "__main__":
    main()
