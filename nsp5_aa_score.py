"""Score Study AA (METHODS-LOG 2026-11-15, registered a8b01649): full GMM-EVA and CaliFrame's placement
inside it. Paired differences in plain accuracy, videos resampled within each benchmark, 5,000 draws,
seed 0 (nsp5_pooled.pooled), LVBench and CG-Bench mini pooled.

    AA1 (primary)  aa_cal_gmm - aa_gmmeva_full       CaliFrame's placement inside GMM-EVA vs full GMM-EVA
    AA2            aa_cal_img - aa_gmmeva_full       CaliFrame vs full GMM-EVA, same image input
    AA3            aa_cal_gmm - aa_cal_img           GMM-EVA's allocation added to CaliFrame (descriptive)
    AA4            aa_gmmeva_full - aa_gmmeva_sel_img  GMM-EVA's allocation added to its selection (descriptive)

Descriptive: each image arm against its video-input counterpart, and the visual tokens and primary
frames per question logged in each row.

    .venv/bin/python nsp5_aa_score.py          # from research/TimeLens
"""
import json

import numpy as np

from nsp5_pooled import path
from nsp5_wave1_score import show

LC = ("lvbench", "cgbench")


def token_stats(arm: str) -> str:
    vt, prim = [], []
    for b in LC:
        for line in open(path(arm, b), encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                vt.append(r.get("visual_tokens", np.nan))
                prim.append(r.get("n_primary", np.nan))
    return (f"{arm}: visual tokens per question mean {np.nanmean(vt):,.0f} (min {np.nanmin(vt):,.0f}, "
            f"max {np.nanmax(vt):,.0f}); primary frames mean {np.nanmean(prim):.1f}")


def main() -> None:
    show("AA1 (primary)", "aa_cal_gmm", "aa_gmmeva_full", LC)
    show("AA2", "aa_cal_img", "aa_gmmeva_full", LC)
    show("AA3 (descriptive)", "aa_cal_gmm", "aa_cal_img", LC)
    show("AA4 (descriptive)", "aa_gmmeva_full", "aa_gmmeva_sel_img", LC)
    print("== image input against video input (descriptive)")
    show("CaliFrame, images minus video", "aa_cal_img", "q8_kelly_eps0", LC)
    show("GMM-EVA's selection, images minus video", "aa_gmmeva_sel_img", "f2c_topk_nr", LC)
    print("== visual tokens")
    for arm in ("aa_cal_gmm", "aa_gmmeva_full", "aa_cal_img", "aa_gmmeva_sel_img"):
        print("  " + token_stats(arm))


if __name__ == "__main__":
    main()
