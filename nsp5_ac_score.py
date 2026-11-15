"""Score Study AC (METHODS-LOG 2026-11-15, registered a28fcb95; descriptive): Study AA's four arms on
Video-MME, and GMM-EVA's selection answered by Gemma-4-31B. Paired differences in plain accuracy, videos
resampled within each benchmark, 5,000 draws, seed 0 (nsp5_pooled.pooled).

    (a) AA1-AA4 on Video-MME, and pooled over LVBench, CG-Bench and Video-MME (AA's registered
        two-benchmark tests stay primary); each image arm against its video-input arm
    (b) M2 minus GMM-EVA's selection with Gemma-4-31B (Study K settings, 64 frames), LVBench and
        Video-MME pooled and per benchmark; M2 minus uniform with the same model, for scale

    .venv/bin/python nsp5_ac_score.py          # from research/TimeLens
"""
from nsp5_wave1_score import show

V, LCV, LV = ("videomme",), ("lvbench", "cgbench", "videomme"), ("lvbench", "videomme")
AA = (("AA1", "aa_cal_gmm", "aa_gmmeva_full"), ("AA2", "aa_cal_img", "aa_gmmeva_full"),
      ("AA3", "aa_cal_gmm", "aa_cal_img"), ("AA4", "aa_gmmeva_full", "aa_gmmeva_sel_img"))


def main() -> None:
    print("== (a) Study AA's arms on Video-MME, and pooled with LVBench and CG-Bench")
    for label, a, r in AA:
        show(f"{label}, Video-MME", a, r, V)
        show(f"{label}, three benchmarks", a, r, LCV)
    show("CaliFrame, images minus video, Video-MME", "aa_cal_img", "q8_kelly_eps0", V)
    show("GMM-EVA's selection, images minus video, Video-MME", "aa_gmmeva_sel_img", "f2c_topk_nr", V)
    print("== (b) Gemma-4-31B")
    show("M2 minus GMM-EVA's selection", "k_q8_kelly_eps0_gemma4_31b", "k_f2c_topk_nr_gemma4_31b", LV)
    show("M2 minus uniform", "k_q8_kelly_eps0_gemma4_31b", "k_uniform_gemma4_31b", LV)


if __name__ == "__main__":
    main()
