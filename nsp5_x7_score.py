"""Score Study X7 (reviewer wave 2, METHODS-LOG 2026-11-12), descriptive: paired differences in plain
accuracy, videos resampled within each benchmark, 5,000 draws, seed 0 (nsp5_pooled.pooled).

(a) 256 frames under the shared 14,336 tokens (56 a frame) on LVBench, CG-Bench and Video-MME: M2
    minus uniform and minus GMM-EVA's selection, with the same differences at 64 and 128 frames on
    those benchmarks beside them. Study G's slope refit with N = 256 added is nsp5_trend.py on the
    same three benchmarks (printed command below).
(b) Tokens per frame at 64 frames on LVBench and CG-Bench: M2 minus uniform at 112, 224 (the
    harness default) and 384 tokens a frame.

    .venv/bin/python nsp5_x7_score.py          # from research/TimeLens
"""
from nsp5_wave1_score import show

LCV = ("lvbench", "cgbench", "videomme")
LC = ("lvbench", "cgbench")
UNI64 = "e3_uniform:uniform"           # uniform at 64 frames: its LVBench stem differs


def main() -> None:
    print("== (a) LVBench, CG-Bench and Video-MME pooled")
    for n, m2, uni, sel in ((64, "q8_kelly_eps0", UNI64, "f2c_topk_nr"),
                            (128, "n128_q8_kelly_eps0", None, "n128_f2c_topk"),
                            (256, "n256_q8_kelly_eps0", "n256_uniform", "n256_f2c_topk")):
        if uni:
            show(f"N={n} M2 - uniform", m2, uni, LCV)
        show(f"N={n} M2 - GMM-EVA's selection", m2, sel, LCV)
    print("slope refit: .venv/bin/python nsp5_trend.py n16_q8_kelly_eps0:n16_f2c_topk:16 "
          "n32_q8_kelly_eps0:n32_f2c_topk:32 q8_kelly_eps0:f2c_topk_nr:64 "
          "n128_q8_kelly_eps0:n128_f2c_topk:128 n256_q8_kelly_eps0:n256_f2c_topk:256 "
          "--benches=lvbench,cgbench,videomme")
    print("== (b) tokens per frame at 64 frames, LVBench and CG-Bench pooled")
    for tok, m2, uni in ((112, "x7_tok112_q8_kelly_eps0", "x7_tok112_uniform"),
                         (224, "q8_kelly_eps0", UNI64),
                         (384, "x7_tok384_q8_kelly_eps0", "x7_tok384_uniform")):
        show(f"{tok} tokens a frame: M2 - uniform", m2, uni, LC)


if __name__ == "__main__":
    main()
