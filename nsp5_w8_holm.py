"""Study W8, the Holm part (NS-P5, METHODS-LOG 2026-11-12, "Reviewer-requested experiments, wave 1"):
Holm-adjusted p-values for the paper's headline claims, beside the 95 % intervals its tables print.
Qwen2.5-VL-7B, 64 frames unless a family says otherwise; CaliFrame is M2, q8_kelly_eps0.

Families, each adjusted on its own:
  T1   Table 1 (tab:main), Qwen2.5-VL-7B: CaliFrame minus uniform on each benchmark and pooled (6)
  T2   Table 2 (tab:selectors): CaliFrame minus each published selector, pooled: BOLT, AKS and
       GMM-EVA's selection (3)
  T4   Table 4 (tab:ablation): CaliFrame minus each ablation row, pooled (15)
  supplement: Table 1's Qwen3-VL-8B rows as their own family (6), and both answer models' rows of
       Table 1 as one family (12)

p is the two-sided bootstrap p-value from the same draws as the printed interval (videos resampled
within each benchmark, 5,000 draws, seed 0: nsp5_pooled.py), the inversion of the percentile
interval with the (1 + count) / (B + 1) correction. It cannot fall below 2 / 5,001 = 0.0004; a value
at that floor (no draw on the other side of zero) is printed with "<=", and so is its Holm value,
which is then m x 0.0004 at most. Beside it, the normal approximation with the bootstrap standard
error, which has no floor. Every interval is asserted equal to nsp5_pooled.pooled's.

    .venv/bin/python nsp5_w8_holm.py

writes logs/nsp5/analysis/w8_holm.json and w8_holm.txt.
"""
from __future__ import annotations

from nsp5_pooled import BENCH, pooled
from nsp5_w_paired import (DRAWS, GMM, NAMES, OURS, UNIFORM, boot, by_bench, correct, half_up, holm,
                           p_boot, p_normal, write_outputs)

OURS_Q3, UNIFORM_Q3 = "q8_kelly_eps0_q3", "g1_q3vl_uniform:uniform_q3"
FLOOR = 2 / (DRAWS + 1)

T1 = [(NAMES[b], OURS, UNIFORM, (b,)) for b in BENCH] + [("pooled", OURS, UNIFORM, BENCH)]
T1_Q3 = [(f"Qwen3-VL-8B {NAMES[b]}", OURS_Q3, UNIFORM_Q3, (b,)) for b in BENCH] + \
        [("Qwen3-VL-8B pooled", OURS_Q3, UNIFORM_Q3, BENCH)]
T2 = [("BOLT", OURS, "bolt_own", BENCH), ("AKS", OURS, "aks_own", BENCH),
      ("GMM-EVA selection", OURS, GMM, BENCH)]
T4 = [(label, OURS, arm, BENCH) for label, arm in (
    ("Options appended (F2C)", "f2c_kelly_eps0"), ("SigLIP, per-option max.", "cal_a2_kelly_eps0"),
    ("CLIP, question (BOLT's)", "kelly_clip_eps0"), ("BLIP, question (AKS's)", "cal_kelly_blip_eps0"),
    ("Top-k", "q8opt_topk_nr"), ("BOLT's rule, exponent 3", "bolt_its_q8"),
    ("BOLT's rule, fitted exponent", "cal_q8_its_fit"), ("Q-Frame's rule, 128 candidates", "m_qframe128_q8"),
    ("Q-Frame's rule, every second", "m_qframe_all_q8"), ("AKS's rule", "aks_ada_q8"),
    ("tau/2 = 0.018", "q8_kelly_eps0_tau179"), ("2 tau = 0.072", "q8_kelly_eps0_tau716"),
    ("uniform share 0.1", "q8_kelly_eps010"), ("uniform share 0.25", "q8_kelly_eps025"),
    ("uniform share 0.5", "q8_kelly_nr"))]
FAMILIES = (("T1: Table 1, Qwen2.5-VL-7B, CaliFrame minus uniform", T1),
            ("T2: Table 2, CaliFrame minus each published selector, pooled", T2),
            ("T4: Table 4, CaliFrame minus each ablation row, pooled", T4),
            ("supplement: Table 1, Qwen3-VL-8B, CaliFrame minus uniform", T1_Q3),
            ("supplement: Table 1, both answer models as one family", T1 + T1_Q3))


def run_test(label: str, ours: str, ref: str, benches: tuple) -> dict:
    a, r = correct(ours, benches), correct(ref, benches)
    p = boot(a, r, by_bench(a))
    _, d, (lo, hi) = pooled(ours, ref, benches=benches)
    assert (d, lo, hi) == (p.diff, p.lo, p.hi), (label, d, lo, hi, p.diff, p.lo, p.hi)
    return {"label": label, "ours": ours, "ref": ref, "benches": list(benches), "n": p.n,
            "diff": p.diff, "ci": [p.lo, p.hi], "table": f"{half_up(p.diff)} [{half_up(p.lo)}, {half_up(p.hi)}]",
            "p": p_boot(p.draws), "p_normal": p_normal(p.diff, p.draws)}


def fmt_p(p: float, at_floor: bool) -> str:
    return f"<={p:.4f}" if at_floor else (f"{p:.4f}" if p >= 1e-4 else f"{p:.1e}")


def main() -> None:
    cache: dict = {}
    data, lines = {"draws": DRAWS, "seed": 0, "floor": FLOOR, "families": {}}, [
        "W8 (Holm part): Holm-adjusted p-values beside the printed 95% intervals; p from the same 5,000 "
        "video-bootstrap draws (floor 0.0004), p (normal) from the bootstrap standard error"]
    for name, tests in FAMILIES:
        res = []
        for label, ours, ref, benches in tests:
            key = (ours, ref, benches)
            if key not in cache:
                cache[key] = run_test(label, ours, ref, benches)
            res.append(dict(cache[key], label=label))
        for field in ("p", "p_normal"):
            for x, adj in zip(res, holm([x[field] for x in res])):
                x[f"{field}_holm"] = adj
        data["families"][name] = res
        lines += ["", f"{name} ({len(res)} tests)",
                  f"  {'comparison':<32}{'n':>6}  {'difference [95% CI]':<27}{'as printed':<20}"
                  f"{'p':>9}{'Holm p':>9}{'p (normal)':>12}{'Holm (normal)':>14}  Holm p < 0.05"]
        for x in res:
            floor = x["p"] <= FLOOR + 1e-12
            lines.append(f"  {x['label'][:31]:<32}{x['n']:>6}  {x['diff']:+6.2f} [{x['ci'][0]:+6.2f}, {x['ci'][1]:+6.2f}]   "
                         f"{x['table']:<20}{fmt_p(x['p'], floor):>9}{fmt_p(x['p_holm'], floor):>9}"
                         f"{fmt_p(x['p_normal'], False):>12}{fmt_p(x['p_normal_holm'], False):>14}  "
                         f"{'yes' if x['p_holm'] < 0.05 else 'no'}")
    write_outputs("w8_holm", data, lines)


if __name__ == "__main__":
    main()
