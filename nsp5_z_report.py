"""Study Z (METHODS-LOG 2026-11-14), descriptive: the paper's CG-Bench arms on CG-Bench's full split without the
calibration videos (the mini split's 3,000 questions and the 8,256 new ones, 11,256 questions), and the five-benchmark
pools of the ablation table with that split in place of the mini split, per question and as the mean of the five
benchmarks. Paired differences, videos resampled within each benchmark (the two CG-Bench parts in one stratum, since
they share videos), 5,000 draws, seed 0.

    .venv/bin/python nsp5_z_report.py        # from research/TimeLens; writes logs/nsp5/analysis/z_full.json
"""
import json
import os

import numpy as np

from nsp5_compare import video_of
from nsp5_pooled import BENCH, SIZE, load, path

M2 = "q8_kelly_eps0"
FULL_SIZE = {"cgbench": 3000, "cgbench_fresh": 8256}
DRAWS, SEED = 5000, 0
# The ablation table (make_tables.ablation_table), Table 3's selectors, and Fig. 4(b)'s temperatures
ARMS = ["f2c_kelly_eps0", "u_q8stem_kelly_eps0", "cal_a2_kelly_eps0", "kelly_clip_eps0", "cal_kelly_blip_eps0",
        "q8opt_topk_nr", "t_stack4", "bolt_its_q8", "cal_q8_its_fit", "m_qframe128_q8", "m_qframe_all_q8", "aks_ada_q8",
        "q8_kelly_eps0_tau179", "q8_kelly_eps0_tau716", "q8_kelly_eps010", "q8_kelly_eps025", "q8_kelly_nr",
        "e3_uniform:uniform", "bolt_own", "aks_own", "f2c_topk_nr",
        "q8_kelly_eps0_tau100", "q8_kelly_eps0_tau250", "q8_kelly_eps0_tau500", "q8_kelly_eps0_tau1000"]
# Fig. 4(c, d): frame budgets, (CaliFrame, GMM-EVA's selection, uniform) per budget
BUDGETS = {16: ("n16_q8_kelly_eps0", "n16_f2c_topk", "n16_uniform"),
           32: ("n32_q8_kelly_eps0", "n32_f2c_topk", "n32_uniform"),
           64: (M2, "f2c_topk_nr", "e3_uniform:uniform"),
           128: ("n128_q8_kelly_eps0", "n128_f2c_topk", "n128_uniform")}


def full(arm: str) -> dict:
    """The arm's answers on CG-Bench's full split, or {} while a part is missing or incomplete."""
    out = {}
    for part, n in FULL_SIZE.items():
        p = path(arm, part)
        rows = load(p) if os.path.exists(p) else {}
        if len(rows) != n:
            return {}
        out.update(rows)
    return out


def strata_diff(pairs: list) -> tuple:
    """Pooled difference over questions, its 95% interval with videos resampled within each stratum, and the
    mean of the strata's own differences. pairs: one (a, r) answer-dict pair per stratum."""
    strata, diffs = [], []
    for a, r in pairs:
        keys = sorted(set(a) & set(r))
        d = np.array([a[k] - r[k] for k in keys])
        _, inv = np.unique([video_of(k) for k in keys], return_inverse=True)
        strata.append((np.bincount(inv, weights=d), np.bincount(inv).astype(float)))
        diffs.append(d)
    rng = np.random.default_rng(SEED)
    means = []
    for _ in range(DRAWS):
        s = c = 0.0
        for sums, cnts in strata:
            idx = rng.integers(0, len(sums), len(sums))
            s += sums[idx].sum()
            c += cnts[idx].sum()
        means.append(s / c)
    allq = np.concatenate(diffs)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return 100 * allq.mean(), 100 * lo, 100 * hi, 100 * float(np.mean([d.mean() for d in diffs]))


def five(arm: str, cg: str) -> list:
    """(M2, arm) answer dicts per benchmark, CG-Bench as its mini split ('mini') or its full split ('full')."""
    pairs = []
    for b in BENCH:
        if b == "cgbench" and cg == "full":
            pairs.append((full(M2), full(arm)))
        else:
            pairs.append((load(path(M2, b)), load(path(arm, b))))
    return pairs


def acc(rows: dict) -> float:
    return 100 * float(np.mean(list(rows.values())))


def complete5(arm: str) -> bool:
    return all(os.path.exists(path(arm, b)) and len(load(path(arm, b))) == SIZE[b] for b in BENCH)


def main() -> None:
    report = {"arms": {}, "budgets": {}}
    m2 = full(M2)
    print(f"CaliFrame on the full split: {acc(m2):.2f} (n={len(m2)})")
    print(f"{'arm':<26} {'mini':>6} {'new':>6} {'full':>6}  {'M2-arm full [95%]':>24}  "
          f"{'pooled5 mini':>12} {'pooled5 full [95%]':>26} {'mean5 full':>10}")
    for arm in ARMS:
        rows = full(arm)
        if not rows:
            print(f"{arm:<26} incomplete on the full split")
            continue
        mini = load(path(arm, "cgbench"))
        new = load(path(arm, "cgbench_fresh"))
        d, lo, hi, _ = strata_diff([(m2, rows)])
        entry = {"mini": acc(mini), "new": acc(new), "full": acc(rows), "m2_minus_full": [d, lo, hi]}
        line = f"{arm:<26} {acc(mini):6.2f} {acc(new):6.2f} {acc(rows):6.2f}  {d:+6.2f} [{lo:+6.2f}, {hi:+6.2f}]"
        if complete5(arm):
            dm, _, _, _ = strata_diff(five(arm, "mini"))
            df, flo, fhi, mean5 = strata_diff(five(arm, "full"))
            entry.update(pooled5_mini=dm, pooled5_full=[df, flo, fhi], mean5_full=mean5)
            line += f"  {dm:+12.2f} {df:+8.2f} [{flo:+6.2f}, {fhi:+6.2f}] {mean5:+10.2f}"
        report["arms"][arm] = entry
        print(line)
    print("== frame budgets on the full split (Fig. 4c, d)")
    for n, (ours, gmm, uni) in BUDGETS.items():
        o, g, u = full(ours), full(gmm), full(uni)
        if not (o and g and u):
            print(f"  {n:3d} frames: incomplete")
            continue
        dg, glo, ghi, _ = strata_diff([(o, g)])
        du, ulo, uhi, _ = strata_diff([(o, u)])
        report["budgets"][n] = {"califrame": acc(o), "gmm_sel": acc(g), "uniform": acc(u),
                                "minus_gmm": [dg, glo, ghi], "minus_uniform": [du, ulo, uhi]}
        print(f"  {n:3d} frames: CaliFrame {acc(o):5.2f}  GMM-EVA's selection {acc(g):5.2f}  uniform {acc(u):5.2f}  "
              f"minus GMM {dg:+5.2f} [{glo:+5.2f}, {ghi:+5.2f}]  minus uniform {du:+5.2f} [{ulo:+5.2f}, {uhi:+5.2f}]")
    os.makedirs("logs/nsp5/analysis", exist_ok=True)
    with open("logs/nsp5/analysis/z_full.json", "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print("wrote logs/nsp5/analysis/z_full.json")


if __name__ == "__main__":
    main()
