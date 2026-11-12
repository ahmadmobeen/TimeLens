"""Study W5 (NS-P5, METHODS-LOG 2026-11-12, "Reviewer-requested experiments, wave 1"): CaliFrame's
(M2, q8_kelly_eps0) lead over GMM-EVA's selection (f2c_topk_nr), Qwen2.5-VL-7B, 64 frames.

  lobo    the pooled lead with each benchmark left out in turn (four benchmarks pooled)
  equal   the benchmark-equal average: the mean of the five per-benchmark leads, its interval from
          the same draws (each draw's five per-benchmark differences, averaged)
  lvb     LongVideoBench by its official duration groups (8-15 s, 15-60 s, 3-10 min, 15-60 min)

Paired differences in plain accuracy, in points; 95 % intervals from the bootstrap of
nsp5_pooled.py (videos resampled within each benchmark, 5,000 draws, seed 0). Descriptive; W5 has no
registered reading. Every pooled row is checked against nsp5_pooled.pooled itself.

    .venv/bin/python nsp5_w5_lead.py

writes logs/nsp5/analysis/w5_lead.json and w5_lead.txt.
"""
from __future__ import annotations

import bench
from nsp5_pooled import BENCH, pooled
from nsp5_w_paired import GMM, NAMES, OURS, UNIFORM, boot, by_bench, correct, write_outputs

LVB_GROUPS = {"15": "8-15 s", "60": "15-60 s", "600": "3-10 min", "3600": "15-60 min"}


def check(p, a: str, r: str, benches) -> None:
    """The row must be the tables' own number (nsp5_pooled.pooled)."""
    _, d, (lo, hi) = pooled(a, r, benches=benches)
    assert (d, lo, hi) == (p.diff, p.lo, p.hi), (benches, d, lo, hi, p.diff, p.lo, p.hi)


def main() -> None:
    m, g, u = correct(OURS), correct(GMM), correct(UNIFORM)
    keys = by_bench(m)
    full = boot(m, g, keys)
    check(full, OURS, GMM, BENCH)
    data = {"comparison": f"{OURS} minus {GMM}", "full": full.record(),
            "per_benchmark": {b: {k: round(v, 3) for k, v in x.items()} for b, x in full.per.items()}}
    lines = ["W5: CaliFrame (M2) minus GMM-EVA's selection, Qwen2.5-VL-7B, 64 frames (points; 95% video-bootstrap interval)",
             "", f"all five benchmarks    n={full.n:5d}  {full.a:5.2f} vs {full.r:5.2f}  {full.text()}"]
    for b, x in full.per.items():
        lines.append(f"  {NAMES[b]:<20} n={x['n']:5d}  {x['a']:5.2f} vs {x['r']:5.2f}  {x['diff']:+6.2f}")

    lines += ["", "leave one benchmark out (pooled over the other four)"]
    data["lobo"] = {}
    for left in BENCH:
        rest = tuple(b for b in BENCH if b != left)
        p = boot(m, g, {b: keys[b] for b in rest})
        check(p, OURS, GMM, rest)
        data["lobo"][left] = p.record()
        lines.append(f"  without {NAMES[left]:<16} n={p.n:5d}  {p.a:5.2f} vs {p.r:5.2f}  {p.text()}")

    data["benchmark_equal"] = {"diff": round(full.equal, 3), "ci": [round(full.equal_lo, 3), round(full.equal_hi, 3)]}
    lines += ["", "benchmark-equal average (mean of the five per-benchmark leads)",
              f"  {full.equal:+6.2f} [{full.equal_lo:+6.2f}, {full.equal_hi:+6.2f}]"
              f"   (question-weighted pooled: {full.diff:+.2f})"]

    group = {it.key: str(it.meta["duration_group"]) for it in bench.load("lvb")}
    lines += ["", "LongVideoBench by official duration group",
              f"  {'group':<10}{'n':>5}{'videos':>7}{'uniform':>9}{'GMM-EVA':>9}{'M2':>7}"
              f"   {'M2 - GMM-EVA [95% CI]':<26}{'M2 - uniform [95% CI]'}"]
    data["lvb_duration_groups"] = {}
    for gid, label in LVB_GROUPS.items():
        ks = {"lvb": [k for k in keys["lvb"] if group[k] == gid]}
        mg, mu = boot(m, g, ks), boot(m, u, ks)
        data["lvb_duration_groups"][label] = {"m2_minus_gmm": mg.record(), "m2_minus_uniform": mu.record()}
        lines.append(f"  {label:<10}{mg.n:>5}{mg.videos:>7}{mu.r:>9.2f}{mg.r:>9.2f}{mg.a:>7.2f}"
                     f"   {mg.text():<26}{mu.text()}")
    write_outputs("w5_lead", data, lines)


if __name__ == "__main__":
    main()
