"""Study H (METHODS-LOG 2026-11-09): M2's gain over uniform sampling on LVBench by answer model.

Primary tests H1 (Qwen2.5-VL-32B) and H2 (Qwen2.5-VL-72B): M2 minus uniform in plain accuracy, a
95% interval from a paired bootstrap that resamples videos (5,000 draws, seed 0), and two-sided
bootstrap p-values, Holm-adjusted across the two. Secondary, descriptive: the gain at 7B, 32B and
72B side by side, and each model's accuracy on the questions where uniform sampling puts no frame
on the evidence against those where it puts some (frame choice does not depend on the answer
model, so one uniform file's in_span serves every model).

    .venv/bin/python nsp5_score_h.py
"""
import json

import numpy as np

from nsp5_compare import video_of

DRAWS, SEED = 5000, 0
MODELS = [("Qwen2.5-VL-7B", "e3_uniform", "q8_kelly_eps0"),
          ("Qwen2.5-VL-32B", "h_uniform_q25_32b", "h_q8_kelly_eps0_q25_32b"),
          ("Qwen2.5-VL-72B", "h_uniform_q25_72b", "h_q8_kelly_eps0_q25_72b")]
PRIMARY = {"Qwen2.5-VL-32B": "H1", "Qwen2.5-VL-72B": "H2"}


def rows(name: str) -> dict:
    with open(f"logs/nsp5/{name}.jsonl", encoding="utf-8") as fh:
        return {r["key"]: r for r in map(json.loads, filter(str.strip, fh))}


def boot(diff: np.ndarray, groups: list) -> np.ndarray:
    """Bootstrap means of the paired difference, whole videos resampled (nsp5_compare.boot_ci)."""
    rng = np.random.default_rng(SEED)
    _, inv = np.unique(np.asarray(groups), return_inverse=True)
    sums, cnts = np.bincount(inv, weights=diff), np.bincount(inv).astype(float)
    idx = rng.integers(0, len(sums), size=(DRAWS, len(sums)))
    return sums[idx].sum(1) / cnts[idx].sum(1)


def holm(p: list) -> list:
    out, run = [0.0] * len(p), 0.0
    for rank, i in enumerate(np.argsort(p)):
        run = max(run, min(1.0, (len(p) - rank) * p[i]))
        out[i] = run
    return out


def main() -> None:
    ref_span = rows("h_uniform_q25_32b")
    other = rows("h_uniform_q25_72b")
    assert all(ref_span[k]["in_span"] == other[k]["in_span"] for k in ref_span), "uniform frames differ"
    covered = {k: r["in_span"] > 0 for k, r in ref_span.items()}
    res, pvals = [], []
    for model, uni, ours in MODELS:
        u, m = rows(uni), rows(ours)
        keys = sorted(set(u) & set(m))
        assert len(keys) == 1534, (model, len(keys))
        a = np.array([float(u[k]["correct"]) for k in keys])
        b = np.array([float(m[k]["correct"]) for k in keys])
        means = boot(b - a, [video_of(k) for k in keys])
        lo, hi = np.percentile(means, [2.5, 97.5])
        p = min(1.0, 2 * min(np.mean(means <= 0), np.mean(means >= 0)))
        split = {}
        for name, sel in (("no frame on evidence", False), ("some frame on evidence", True)):
            idx = [i for i, k in enumerate(keys) if covered[k] == sel]
            split[name] = (len(idx), 100 * a[idx].mean(), 100 * b[idx].mean())
        res.append((model, 100 * a.mean(), 100 * b.mean(), 100 * (b - a).mean(), 100 * lo, 100 * hi,
                    p, split))
        if model in PRIMARY:
            pvals.append(p)
    adj = dict(zip([r[0] for r in res if r[0] in PRIMARY], holm(pvals)))
    floor = 1 / DRAWS
    print("LVBench, 1,534 questions, 64 frames; M2 minus uniform, 95% interval over videos")
    for model, ua, ma, d, lo, hi, p, split in res:
        test = PRIMARY.get(model, "  ")
        ps = f"p {'<' if p < floor else '='} {max(p, floor):.4f}"
        hs = f", Holm p {'<' if adj.get(model, 1) < floor else '='} {max(adj[model], floor):.4f}" \
            if model in adj else ""
        print(f"{test} {model:15s} uniform {ua:5.2f}  M2 {ma:5.2f}  diff {d:+6.2f} "
              f"[{lo:+.2f}, {hi:+.2f}]  {ps}{hs}")
        for name, (n, a_, b_) in split.items():
            print(f"     {name:23s} n={n:4d}  uniform {a_:5.2f}  M2 {b_:5.2f}  diff {b_ - a_:+6.2f}")


if __name__ == "__main__":
    main()
