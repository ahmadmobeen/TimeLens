"""Study Q (NeedleScan METHODS-LOG 2026-11-12, registered in 02c2ca7): is CaliFrame's probability
calibrated question by question?

On LVBench and CG-Bench, the two evaluation sets that annotate the evidence (neither used in the
calibration): for each question and alpha, the fewest seconds holding at least alpha of the
probability, and the share of the evidence seconds inside them (coverage). Calibrated means a mean
coverage of alpha, overall (Q1) and within each third of the questions by concentration (Q2: the
share of the video's seconds that hold half the probability). CPU only:

    .venv/bin/python nsp5_reliability.py

prints Q1, Q2 and the descriptive grid and writes logs/nsp5/reliability.json.
"""
import json

import numpy as np

import bench
from nsp5_compare import video_of
from nsp5_method_figures import evidence_mask
from t1_data import lvbench_items, span_seconds

TAU = 3.58
ALPHAS = (0.25, 0.5, 0.75, 0.9)
Q1_ALPHAS = (0.25, 0.5, 0.75)
TOL = 0.10
CURVES = {"lvbench": "data/t1/priors/qwen3vl_emb_8b/curves_maxopt.npz",
          "cgbench": "data/t1/priors/qwen3vl_emb_8b/cgbench_curves_maxopt.npz"}
VARIANTS = {"CaliFrame": ("softmax", TAU), "softmax tau/2": ("softmax", TAU / 2),
            "softmax 2 tau": ("softmax", 2 * TAU), "BOLT's rule, exponent 3": ("power", 3.0)}
THIRDS = ("most concentrated", "middle", "most spread")
OUT = "logs/nsp5/reliability.json"
DRAWS, SEED = 5000, 0


def probability(z: np.ndarray, transform: str, c: float) -> np.ndarray:
    z = z.astype(np.float64)
    if transform == "softmax":
        x = z / c
        p = np.exp(x - x.max())
    else:                                    # BOLT's rule: min-max normalised scores to a power
        s = z - z.min()
        p = (s / s.max()) ** c if s.max() > 0 else np.ones_like(z)
    return p / p.sum()


def questions(b: str) -> list:
    """(row-style key, scores, evidence mask) for every question of the benchmark."""
    Z = np.load(CURVES[b])
    out = []
    if b == "lvbench":
        for it in lvbench_items():
            k = f"{it['key']}>>>{it['uid']}"
            z = Z[f"lvbench:{k}"]
            out.append((k, z, evidence_mask(it["span"], len(z))))
    else:
        for it in bench.load_cgbench():
            z = Z[it.key]
            m = np.zeros(len(z), bool)
            for a, c in it.spans:
                lo, hi = span_seconds(a, c, len(z))
                m[lo:hi] = True
            out.append((it.key, z, m))
    return out


def coverage(p: np.ndarray, mask: np.ndarray) -> tuple:
    """Coverage of the alpha-region for each alpha, and the share of seconds holding half of p."""
    order = np.argsort(-p, kind="stable")
    cum = np.cumsum(p[order])
    n_ev = mask.sum()
    cov = {}
    for a in ALPHAS:
        k = int(np.searchsorted(cum, a)) + 1
        cov[a] = float(mask[order[:k]].sum() / n_ev)
    conc = (int(np.searchsorted(cum, 0.5)) + 1) / len(p)
    return cov, conc


def grid(qs: list, transform: str, c: float, thirds: np.ndarray) -> dict:
    """Mean coverage per alpha, overall and per third, with video-bootstrap 95% intervals."""
    cov = np.array([[coverage(probability(z, transform, c), m)[0][a] for a in ALPHAS] for _, z, m in qs])
    _, inv = np.unique([video_of(k) for k, _, _ in qs], return_inverse=True)
    nv = inv.max() + 1
    groups = {"all": np.ones(len(qs), bool), **{t: thirds == i for i, t in enumerate(THIRDS)}}
    sums = {g: np.stack([np.bincount(inv, weights=cov[:, j] * sel, minlength=nv) for j in range(len(ALPHAS))])
            for g, sel in groups.items()}
    cnts = {g: np.bincount(inv, weights=sel.astype(float), minlength=nv) for g, sel in groups.items()}
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, nv, size=(DRAWS, nv))
    out = {}
    for g in groups:
        point = sums[g].sum(1) / cnts[g].sum()
        boot = sums[g][:, idx].sum(2) / cnts[g][idx].sum(1)          # (alphas, draws)
        lo, hi = np.percentile(boot, [2.5, 97.5], axis=1)
        out[g] = {str(a): {"coverage": float(point[j]), "ci": [float(lo[j]), float(hi[j])]}
                  for j, a in enumerate(ALPHAS)}
        out[g]["n"] = int(groups[g].sum())
    return out


def main() -> None:
    report, holds = {}, True
    for b in CURVES:
        qs = [q for q in questions(b) if q[2].any()]
        concs = np.array([coverage(probability(z, "softmax", TAU), m)[1] for _, z, m in qs])
        ranks = np.argsort(np.argsort(concs, kind="stable"), kind="stable")
        thirds = np.minimum(3 * ranks // len(qs), 2)
        report[b] = {"questions": len(qs), "concentration_cuts": [float(np.percentile(concs, 100 / 3)),
                                                                   float(np.percentile(concs, 200 / 3))]}
        for name, (transform, c) in VARIANTS.items():
            report[b][name] = grid(qs, transform, c, thirds)
        g = report[b]["CaliFrame"]
        q1 = {a: g["all"][str(a)] for a in Q1_ALPHAS}
        q2 = {t: g[t]["0.5"] for t in THIRDS}
        ok = (all(abs(v["coverage"] - a) <= TOL for a, v in q1.items())
              and all(abs(v["coverage"] - 0.5) <= TOL for v in q2.values()))
        holds &= ok
        print(f"== {b}: {len(qs)} questions; half the probability on {100 * np.median(concs):.1f}% of the seconds (median)")
        for a, v in q1.items():
            print(f"  Q1 alpha {a:.2f}: coverage {v['coverage']:.3f} [{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]")
        for t, v in q2.items():
            print(f"  Q2 {t:17s} (n={g[t]['n']}): coverage of the 50% region {v['coverage']:.3f} "
                  f"[{v['ci'][0]:.3f}, {v['ci'][1]:.3f}]")
        for name in VARIANTS:
            row = report[b][name]
            cells = "  ".join(f"{a}: {row['all'][str(a)]['coverage']:.2f}" for a in ALPHAS)
            thirds_50 = " / ".join(f"{row[t]['0.5']['coverage']:.2f}" for t in THIRDS)
            print(f"  {name:24s} all {cells} | 50% region by third {thirds_50}")
    report["holds"] = bool(holds)
    print("Reading:", "HOLDS on both benchmarks" if holds else "does not hold (see the values above)")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
