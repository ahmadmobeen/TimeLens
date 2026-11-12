"""Study P (NeedleScan METHODS-LOG 2026-11-12, registered in 027aa7d): CaliFrame's lead over top-k on
questions whose calibrated probability has two or more major peaks, against the rest.

Exploratory, on existing 64-frame runs with Qwen2.5-VL-7B, CPU only:

    .venv/bin/python nsp5_multipeak.py

prints P1 and the descriptive contrasts and writes logs/nsp5/multipeak.json.

Multi-peak, as registered: CaliFrame's probability over the seconds (8B per-option maximum, tau 3.58);
the fewest seconds holding 80% of it, grouped into regions wherever consecutive seconds are at most
60 s apart; multi-peak when two or more regions hold at least 20% of the probability each.
"""
import json

import numpy as np

import bench
from nsp5_compare import video_of
from nsp5_pooled import BENCH, SIZE, load, path

TAU, MASS, GAP, SHARE = 3.58, 0.8, 60, 0.2
SPAN_GAP = 10                     # CG-Bench descriptive: annotated spans more than 10 s apart
CURVES = {"lvbench": "data/t1/priors/qwen3vl_emb_8b/curves_maxopt.npz"}
CURVES.update({b: f"data/t1/priors/qwen3vl_emb_8b/{b}_curves_maxopt.npz" for b in BENCH[1:]})
OURS = "q8_kelly_eps0"
OTHERS = {"top-k": "q8opt_topk_nr", "GMM-EVA's selection": "f2c_topk_nr", "uniform": "e3_uniform:uniform"}
OUT = "logs/nsp5/multipeak.json"
DRAWS, SEED = 5000, 0


def major_peaks(z: np.ndarray) -> int:
    """Regions of the 80% highest-probability seconds (gaps of at most 60 s) holding >= 20% each."""
    x = z.astype(np.float64) / TAU
    p = np.exp(x - x.max())
    p /= p.sum()
    order = np.argsort(-p)
    k = int(np.searchsorted(np.cumsum(p[order]), MASS)) + 1
    top = np.sort(order[:k])
    regions, mass = [], p[top[0]]
    for a, b in zip(top[:-1], top[1:]):
        if b - a > GAP:
            regions.append(mass)
            mass = 0.0
        mass += p[b]
    regions.append(mass)
    return int(sum(m >= SHARE for m in regions))


def curve_key(b: str, key: str) -> str:
    return f"lvbench:{key}" if b == "lvbench" else key


def flags() -> dict:
    """bench -> {question key: True if multi-peak}."""
    out = {}
    for b in BENCH:
        Z = np.load(CURVES[b])
        rows = load(path(OURS, b))
        out[b] = {k: bool(major_peaks(Z[curve_key(b, k)]) >= 2) for k in rows}
    return out


def cg_multispan() -> dict:
    """CG-Bench question key -> True if its annotated spans form two or more groups > 10 s apart."""
    out = {}
    for it in bench.load_cgbench():
        spans = sorted((float(a), float(c)) for a, c in it.spans)
        groups = 1
        end = spans[0][1]
        for a, c in spans[1:]:
            if a > end + SPAN_GAP:
                groups += 1
            end = max(end, c)
        out[it.key] = bool(groups >= 2)
    return out


def contrast(other: str, group: dict, benches=BENCH) -> dict:
    """(OURS - other | group True) - (OURS - other | group False), pooled over benches, with a
    bootstrap that resamples videos within each benchmark."""
    strata, per = [], {}
    for b in benches:
        A, R = load(path(OURS, b)), load(path(other, b))
        assert len(A) == SIZE[b] and len(R) == SIZE[b], (b, other, len(A), len(R))
        ks = sorted(k for k in A if k in group[b])
        d = np.array([A[k] - R[k] for k in ks])
        g = np.array([group[b][k] for k in ks])
        _, inv = np.unique([video_of(k) for k in ks], return_inverse=True)
        nv = inv.max() + 1
        strata.append(tuple(np.bincount(inv, weights=w, minlength=nv)
                            for w in (d * g, g.astype(float), d * ~g, (~g).astype(float))))
        per[b] = {"n_multi": int(g.sum()), "n_single": int((~g).sum()),
                  "multi": 100 * float(d[g].mean()) if g.any() else None,
                  "single": 100 * float(d[~g].mean())}

    def stat(parts):
        sm, nm, ss, ns = (sum(p[i].sum() for p in parts) for i in range(4))
        return 100 * (sm / nm), 100 * (ss / ns)

    multi, single = stat(strata)
    rng = np.random.default_rng(SEED)
    draws = []
    for _ in range(DRAWS):
        parts = []
        for s in strata:
            idx = rng.integers(0, len(s[0]), len(s[0]))
            parts.append(tuple(x[idx] for x in s))
        m, s = stat(parts)
        draws.append((m, s, m - s))
    draws = np.array(draws)
    ci = lambda j: [float(v) for v in np.percentile(draws[:, j], [2.5, 97.5])]
    return {"multi": multi, "multi_ci": ci(0), "single": single, "single_ci": ci(1),
            "interaction": multi - single, "interaction_ci": ci(2), "per_bench": per}


def main() -> None:
    group = flags()
    sizes = {b: (int(sum(group[b].values())), len(group[b])) for b in BENCH}
    print("multi-peak questions:", {b: f"{m} of {n}" for b, (m, n) in sizes.items()},
          f"total {sum(m for m, _ in sizes.values())} of {sum(n for _, n in sizes.values())}")
    report = {"definition": {"tau": TAU, "mass": MASS, "gap_s": GAP, "share": SHARE}, "sizes": sizes}
    for name, arm in OTHERS.items():
        r = contrast(arm, group)
        report[name] = r
        tag = "P1" if name == "top-k" else "descriptive"
        print(f"{tag}: CaliFrame minus {name}: multi-peak {r['multi']:+.2f} {np.round(r['multi_ci'], 2)}, "
              f"single-peak {r['single']:+.2f} {np.round(r['single_ci'], 2)}, "
              f"difference {r['interaction']:+.2f} {np.round(r['interaction_ci'], 2)}")
        for b, e in r["per_bench"].items():
            m = "n/a" if e["multi"] is None else f"{e['multi']:+.2f}"
            print(f"    {b:9s} multi {m} (n={e['n_multi']}), single {e['single']:+.2f} (n={e['n_single']})")
    spans = {"cgbench": cg_multispan()}
    r = contrast(OTHERS["top-k"], spans, benches=("cgbench",))
    report["cgbench_annotated_spans"] = r
    print(f"descriptive, CG-Bench annotated spans > {SPAN_GAP} s apart: CaliFrame minus top-k "
          f"{r['multi']:+.2f} (n={r['per_bench']['cgbench']['n_multi']}) vs {r['single']:+.2f}, "
          f"difference {r['interaction']:+.2f} {np.round(r['interaction_ci'], 2)}")
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
