"""NS-P5 Study W (METHODS-LOG 2026-11-12, "Reviewer-requested experiments, wave 1"): shared data for
the CPU analyses W1, W2, W3 and W8. Logged outputs only, nothing is run.

  runs          the run logs in logs/nsp5 (one JSON row per question: correct, in_span = frames on the
                evidence, n_frames); an arm 'a:b' has LVBench stem a and CG-Bench stem b, as in
                nsp5_pooled.path
  E3            the LVBench oracle ladder logged no in_span; its frames are deterministic, so they are
                recomputed from the logged centre and duration (as nsp5_method_figures.py does) and
                checked against S0's rerun of three of its arms, which logged in_span
  Fig. 3 runs   the runs of tools/s_law.realized (accuracy against log coverage C), with its flags
  S variants    the placements of tools/s_law.VARIANTS and its registered P-S2 arms: p(E) per question
  bootstrap     videos resampled within each benchmark, 5,000 draws, seed 0 (the registration's setup)

Imports tools/s_law.py for the variants, the probability and the evidence, rather than re-deriving them.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np

TL = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.abspath(os.path.join(
    TL, "../../docs/research/projects/NS-P5-adaptive-sampling/paper/cvpr2027/tools"))
sys.path.insert(0, TL)
sys.path.insert(0, TOOLS)
import s_law  # noqa: E402  (chdirs to research/TimeLens)
from eval_foveate import kernel_times  # noqa: E402
from nsp5_compare import video_of  # noqa: E402
from nsp5_method_figures import E3_OFFPEAK_SHIFT  # noqa: E402
from nsp5_method_figures import in_span as times_in_span  # noqa: E402
from nsp5_pooled import L, SIZE, path  # noqa: E402

OUT_DIR = os.path.join(TL, "logs/nsp5/analysis")
N = 64
DRAWS, SEED = 5000, 0
BENCHES = ("lvbench", "cgbench")
CG_PRE = "x1_cgbench_"

# The LVBench oracle ladder (E3): its arm names in the harness, and S0, the 400-question rerun of
# three of them with in_span logged (METHODS-LOG 2026-10-25), used to check the recomputation.
E3 = {"e3_uniform": "uniform", "e3_gauss_s15": "gauss_s15", "e3_gauss_s60": "gauss_s60",
      "e3_gauss_s240": "gauss_s240", "e3_offpeak_s60": "offpeak_s60"}
S0 = {"s0_uniform": "e3_uniform", "s0_gauss_s15": "e3_gauss_s15", "s0_offpeak_s60": "e3_offpeak_s60"}


# ----------------------------------------------------------------------------------------------- runs
def file_rows(stem: str) -> dict[str, dict] | None:
    """The rows of logs/nsp5/<stem>.jsonl keyed by question, or None if the file is missing."""
    p = os.path.join(L, f"{stem}.jsonl")
    if not os.path.exists(p):
        return None
    out = {}
    with open(p, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                if "key" in r:
                    out[r["key"]] = r
    return out


def stem_of(arm: str, bench: str) -> str:
    """The file stem of an arm on a benchmark, as nsp5_pooled.path names it."""
    return os.path.basename(path(arm, bench))[:-len(".jsonl")]


def e3_in_span(stem: str, rows: dict[str, dict]) -> dict[str, int]:
    """Frames on the evidence for an E3 run, from its logged centre and duration: the kernel's
    times with the harness's +-0.5 s tolerance (nsp5_method_figures.in_span)."""
    arm = E3[stem]
    return {k: times_in_span(kernel_times(arm, r["center_s"], r["duration_s"], N,
                                          offpeak_shift=E3_OFFPEAK_SHIFT), r["span"])
            for k, r in rows.items()}


def e3_check() -> dict:
    """The recomputed E3 frames on the evidence against logged ones: S0's 400-question reruns of
    three E3 arms (which give identical answers) and logged 64-frame uniform LVBench runs."""
    out = {}
    for s0, e3 in S0.items():
        S, E = file_rows(s0), file_rows(e3)
        rec = e3_in_span(e3, E)
        d = np.array([rec[k] - S[k]["in_span"] for k in S])
        out[f"{e3} vs {s0}"] = {"questions": len(d), "equal": int((d == 0).sum()),
                                "max_abs_diff": int(np.abs(d).max()),
                                "answers_equal": int(sum(S[k]["pred_letter"] == E[k]["pred_letter"] for k in S))}
    E = file_rows("e3_uniform")
    rec = e3_in_span("e3_uniform", E)
    for u in ("h_uniform_q25_32b", "g1_q3vl_uniform", "k_uniform_gemma4_31b"):
        U = file_rows(u)
        d = np.array([rec[k] - U[k]["in_span"] for k in E])
        out[f"e3_uniform vs {u}"] = {"questions": len(d), "equal": int((d == 0).sum()),
                                     "max_abs_diff": int(np.abs(d).max())}
    return out


def run_arrays(stem: str, keys: list[str]) -> tuple[np.ndarray, np.ndarray] | None:
    """(correct, frames on the evidence) per question in `keys` order, NaN where the run has no row.
    E3 runs get recomputed in_span; e3_uniform takes s_law's borrowed one (h_uniform_q25_32b, the
    same uniform frames), so that C matches the paper's Fig. 3."""
    rows = file_rows(stem)
    if rows is None:
        return None
    if stem in E3:
        if stem in s_law.SPAN_FROM:
            src = file_rows(s_law.SPAN_FROM[stem])
            ins = {k: r["in_span"] for k, r in src.items()}
        else:
            ins = e3_in_span(stem, rows)
    else:
        ins = {k: r.get("in_span") for k, r in rows.items()}
    y = np.full(len(keys), np.nan)
    n = np.full(len(keys), np.nan)
    for i, k in enumerate(keys):
        r = rows.get(k)
        if r is not None and ins.get(k) is not None:
            y[i] = float(bool(r["correct"]))
            n[i] = float(ins[k])
    return y, n


def bench_keys(bench: str) -> list[str]:
    """Every question of the benchmark (the method of record's run holds all of them), sorted."""
    rows = file_rows(stem_of("q8_kelly_eps0", bench))
    assert rows is not None and len(rows) == SIZE[bench], bench
    return sorted(rows)


def videos(keys: list[str]) -> list[str]:
    return [video_of(k) for k in keys]


# --------------------------------------------------------------------------------------- Fig. 3 runs
def fig3_arrays(bench: str, keys: list[str]) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """The Fig. 3 runs with their correct and frames-on-evidence arrays ([runs, Q], `keys` order)."""
    runs = fig3_runs(bench)
    arrs = [run_arrays(r["stem"], keys) for r in runs]
    return runs, np.stack([a[0] for a in arrs]), np.stack([a[1] for a in arrs])


def fig3_runs(bench: str) -> list[dict]:
    """The runs of tools/s_law.realized on this benchmark, complete ones only, with its flags:
    `registered` (P-S1/P-S2 arms, drawn on Fig. 3 but outside the CG-Bench fit) and `published`."""
    out = []
    for label, arm in [(v[0], v[1]) for v in s_law.VARIANTS] + s_law.EXTRA_RUNS:
        stem = stem_of(arm.split(":")[0] if bench == "lvbench" else arm, bench)
        rows = file_rows(stem)
        if rows is None or len(rows) != SIZE[bench]:
            continue
        base = arm.split(":")[0]
        out.append({"name": label, "arm": arm, "stem": stem, "published": base in s_law.PUBLISHED,
                    "registered": bench == "cgbench" and base in s_law.REGISTERED})
    return out


# ---------------------------------------------------------------------------------------- S variants
def placements() -> list[dict]:
    """Every placement that defines a probability: s_law.VARIANTS, the three registered P-S2 arms
    (Study E's hedged scorers) and C2's two redone hedged scorers. Each with how s_law computes it."""
    out = [{"name": v[0], "arm": v[1], "scorer": v[2], "transform": v[3], "c": v[4], "share": v[5],
            "group": "variant"} for v in s_law.VARIANTS]
    for arm, (scorer, c, _) in s_law.P_S2.items():
        out.append({"name": s_law.LABELS[arm], "arm": arm, "scorer": scorer, "transform": "softmax",
                    "c": c, "share": 0.5, "group": "p_s2"})
    out += [{"name": "SigLIP tau 1.57 share 0.5", "arm": "cal_a2_kelly_nr", "scorer": "SigLIP",
             "transform": "softmax", "c": 1.57, "share": 0.5, "group": "c2_hedged"},
            {"name": "BLIP tau 18.71 share 0.5", "arm": "cal_kelly_blip_nr", "scorer": "BLIP",
             "transform": "softmax", "c": 18.71, "share": 0.5, "group": "c2_hedged"}]
    return out


def evidence_mass(bench: str, plist: list[dict]) -> tuple[list[str], np.ndarray]:
    """Run keys and p(E) per placement and question ([P, Q]), with s_law's probability and evidence."""
    Z, questions = s_law.evidence(bench)
    keys = [k.split(":", 1)[1] if bench == "lvbench" else k for k, _ in questions]
    pe = np.zeros((len(plist), len(questions)))
    for j, (key, mask) in enumerate(questions):
        cache = {}
        for i, p in enumerate(plist):
            z = cache.setdefault(p["scorer"], Z[p["scorer"]][key].astype(np.float64))
            pe[i, j] = s_law.probability(z, p["transform"], p["c"], p["share"])[mask(len(z))].sum()
    return keys, pe


def predictors(pe: np.ndarray) -> dict[str, np.ndarray]:
    """Per-question predictors of a placement from p(E) alone (registration W3)."""
    return {"S": np.log1p(N * pe),                                   # log(1 + N p(E))
            "Npe": N * pe,                                           # expected frames on the evidence
            "hit": 1.0 - (1.0 - pe) ** N,                            # chance of at least one frame
            "nll": -np.log(np.maximum(pe, s_law.FLOOR))}             # held-out NLL of the evidence


def s_arrays(bench: str) -> dict:
    """Every placement with a probability on one benchmark: question keys (s_law's order), videos,
    p(E) [P, Q], and correct [P, Q] (NaN rows for runs not complete). Flags: `line` (the paper's S
    line: s_law.VARIANTS minus the registered P-S1 arms on CG-Bench), `registered` (P-S1 and P-S2,
    CG-Bench), `base12` (the twelve CG-Bench variants P-S1's line was fitted on, 2026-11-10)."""
    plist = placements()
    keys, pe = evidence_mass(bench, plist)
    acc = np.full(pe.shape, np.nan)
    for i, p in enumerate(plist):
        a = accuracy(p["arm"], bench, keys)
        if a is not None:
            acc[i] = a
        reg = bench == "cgbench" and (p["arm"] in s_law.P_S1 or p["group"] == "p_s2")
        p["registered"] = reg
        p["line"] = p["group"] == "variant" and not reg and a is not None
        p["base12"] = p["line"] and bench == "cgbench" and not p["arm"].startswith("cal_")
    return {"keys": keys, "vid": videos(keys), "plist": plist, "pe": pe, "acc": acc}


def loo_residuals(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Leave-one-out prediction errors of a least-squares line along the last axis (closed form:
    residual / (1 - leverage))."""
    line = fit_lines(x, y)
    res = y - (line["intercept"][..., None] + line["slope"][..., None] * x)
    lev = 1 / x.shape[-1] + (x - line["xbar"][..., None]) ** 2 / line["sxx"][..., None]
    return res / (1 - lev)


def accuracy(arm: str, bench: str, keys: list[str]) -> np.ndarray | None:
    """Correct (0/1) per question for an arm in `keys` order, None unless the run is complete."""
    stem = stem_of(arm.split(":")[0] if bench == "lvbench" else arm, bench)
    rows = file_rows(stem)
    if rows is None or len(rows) != SIZE[bench]:
        return None
    return np.array([float(bool(rows[k]["correct"])) for k in keys])


# ----------------------------------------------------------------------------------------- bootstrap
class VideoBoot:
    """Video bootstrap for one benchmark: `counts[b, v]` is how often video v is drawn in draw b.
    A statistic that is a mean over questions becomes (counts @ per-video sums) / (counts @ per-video
    question counts), so every run and every draw shares the same resampled videos."""

    def __init__(self, vids: list[str], rng: np.random.Generator, draws: int = DRAWS):
        self.uniq, self.inv = np.unique(np.asarray(vids), return_inverse=True)
        V = len(self.uniq)
        idx = rng.integers(0, V, size=(draws, V))
        flat = (idx + np.arange(draws)[:, None] * V).ravel()
        self.counts = np.bincount(flat, minlength=draws * V).reshape(draws, V).astype(float)
        self.V, self.draws = V, draws

    def sums(self, values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Per-video sums of `values` [..., Q] and of the questions present (NaN = absent)."""
        v = np.asarray(values, float)
        present = ~np.isnan(v)
        flat_v = np.where(present, v, 0.0).reshape(-1, v.shape[-1])
        flat_p = present.reshape(-1, v.shape[-1]).astype(float)
        s = np.zeros((flat_v.shape[0], self.V))
        c = np.zeros((flat_v.shape[0], self.V))
        for i in range(flat_v.shape[0]):
            s[i] = np.bincount(self.inv, weights=flat_v[i], minlength=self.V)
            c[i] = np.bincount(self.inv, weights=flat_p[i], minlength=self.V)
        return s.reshape(v.shape[:-1] + (self.V,)), c.reshape(v.shape[:-1] + (self.V,))

    def means(self, values: np.ndarray) -> np.ndarray:
        """Bootstrap means [draws, ...] of `values` [..., Q] over the questions present."""
        s, c = self.sums(values)
        num = self.counts @ s.reshape(-1, self.V).T
        den = self.counts @ c.reshape(-1, self.V).T
        return (num / den).reshape((self.draws,) + s.shape[:-1])


def make_boots(vids_by_bench: dict[str, list[str]]) -> dict[str, VideoBoot]:
    """One VideoBoot per benchmark from a single seed-0 generator, LVBench first. The draws depend
    only on each benchmark's set of videos, not on the order of its questions."""
    rng = np.random.default_rng(SEED)
    return {b: VideoBoot(vids_by_bench[b], rng) for b in BENCHES if b in vids_by_bench}


def boot_for(bench: str, vids: list[str]) -> VideoBoot:
    """The benchmark's VideoBoot exactly as make_boots draws it for both benchmarks (LVBench's
    draws come first from the generator), for questions whose videos are `vids`."""
    if bench == "lvbench":
        return make_boots({"lvbench": vids})["lvbench"]
    return make_boots({"lvbench": videos(bench_keys("lvbench")), bench: vids})[bench]


# --------------------------------------------------------------------------------------------- lines
def fit_lines(x: np.ndarray, y: np.ndarray) -> dict[str, np.ndarray]:
    """Least-squares lines y = a + b x along the last axis (one line per leading index)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    n = x.shape[-1]
    xm, ym = x.mean(-1, keepdims=True), y.mean(-1, keepdims=True)
    sxx = ((x - xm) ** 2).sum(-1)
    sxy = ((x - xm) * (y - ym)).sum(-1)
    syy = ((y - ym) ** 2).sum(-1)
    b = sxy / sxx
    a = ym[..., 0] - b * xm[..., 0]
    sse = np.maximum(syy - b * sxy, 0.0)
    return {"intercept": a, "slope": b, "r2": 1 - sse / syy, "resid_sd": np.sqrt(sse / (n - 2)),
            "xbar": xm[..., 0], "sxx": sxx, "n": n}


def t_quantile(p: float, df: int) -> float:
    """Quantile of Student's t (no scipy in this venv): Simpson's rule on the density, bisection."""
    c = math.exp(math.lgamma((df + 1) / 2) - math.lgamma(df / 2)) / math.sqrt(df * math.pi)

    def cdf(x: float) -> float:
        grid = np.linspace(0.0, x, 4001)
        f = c * (1 + grid ** 2 / df) ** (-(df + 1) / 2)
        h = x / 4000
        return 0.5 + h / 3 * (f[0] + f[-1] + 4 * f[1:-1:2].sum() + 2 * f[2:-1:2].sum())

    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if cdf(mid) < p else (lo, mid)
    return (lo + hi) / 2


def prediction_interval(line: dict, x0, level: float = 0.95) -> tuple:
    """Prediction and t-based prediction interval of a fitted line at x0 (as the registered P-S1)."""
    t = t_quantile(0.5 + level / 2, int(line["n"]) - 2)
    pred = line["intercept"] + line["slope"] * x0
    half = t * line["resid_sd"] * np.sqrt(1 + 1 / line["n"] + (x0 - line["xbar"]) ** 2 / line["sxx"])
    return pred, pred - half, pred + half


def ci(samples: np.ndarray, axis: int = 0) -> list:
    """95% percentile interval."""
    lo, hi = np.percentile(samples, [2.5, 97.5], axis=axis)
    return [lo.tolist() if np.ndim(lo) else float(lo), hi.tolist() if np.ndim(hi) else float(hi)]


def write_json(name: str, obj: dict) -> str:
    os.makedirs(OUT_DIR, exist_ok=True)
    out = os.path.join(OUT_DIR, name)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1)
    return out
