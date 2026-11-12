"""NS-P5 Study W1 (METHODS-LOG 2026-11-12, wave 1, Study W): the per-question log relation.

Registered: correctness on log(1 + n_E) with question fixed effects (logistic and linear), pooled
over every 64-frame Qwen2.5-VL-7B run on LVBench and CG-Bench; forms log(1 + n), n, 1[n >= 1],
sqrt(n) and free bins (0, 1, 2-3, 4-7, 8-15, 16-31, 32+), compared by video-blocked cross-validated
log loss. Reading: log or a concave form wins and the effect stays positive among n >= 1.

Choices where the registration leaves room (the most standard reading of each):
  logistic   conditional (fixed-effects) logit: the question effects are conditioned out, so the
             held-out score of a question needs no intercept fitted on it. CV log loss = the
             held-out conditional negative log-likelihood (nats). Robustness: the same folds scored
             with an ordinary logit whose question intercept is refitted on the held-out question
             (binary log loss per observation).
  linear     linear probability model, within-question (question-demeaned) least squares; its
             held-out squared error is reported beside.
  folds      5, by video, stratified by benchmark (each benchmark's videos shuffled with
             random.Random(0), fold = position mod 5: prior_encode.nll_cv's scheme).
  runs       every logged LVBench / CG-Bench run with 64 frames and Qwen2.5-VL-7B answering, one row
             per question; superseded runs whose rows were copied into a corrected rerun (the "_nr"
             refill fixes) and the router run whose rows copy two other runs are counted once; runs
             with no frames-on-evidence log that cannot be recomputed are left out (listed).
  E3         the oracle ladder's frames on the evidence are recomputed (nsp5_w_data.e3_in_span).
  intervals  videos resampled within each benchmark, 5,000 draws, seed 0. One-parameter logits are
             re-solved exactly in every draw; the free-bins logit takes one Newton step from the
             full-sample fit (the one-step bootstrap); the linear model is exact.
  winner     the form with the lowest CV log loss. "Concave" = log(1 + n) or sqrt(n), or the free
             bins if they rise and their gain per frame falls from bin to bin.
  n >= 1     the log(1 + n) slope refitted on the observations with at least one frame on the
             evidence (question fixed effects within them).

    env -u VIRTUAL_ENV OMP_NUM_THREADS=8 .venv/bin/python -u nsp5_w1_law.py

writes logs/nsp5/analysis/w1_law.json and prints the summary.
"""
from __future__ import annotations

import os
import random
import time

import numpy as np

import nsp5_w_data as W

FORMS = ("log", "n", "thr", "sqrt", "bins")
BINS = ((0, 0), (1, 1), (2, 3), (4, 7), (8, 15), (16, 31), (32, 10 ** 9))
BIN_NAMES = ("0", "1", "2-3", "4-7", "8-15", "16-31", "32+")
FOLDS = 5

# Run selection (see the docstring). Prefixes and markers on the run's stem (CG-Bench: after x1_cgbench_).
OTHER_ANSWERER = ("g1_", "g2_", "h_", "k_", "l_")        # Qwen3-VL-8B, Qwen3.5-4B, 32B/72B, Gemma, PLM
OTHER_BUDGET = ("n16_", "n32_", "n128_", "gb32_", "pub32_")
NOT_A_RUN = ("e1", "e2", "s0_", "gate_", "e3_smoke", "e3_timing", "e4_smoke", "topk_smoke")
# Study W analyses existing outputs: the GPU arms registered with it (S, T, U, V) are not in its pool
WAVE1 = ("fresh_", "t_", "n128_t_", "u_", "v_dose")
PLACEBO_CG = 2683            # CG-Bench's same-video placebos skip its 317 single-question videos
SUPERSEDED = {"q8_kelly": "q8_kelly_nr", "a2_kelly": "a2_kelly_nr", "kelly_clip": "kelly_clip_nr",
              "kelly_blip": "kelly_blip_nr", "q8_kelly_placebo": "q8_kelly_placebo_nr",
              "ge1_opt_topk": "q8opt_topk_nr", "q8opt_topk": "q8opt_topk_nr"}
COPIES = {"q8_switch": "rows copy q8_kelly_nr / q8opt_topk_nr (the router picks one per question)"}


def classify(stem: str) -> tuple[str, str]:
    """(benchmark or '', reason excluded or '') for a log file stem."""
    if stem.startswith("x1_"):
        if not stem.startswith(W.CG_PRE):
            return "", "other benchmark"
        bench, s = "cgbench", stem[len(W.CG_PRE):]
    else:
        bench, s = "lvbench", stem
    if s.startswith(NOT_A_RUN) or ".shard" in s:
        return bench, "not a full run (smoke, timing, other dataset, S0 subset or shard)"
    if s.startswith(WAVE1):
        return bench, "registered with Study W (wave 1 GPU arm), not an existing output"
    if s.startswith(OTHER_ANSWERER) or "_q3" in s:
        return bench, "other answer model"
    if s.startswith(OTHER_BUDGET):
        return bench, "other frame budget or resolution"
    if s in SUPERSEDED:
        return bench, f"superseded by {SUPERSEDED[s]} (rows copied into it)"
    if s in COPIES:
        return bench, COPIES[s]
    return bench, ""


def select_runs() -> tuple[dict[str, list[str]], dict[str, str]]:
    stems = sorted(n[:-6] for n in os.listdir(W.L) if n.endswith(".jsonl"))
    keep, why = {"lvbench": [], "cgbench": []}, {}
    keys = {b: set(W.bench_keys(b)) for b in W.BENCHES}
    for stem in stems:
        bench, reason = classify(stem)
        if not bench:
            continue
        if not reason:
            rows = W.file_rows(stem)
            full = len(rows or ()) == W.SIZE[bench] or ("placebo" in stem and len(rows or ()) == PLACEBO_CG)
            if not rows:
                reason = "empty"
            elif not full or not set(rows) <= keys[bench]:
                reason = "incomplete or not the benchmark's question set"
            elif any(r.get("n_frames") != W.N for r in rows.values()):
                reason = "rows with other than 64 frames"
            elif stem not in W.E3 and any(r.get("in_span") is None for r in rows.values()):
                reason = "no frames-on-evidence log (not recomputable from logged data)"
        if reason:
            why[stem] = reason
        else:
            keep[bench].append(stem)
    return keep, why


# ------------------------------------------------------------------------------------------ the data
def build(runs: dict[str, list[str]]) -> dict:
    """Questions x runs arrays, pooled over the benchmarks; absent rows are masked."""
    T = max(len(v) for v in runs.values())
    parts = []
    for bench in W.BENCHES:
        keys = W.bench_keys(bench)
        Y = np.zeros((len(keys), T))
        Nn = np.zeros((len(keys), T))
        M = np.zeros((len(keys), T), bool)
        for t, stem in enumerate(runs[bench]):
            y, n = W.run_arrays(stem, keys)
            ok = ~np.isnan(y)
            Y[ok, t], Nn[ok, t], M[ok, t] = y[ok], n[ok], True
        parts.append({"bench": np.array([bench] * len(keys)), "keys": keys, "vid": W.videos(keys),
                      "Y": Y, "N": Nn, "M": M})
    return {"bench": np.concatenate([p["bench"] for p in parts]),
            "keys": sum((p["keys"] for p in parts), []), "vid": sum((p["vid"] for p in parts), []),
            "Y": np.concatenate([p["Y"] for p in parts]), "N": np.concatenate([p["N"] for p in parts]),
            "M": np.concatenate([p["M"] for p in parts])}


def features(n: np.ndarray, form: str) -> np.ndarray:
    """[..., P] features of the frames on the evidence for a form."""
    if form == "log":
        return np.log1p(n)[..., None]
    if form == "n":
        return n[..., None].astype(float)
    if form == "thr":
        return (n >= 1)[..., None].astype(float)
    if form == "sqrt":
        return np.sqrt(n)[..., None]
    return np.stack([(n >= lo) & (n <= hi) for lo, hi in BINS[1:]], -1).astype(float)


def centred(X: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Features minus their mean over each question's present runs (absent entries zero)."""
    m = M[..., None].astype(float)
    mean = (X * m).sum(1, keepdims=True) / np.maximum(m.sum(1, keepdims=True), 1)
    return (X - mean) * m


def folds(vid: list[str], bench: np.ndarray) -> np.ndarray:
    """Video-blocked folds, stratified by benchmark (prior_encode.nll_cv's shuffling)."""
    f = np.zeros(len(vid), int)
    for b in W.BENCHES:
        sel = np.flatnonzero(bench == b)
        u = sorted({vid[i] for i in sel})
        random.Random(0).shuffle(u)
        fold = {v: i % FOLDS for i, v in enumerate(u)}
        f[sel] = [fold[vid[i]] for i in sel]
    return f


# --------------------------------------------------------------------------------- conditional logit
def cl_terms(beta: np.ndarray, Xc: np.ndarray, Y: np.ndarray, M: np.ndarray, hess: bool = True):
    """Per-question conditional log-likelihood, score and Hessian. The denominator, a sum over every
    set of k_q runs, is the elementary symmetric polynomial of exp(eta) built run by run (with its
    first and second derivatives); features are centred within question, which leaves it unchanged."""
    Q, T, P = Xc.shape
    eta = Xc @ beta
    w = np.where(M, np.exp(eta), 0.0)
    k = (Y * M).sum(1).astype(int)
    E = np.zeros((Q, T + 1))
    E[:, 0] = 1.0
    D = np.zeros((Q, T + 1, P))
    H = np.zeros((Q, T + 1, P, P)) if hess else None
    for t in range(T):
        wt, xt = w[:, t], Xc[:, t, :]
        dw = wt[:, None] * xt
        Ep, Dp = E[:, :-1].copy(), D[:, :-1].copy()
        if hess:
            Hp = H[:, :-1].copy()
            H[:, 1:] += ((dw[:, :, None] * xt[:, None, :])[:, None] * Ep[:, :, None, None]
                         + dw[:, None, :, None] * Dp[:, :, None, :]
                         + Dp[:, :, :, None] * dw[:, None, None, :] + wt[:, None, None, None] * Hp)
        D[:, 1:] += dw[:, None, :] * Ep[:, :, None] + wt[:, None, None] * Dp
        E[:, 1:] += wt[:, None] * Ep
    q = np.arange(Q)
    Ek, Dk = E[q, k], D[q, k]
    ym = Y * M
    ll = (ym * eta).sum(1) - np.log(Ek)
    g = (ym[:, :, None] * Xc).sum(1) - Dk / Ek[:, None]
    h = None
    if hess:
        h = -(H[q, k] / Ek[:, None, None] - Dk[:, :, None] * Dk[:, None, :] / Ek[:, None, None] ** 2)
    return ll, g, h


def informative(Y: np.ndarray, M: np.ndarray) -> np.ndarray:
    k, T = (Y * M).sum(1), M.sum(1)
    return (k > 0) & (k < T)


def cl_fit(Xc: np.ndarray, Y: np.ndarray, M: np.ndarray, tol: float = 1e-10,
           start: np.ndarray | None = None) -> np.ndarray:
    """Newton's method with step halving on the (concave) conditional log-likelihood; `start`
    (e.g. the full-sample fit for a CV fold) only saves iterations."""
    beta = np.zeros(Xc.shape[-1]) if start is None else np.array(start, float)
    ll, g, h = cl_terms(beta, Xc, Y, M)
    cur = ll.sum()
    for _ in range(50):
        step = np.linalg.solve(h.sum(0), g.sum(0))
        lam = 1.0
        while True:
            nb = beta - lam * step
            ll2, g2, h2 = cl_terms(nb, Xc, Y, M)
            if ll2.sum() >= cur - 1e-12 or lam < 1e-6:
                break
            lam /= 2
        beta, g, h = nb, g2, h2
        done = abs(ll2.sum() - cur) < tol
        cur = ll2.sum()
        if done:
            break
    return beta


def refit_intercept_loss(eta: np.ndarray, Y: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Binary log loss summed per question with the question's intercept refitted on its own rows
    (the robustness score; informative questions only, others 0)."""
    m = M.astype(float)
    ybar = (Y * m).sum(1) / m.sum(1)
    inf = (ybar > 0) & (ybar < 1)
    a = np.zeros(len(Y))
    a[inf] = np.log(ybar[inf] / (1 - ybar[inf])) - (eta * m).sum(1)[inf] / m.sum(1)[inf]
    for _ in range(100):
        p = 1 / (1 + np.exp(-(a[:, None] + eta)))
        grad = ((Y - p) * m).sum(1)
        curv = (p * (1 - p) * m).sum(1)
        a = np.where(inf, a + grad / np.maximum(curv, 1e-12), a)
    p = np.clip(1 / (1 + np.exp(-(a[:, None] + eta))), 1e-15, 1 - 1e-15)
    loss = -((Y * np.log(p) + (1 - Y) * np.log(1 - p)) * m).sum(1)
    return np.where(inf, loss, 0.0)


# ------------------------------------------------------------------------------------- linear model
def lpm_stats(Xc: np.ndarray, Y: np.ndarray, M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-question within sums: X'X [Q, P, P] and X'y [Q, P] on question-demeaned data."""
    m = M.astype(float)
    yc = (Y - (Y * m).sum(1, keepdims=True) / np.maximum(m.sum(1, keepdims=True), 1)) * m
    return np.einsum("qtp,qtr->qpr", Xc, Xc), np.einsum("qtp,qt->qp", Xc, yc)


def lpm_cv_sse(Xc: np.ndarray, Y: np.ndarray, M: np.ndarray, beta: np.ndarray) -> np.ndarray:
    """Held-out squared error per question: prediction = question mean + centred features x beta."""
    m = M.astype(float)
    ybar = (Y * m).sum(1, keepdims=True) / np.maximum(m.sum(1, keepdims=True), 1)
    return (((Y - ybar - Xc @ beta) ** 2) * m).sum(1)


# -------------------------------------------------------------------------------------- the analysis
def video_counts(data: dict, sel: np.ndarray, boots: dict) -> np.ndarray:
    """Bootstrap weight of each selected question in each draw ([draws, Q]) via its video's count."""
    cols = []
    for b in W.BENCHES:
        bs = boots[b]
        lookup = {v: i for i, v in enumerate(bs.uniq)}
        cols.append((b, bs, lookup))
    wq = np.zeros((W.DRAWS, sel.sum()))
    vids = [data["vid"][i] for i in np.flatnonzero(sel)]
    benches = data["bench"][sel]
    for b, bs, lookup in cols:
        idx = np.flatnonzero(benches == b)
        if len(idx):
            wq[:, idx] = bs.counts[:, [lookup[vids[i]] for i in idx]]
    return wq


def one_d_boot(Xc: np.ndarray, Y: np.ndarray, M: np.ndarray, beta: float, se: float,
               wq: np.ndarray) -> np.ndarray:
    """Exact bootstrap of a one-parameter conditional logit: the weighted score's root per draw,
    from the per-question score on a grid around the estimate (linear interpolation)."""
    for width in (12, 30, 80):
        grid = beta + se * np.linspace(-width, width, 8 * width + 1)
        G = np.stack([cl_terms(np.array([b]), Xc, Y, M, hess=False)[1][:, 0] for b in grid], 1)
        S = wq @ G                                       # [draws, grid], decreasing in beta
        if (S[:, 0] > 0).all() and (S[:, -1] < 0).all():
            j = np.argmax(S < 0, axis=1)                 # first grid point past the root
            s0, s1 = S[np.arange(len(S)), j - 1], S[np.arange(len(S)), j]
            return grid[j - 1] + (grid[j] - grid[j - 1]) * s0 / (s0 - s1)
    raise RuntimeError("bootstrap root outside the grid")


def analyse(data: dict, sel_q: np.ndarray, obs_mask: np.ndarray, boots: dict, label: str,
            forms=FORMS, cv: bool = True) -> dict:
    """Every form on the selected questions and observations: full-sample fits with intervals,
    and the video-blocked CV comparison."""
    Y, Nn = data["Y"][sel_q], data["N"][sel_q]
    M = data["M"][sel_q] & obs_mask[sel_q]
    inf = informative(Y, M)
    Y, Nn, M = Y[inf], Nn[inf], M[inf]
    q_idx = np.flatnonzero(sel_q)[inf]
    sel = np.zeros(len(data["Y"]), bool)
    sel[q_idx] = True
    wq = video_counts(data, sel, boots)
    fold = folds(data["vid"], data["bench"])[q_idx]
    n_obs = int(M.sum())
    out = {"label": label, "questions": int(len(Y)), "observations": n_obs, "forms": {}}
    held = {}
    for form in forms:
        t0 = time.time()
        Xc = centred(features(Nn, form), M)
        beta = cl_fit(Xc, Y, M)
        ll, g, h = cl_terms(beta, Xc, Y, M)
        Hs = h.sum(0)
        se_naive = np.sqrt(np.diag(np.linalg.inv(-Hs)))
        if Xc.shape[-1] == 1:
            draws = one_d_boot(Xc, Y, M, beta[0], se_naive[0], wq)[:, None]
        else:   # one-step bootstrap: beta - (sum w h)^-1 (sum w g), at the full-sample estimate
            Hb = np.einsum("bq,qij->bij", wq, h)
            gb = wq @ g
            draws = beta[None] - np.linalg.solve(Hb, gb[..., None])[..., 0]
        sxx, sxy = lpm_stats(Xc, Y, M)
        lp = np.linalg.solve(sxx.sum(0), sxy.sum(0))
        lp_draws = np.linalg.solve(np.einsum("bq,qij->bij", wq, sxx), (wq @ sxy)[..., None])[..., 0]
        f = {"cl_beta": beta.tolist(), "cl_se_naive": se_naive.tolist(),
             "cl_ci": W.ci(draws), "cl_loglik": float(ll.sum()),
             "lpm_points": (100 * lp).tolist(), "lpm_ci": W.ci(100 * lp_draws)}
        if form == "log":
            f["cl_odds_ratio_per_doubling"] = float(np.exp(beta[0] * np.log(2)))
            f["lpm_points_per_doubling"] = float(100 * lp[0] * np.log(2))
            f["lpm_points_per_doubling_ci"] = W.ci(100 * lp_draws[:, 0] * np.log(2))
        if form == "bins":
            f["bins"] = list(BIN_NAMES[1:])
            mean_n = [float(Nn[M & (Nn >= lo) & (Nn <= hi)].mean()) for lo, hi in BINS]
            f["mean_n_in_bin"] = mean_n
            f["cl_contrast_vs_bin1"] = {BIN_NAMES[j + 1]: [float(beta[j] - beta[0]),
                                                            W.ci(draws[:, j] - draws[:, 0])]
                                        for j in range(1, len(beta))}
            f["lpm_contrast_vs_bin1"] = {BIN_NAMES[j + 1]: [float(100 * (lp[j] - lp[0])),
                                                             W.ci(100 * (lp_draws[:, j] - lp_draws[:, 0]))]
                                         for j in range(1, len(lp))}
            f["shape"] = bins_shape(beta, mean_n)
            f["shape_lpm"] = bins_shape(lp, mean_n)
        if cv:
            cv_cl = np.zeros(len(Y))
            cv_un = np.zeros(len(Y))
            cv_sq = np.zeros(len(Y))
            for k in range(FOLDS):
                tr, te = fold != k, fold == k
                b_k = cl_fit(Xc[tr], Y[tr], M[tr], start=beta)
                cv_cl[te] = -cl_terms(b_k, Xc[te], Y[te], M[te], hess=False)[0]
                cv_un[te] = refit_intercept_loss(features(Nn[te], form) @ b_k, Y[te], M[te])
                sxx_k, sxy_k = lpm_stats(Xc[tr], Y[tr], M[tr])
                cv_sq[te] = lpm_cv_sse(Xc[te], Y[te], M[te], np.linalg.solve(sxx_k.sum(0), sxy_k.sum(0)))
            held[form] = (cv_cl, cv_un, cv_sq)
            f["cv_cond_nll_per_obs"] = float(cv_cl.sum() / n_obs)
            f["cv_refit_intercept_logloss_per_obs"] = float(cv_un.sum() / n_obs)
            f["cv_lpm_mse"] = float(cv_sq.sum() / n_obs)
        out["forms"][form] = f
        print(f"  [{label}] {form:5s} beta {np.round(beta, 4).tolist()}  ({time.time() - t0:.0f} s)", flush=True)
    if cv:
        out["cv"] = cv_summary(held, wq, M.sum(1).astype(float))
    return out


def bins_shape(coef: np.ndarray, mean_n: list) -> dict:
    """Does the free-bins curve rise, and does its gain per frame fall from bin to bin (concave)?"""
    c = np.concatenate([[0.0], np.asarray(coef, float)])
    x = np.asarray(mean_n)
    per_frame = np.diff(c) / np.diff(x)
    return {"rises": bool(np.all(np.diff(c) > 0)), "gain_per_frame": per_frame.tolist(),
            "concave": bool(np.all(np.diff(per_frame) < 0))}


def cv_summary(held: dict, wq: np.ndarray, obs_q: np.ndarray) -> dict:
    """CV losses per observation for each form, and each form minus log(1 + n) with its
    video-bootstrap interval (positive: worse than the log)."""
    out, n_obs = {}, obs_q.sum()
    for j, score in enumerate(("cond_nll", "refit_intercept_logloss", "lpm_mse")):
        base = held["log"][j]
        rows = {}
        for form, v in held.items():
            d = v[j] - base
            boot = (wq @ d) / (wq @ obs_q)
            rows[form] = {"per_obs": float(v[j].sum() / n_obs),
                          "minus_log_per_obs": float(d.sum() / n_obs), "minus_log_ci": W.ci(boot)}
        best = min(rows, key=lambda f: rows[f]["per_obs"])
        out[score] = {"forms": rows, "winner": best}
    return out


def reading(res: dict) -> dict:
    """The registered reading on one analysis: the CV winner is log or concave, and the n >= 1
    slope's interval lies above zero (logit and linear)."""
    win = res["all"]["cv"]["cond_nll"]["winner"]
    shape = res["all"]["forms"]["bins"]["shape"]
    concave = win in ("log", "sqrt") or (win == "bins" and shape["rises"] and shape["concave"])
    pos = res["n_ge_1"]["forms"]["log"]
    positive = pos["cl_ci"][0][0] > 0 and pos["lpm_ci"][0][0] > 0
    return {"cv_winner": win, "winner_log_or_concave": bool(concave),
            "positive_among_n_ge_1": bool(positive), "holds": bool(concave and positive)}


def main() -> None:
    runs, excluded = select_runs()
    print({b: len(v) for b, v in runs.items()}, "runs;", len(excluded), "logs excluded")
    data = build(runs)
    boots = W.make_boots({b: [v for v, bb in zip(data["vid"], data["bench"]) if bb == b]
                          for b in W.BENCHES})
    every = np.ones(len(data["Y"]), bool)
    ge1 = data["N"] >= 1
    res = {"runs": runs, "excluded": excluded,
           "rows": {b: int(data["M"][data["bench"] == b].sum()) for b in W.BENCHES}}
    for scope in ("pooled", "lvbench", "cgbench"):
        sel = every if scope == "pooled" else data["bench"] == scope
        r = {"all": analyse(data, sel, np.ones_like(data["M"]), boots, f"{scope} all"),
             "n_ge_1": analyse(data, sel, ge1, boots, f"{scope} n>=1", forms=("log",), cv=False)}
        r["reading"] = reading(r)
        res[scope] = r
        print(scope, r["reading"], flush=True)
    # sensitivity: only the runs of Fig. 3 (tools/s_law.realized), every one of which is in the pool
    fig3 = {b: [r["stem"] for r in W.fig3_runs(b)] for b in W.BENCHES}
    keep = np.zeros(data["M"].shape, bool)
    for i, b in enumerate(data["bench"]):
        keep[i, :len(runs[b])] = [s in fig3[b] for s in runs[b]]
    r = {"all": analyse(data, every, keep, boots, "fig3 runs", forms=("log", "n", "thr", "sqrt", "bins")),
         "n_ge_1": analyse(data, every, keep & ge1, boots, "fig3 runs n>=1", forms=("log",), cv=False)}
    r["reading"] = reading(r)
    r["runs"] = fig3
    res["fig3_runs_only"] = r
    print("fig3 runs only", r["reading"], flush=True)
    # descriptive: raw accuracy by bin of frames on the evidence, pooled over runs and questions
    Y, Nn, M = data["Y"], data["N"], data["M"]
    res["raw_accuracy_by_bin"] = {
        b: {name: [float(100 * Y[(data["bench"] == b)[:, None] & M & (Nn >= lo) & (Nn <= hi)].mean()),
                   int(((data["bench"] == b)[:, None] & M & (Nn >= lo) & (Nn <= hi)).sum())]
            for name, (lo, hi) in zip(BIN_NAMES, BINS)} for b in W.BENCHES}
    print("wrote", W.write_json("w1_law.json", res))


if __name__ == "__main__":
    main()
