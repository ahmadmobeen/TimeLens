"""NS-P5: the numbers behind METHOD.md and the deck's method slides, from logged data only.

Method M scores every second of a video with Qwen3-VL-Embedding-8B under each answer option,
keeps the best option per second, turns that curve into a probability at temperature 3.58,
and places 64 frames at the quantiles of that probability mixed 50/50 with uniform (Kelly).
This script recomputes, for LVBench:

  tau       the temperature fit (nll_cv) with its whole NLL-vs-tau curve and fold choices
  loc       localization of the stem and max-option curves (top second within 60 s, miss)
  example   one question end to end: per-option curves, probability, placements, answers
  coverage  frames placed on the evidence by uniform, top-k and Kelly, and accuracy
  e3        the oracle ladder's accuracy against its frames on the evidence
  qtype     uniform, top-k and M by LVBench question type
  options   which option's curve carries the top second (needs --peropt first)
  spacing   how close two Kelly frames come (duplicate native frames are re-filled)

  env -u VIRTUAL_ENV PYTHONPATH=. .venv/bin/python -u nsp5_method_figures.py --peropt   # GPU, once
  env -u VIRTUAL_ENV PYTHONPATH=. .venv/bin/python -u nsp5_method_figures.py            # CPU

Writes logs/nsp5/method_figures.json. Nothing here is fit or tuned: tau and eps are M's frozen
constants, and every accuracy comes from the logged runs.
"""
from __future__ import annotations

import argparse
import json
import os
import random

import numpy as np

from eval_foveate import kernel_times
from eval_lvbench_foveate import curve_times
from t1_data import lvbench_items, span_seconds

PRIOR = "data/t1/priors/qwen3vl_emb_8b"
LOGS = "logs/nsp5"
OUT = os.path.join(LOGS, "method_figures.json")
TAU, EPS, N = 3.58, 0.5, 64
EXAMPLE = "k2FIFQIYBvA>>>4731"
TAUS = np.exp(np.linspace(np.log(0.3), np.log(30), 40))
E3_OFFPEAK_SHIFT = 600.0                     # eval_lvbench_foveate's --offpeak_shift default


def rows(name: str) -> dict[str, dict]:
    out = {}
    with open(os.path.join(LOGS, f"{name}.jsonl"), encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                r = json.loads(line)
                out[r["key"]] = r
    return out


def evidence_mask(span: list[float], T: int) -> np.ndarray:
    lo, hi = span_seconds(span[0], span[1], T)
    m = np.zeros(T, bool)
    m[lo:hi] = True
    return m


def nll(z: np.ndarray, m: np.ndarray, t: float) -> float:
    """-log of the softmax(z / t) mass on the evidence seconds (prior_encode.nll_cv's)."""
    x = z / t
    mx, ms = x.max(), x[m].max()
    return float(mx + np.log(np.exp(x - mx).sum()) - ms - np.log(np.exp(x[m] - ms).sum()))


def tau_fit(curves: list[np.ndarray], masks: list[np.ndarray], vids: list[str]) -> dict:
    """prior_encode.nll_cv, keeping the whole table so the figure can show the curve."""
    tab = np.array([[nll(z, m, t) for t in TAUS] for z, m in zip(curves, masks)])
    u = sorted(set(vids))
    random.Random(0).shuffle(u)
    fold = {v: i % 5 for i, v in enumerate(u)}
    f = np.array([fold[v] for v in vids])
    held, chosen = np.zeros(len(curves)), []
    for k in range(5):
        j = int(np.argmin(tab[f != k].mean(0)))
        held[f == k] = tab[f == k, j]
        chosen.append(float(TAUS[j]))
    uni = float(np.mean([np.log(len(m) / m.sum()) for m in masks]))
    return {"taus": TAUS.round(4).tolist(), "nll_mean": tab.mean(0).round(4).tolist(),
            "fold_tau": [round(t, 4) for t in chosen], "tau": round(float(np.median(chosen)), 4),
            "heldout_nll": round(float(held.mean()), 4), "uniform_nll": round(uni, 4),
            "nats_vs_uniform": round(uni - float(held.mean()), 4)}


def localization(curves: list[np.ndarray], masks: list[np.ndarray]) -> dict:
    d = []
    for z, m in zip(curves, masks):
        t = int(np.argmax(z))
        ev = np.flatnonzero(m)
        d.append(0.0 if m[t] else float(np.abs(ev - t).min()))
    d = np.array(d)
    return {"within_60s": round(float(np.mean(d <= 60)), 4), "median_miss_s": float(np.median(d))}


def peaks(p: np.ndarray, k: int = 6, sep: int = 30, half: int = 15) -> list[dict]:
    out: list[int] = []
    for s in np.argsort(-p):
        if all(abs(int(s) - q) > sep for q in out):
            out.append(int(s))
        if len(out) == k:
            break
    return [{"second": s, "mass_pm15s": round(float(p[max(0, s - half):s + half + 1].sum()), 4)}
            for s in out]


def in_span(times: list[float], span: list[float]) -> int:
    """Frames on the evidence, with the harness's +-0.5 s tolerance."""
    a, b = span
    return int(sum(a - 0.5 <= t <= b + 0.5 for t in times))


def compute_peropt() -> None:
    """GPU: per-option 8B curves for every LVBench question (curves_maxopt keeps the max only)."""
    import torch
    from prior_encode import (QUERY_INSTRUCTION, load_qwen3_emb, qwen3_conversation,
                              qwen3_embed, qwen3_inputs, query_forms)
    img = np.load(os.path.join(PRIOR, "lvbench_img.npz"))
    model, proc = load_qwen3_emb("Qwen/Qwen3-VL-Embedding-8B")
    out = {}
    for it in lvbench_items():
        stem, opts = query_forms(it["question"])
        convs = [qwen3_conversation(QUERY_INSTRUCTION, text=f"{stem} {o}") for o in opts]
        with torch.no_grad():
            emb = qwen3_embed(model, dict(qwen3_inputs(proc, convs))).cpu().numpy()
        x = img[it["key"]].astype(np.float32)
        out[f"lvbench:{it['key']}>>>{it['uid']}"] = (emb @ x.T * 100.0).astype(np.float16)
    np.savez(os.path.join(PRIOR, "curves_peropt.npz"), **out)
    print(f"per-option curves for {len(out)} questions -> {PRIOR}/curves_peropt.npz", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--peropt", action="store_true", help="GPU: write curves_peropt.npz and stop")
    args = ap.parse_args()
    if args.peropt:
        compute_peropt()
        return

    items = {f"{it['key']}>>>{it['uid']}": it for it in lvbench_items()}
    mo = np.load(os.path.join(PRIOR, "curves_maxopt.npz"))
    st = np.load(os.path.join(PRIOR, "curves_stem.npz"))
    keys = [k for k in items if f"lvbench:{k}" in mo.files]
    zo = {k: mo[f"lvbench:{k}"].astype(np.float64) for k in keys}
    zs = {k: st[f"lvbench:{k}"].astype(np.float64) for k in keys}
    masks = {k: evidence_mask(items[k]["span"], len(zo[k])) for k in keys}
    vids = [items[k]["key"] for k in keys]
    res: dict = {"n_questions": len(keys), "tau_used": TAU, "eps": EPS, "n_frames": N}

    res["tau"] = {"maxopt": tau_fit([zo[k] for k in keys], [masks[k] for k in keys], vids),
                  "stem": tau_fit([zs[k] for k in keys], [masks[k] for k in keys], vids)}
    res["loc"] = {"maxopt": localization([zo[k] for k in keys], [masks[k] for k in keys]),
                  "stem": localization([zs[k] for k in keys], [masks[k] for k in keys])}
    print("tau:", {q: (v["tau"], v["fold_tau"], v["heldout_nll"]) for q, v in res["tau"].items()},
          flush=True)

    arms = ["e3_uniform", "b1_topk_siglip", "ge1_stem_topk", "ge1_opt_topk", "q8_kelly",
            "e3_gauss_s240", "e3_gauss_s60", "e3_gauss_s15", "e3_offpeak_s60"]
    R = {a: rows(a) for a in arms}
    dur = {k: R["q8_kelly"][k]["duration_s"] for k in keys}      # the harness's own duration

    # frames on the evidence: uniform and the E3 kernels are deterministic, so recompute them
    def placed(arm: str, k: str) -> list[float]:
        it = items[k]
        if arm == "e3_uniform":
            return kernel_times("uniform", it["center"], dur[k], N)
        if arm.startswith("e3_"):
            return kernel_times(arm[3:], min(max(it["center"], 0.0), dur[k]), dur[k], N,
                                offpeak_shift=E3_OFFPEAK_SHIFT)
        raise KeyError(arm)

    cov = {}
    for a in ["e3_uniform", "ge1_stem_topk", "ge1_opt_topk", "q8_kelly"]:
        ins = np.array([in_span(placed(a, k), items[k]["span"]) if a == "e3_uniform"
                        else R[a][k]["in_span"] for k in keys])
        cor = np.array([R[a][k]["correct"] for k in keys], float)
        cov[a] = {"acc": round(cor.mean() * 100, 2), "mean_frames_on_evidence": round(ins.mean(), 2),
                  "median_frames_on_evidence": float(np.median(ins)),
                  "share_zero": round(float(np.mean(ins == 0)), 4),
                  "acc_given_zero": round(cor[ins == 0].mean() * 100, 2),
                  "acc_given_some": round(cor[ins > 0].mean() * 100, 2)}
    length = np.array([items[k]["span"][1] - items[k]["span"][0] for k in keys])
    bins = {"instant (<1 s)": length < 1, "1 to 30 s": (length >= 1) & (length <= 30),
            "30 s to 5 min": (length > 30) & (length <= 300), "over 5 min": length > 300}
    cov["by_evidence_length"] = {
        lab: {"n": int(m.sum()), **{a: round(float(np.mean([R[a][k]["correct"] for k, mm
                                                             in zip(keys, m) if mm])) * 100, 2)
                                    for a in ["e3_uniform", "ge1_opt_topk", "q8_kelly"]}}
        for lab, m in bins.items()}
    zt = [k for k in keys if R["ge1_opt_topk"][k]["in_span"] == 0]
    zk = [k for k in keys if R["q8_kelly"][k]["in_span"] == 0]
    only_k = [k for k in zt if R["q8_kelly"][k]["in_span"] > 0]
    only_t = [k for k in zk if R["ge1_opt_topk"][k]["in_span"] > 0]
    acc = lambda a, ks: round(float(np.mean([R[a][k]["correct"] for k in ks])) * 100, 2)
    cov["hedge"] = {"topk_zero_kelly_some": {"n": len(only_k), "kelly": acc("q8_kelly", only_k),
                                             "topk": acc("ge1_opt_topk", only_k)},
                    "kelly_zero_topk_some": {"n": len(only_t), "kelly": acc("q8_kelly", only_t),
                                             "topk": acc("ge1_opt_topk", only_t)}}
    res["coverage"] = cov

    res["e3"] = {}
    for a in ["e3_offpeak_s60", "e3_uniform", "e3_gauss_s240", "e3_gauss_s60", "e3_gauss_s15"]:
        ins = np.array([in_span(placed(a, k), items[k]["span"]) for k in keys])
        res["e3"][a] = {"acc": acc(a, keys), "mean_frames_on_evidence": round(ins.mean(), 2),
                        "share_zero": round(float(np.mean(ins == 0)), 4)}

    import pandas as pd                       # question types live in the parquet only
    from t1_data import LVB
    qt = pd.read_parquet(f"{LVB}/data/train-00000-of-00001.parquet")
    types = {f"{r.key}>>>{r.uid}": list(r.question_type) for r in qt.itertuples(index=False)}
    res["qtype"] = {}
    for t in sorted({t for k in keys for t in types[k]}):
        ks = [k for k in keys if t in types[k]]
        res["qtype"][t] = {"n": len(ks), "uniform": acc("e3_uniform", ks),
                           "topk_8b_options": acc("ge1_opt_topk", ks), "M": acc("q8_kelly", ks)}

    gaps = []
    for k in keys:
        ts = np.sort(curve_times(zo[k], dur[k], N, "kelly", TAU, EPS))
        gaps.append(float(np.diff(ts).min()))
    gaps = np.array(gaps)
    res["spacing"] = {"min_gap_median_s": round(float(np.median(gaps)), 3),
                      "share_min_gap_below_1_30s": round(float(np.mean(gaps < 1 / 30)), 4),
                      "share_min_gap_below_1s": round(float(np.mean(gaps < 1.0)), 4)}

    po_path = os.path.join(PRIOR, "curves_peropt.npz")
    if os.path.exists(po_path):
        po = np.load(po_path)
        gold_top, gold_loc, answer = [], [], []
        for k in keys:
            c = po[f"lvbench:{k}"].astype(np.float64)               # [n_opts, T]
            g = "ABCDEFGH".index(items[k]["gold"])
            winner = int(np.argmax(c.max(1)))
            gold_top.append(winner == g)
            m, t = masks[k], int(np.argmax(c[g]))
            ev = np.flatnonzero(m)
            gold_loc.append(0.0 if m[t] else float(np.abs(ev - t).min()))
            answer.append(winner == g)
        res["options"] = {
            "share_top_second_from_gold_option": round(float(np.mean(gold_top)), 4),
            "gold_option_curve_within_60s": round(float(np.mean(np.array(gold_loc) <= 60)), 4),
            "gold_option_curve_median_miss_s": float(np.median(gold_loc)),
            "embedder_as_answerer_acc": round(float(np.mean(answer)) * 100, 2)}

    # the worked example, end to end
    k = EXAMPLE
    it = items[k]
    z = zo[k][:max(1, int(np.ceil(dur[k])))]
    p = {str(t): (lambda q: (q / q.sum()))(np.exp((z - z.max()) / t)) for t in (1.0, TAU, 10.0)}
    mix = (1 - EPS) * p[str(TAU)] + EPS / len(z)
    kel = curve_times(zo[k], dur[k], N, "kelly", TAU, EPS)
    top = curve_times(zo[k], dur[k], N, "topk", TAU, EPS)
    uni = kernel_times("uniform", it["center"], dur[k], N)
    ex = {"key": k, "question": it["question"], "gold": it["gold"], "span": it["span"],
          "duration_s": dur[k], "T": len(z), "stem_curve": np.round(zs[k][:len(z)], 3).tolist(),
          "maxopt_curve": np.round(z, 3).tolist(),
          "p": {t: np.round(v, 6).tolist() for t, v in p.items()},
          "mixture": np.round(mix, 6).tolist(), "kelly_times": [round(t, 3) for t in kel],
          "topk_times": top, "uniform_times": [round(t, 3) for t in uni],
          "frames_on_evidence": {"uniform": in_span(uni, it["span"]), "topk": in_span(top, it["span"]),
                                 "kelly": in_span(kel, it["span"])},
          "mass_on_evidence": {t: round(float(v[masks[k][:len(z)]].sum()), 4) for t, v in p.items()},
          "peaks": peaks(p[str(TAU)]),
          "answers": {a: R[a][k]["pred_letter"] for a in ["e3_uniform", "b1_topk_siglip",
                                                          "ge1_stem_topk", "ge1_opt_topk", "q8_kelly"]}}
    if os.path.exists(po_path):
        ex["option_curves"] = np.round(np.load(po_path)[f"lvbench:{k}"].astype(np.float64)[:, :len(z)],
                                       3).tolist()
    res["example"] = ex
    with open(OUT, "w", encoding="utf-8") as fh:
        json.dump(res, fh)
    print(f"-> {OUT}", flush=True)
    print(json.dumps({q: v for q, v in res.items() if q not in ("example", "tau")}, indent=1))


if __name__ == "__main__":
    main()
