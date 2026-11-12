"""Study W6 (NS-P5, METHODS-LOG 2026-11-12, "Reviewer-requested experiments, wave 1"): where
CaliFrame's (M2, q8_kelly_eps0) gain comes from. Qwen2.5-VL-7B, 64 frames; uniform sampling
(e3_uniform on LVBench, x1_<bench>_uniform elsewhere), GMM-EVA's selection (f2c_topk_nr) and M2.

  category   M2 minus uniform and minus GMM-EVA's selection by each benchmark's official question
             category: LVBench question_type (a question can carry several and counts in each),
             Video-MME task_type, LongVideoBench question_category (and its level), MLVU task,
             CG-Bench sub_category (and its perception / cognition / hallucination group)
  duration   by video duration (the harness's duration_s): five fixed bins pooled over the five
             benchmarks, and the official groups of Video-MME and LongVideoBench
  evidence   by evidence length, the union of the annotated intervals, in the bins of
             nsp5_method_figures.py (under 1 s, 1-30 s, 30 s-5 min, over 5 min), on LVBench and
             CG-Bench (the two benchmarks that annotate evidence) and on both pooled; and by
             CG-Bench's number of clue intervals (1, 2, 3 or more)
  steering   LVBench: the option whose 8B per-option curve (curves_peropt.npz) peaks highest, its
             accuracy as an answer, and how often the answer model picks it under uniform, GMM-EVA's
             selection and M2. Only LVBench has per-option scores (the other benchmarks store the
             max over options; their per-option curves need the GPU text embedder), so on all five
             the analysis is limited to the answer model's own picks: how often two placements'
             wrong answers are the same option.
  won/lost   questions M2 wins and loses against uniform on each benchmark; on LVBench and CG-Bench
             the lost (and won) ones sorted into long evidence (over 5 min), evidence missed (no M2
             frame on it) and evidence covered but answered wrong, in that order of precedence.

Differences in points with 95 % intervals from the video bootstrap of nsp5_pooled.py (videos
resampled within each benchmark, 5,000 draws, seed 0) inside each subset. Descriptive.

    .venv/bin/python nsp5_w6_breakdown.py

writes logs/nsp5/analysis/w6_breakdown.json and w6_breakdown.txt.
"""
from __future__ import annotations

import math
from collections import Counter
from dataclasses import dataclass

import numpy as np
import pandas as pd

import bench
from nsp5_compare import LVB
from nsp5_pooled import BENCH
from nsp5_w_paired import GMM, NAMES, OURS, UNIFORM, boot, rows, by_bench, write_outputs

ARMS = {"U": UNIFORM, "G": GMM, "M": OURS}
LETTERS = "ABCDEFGH"
N_FRAMES = 64
PRIOR = "data/t1/priors/qwen3vl_emb_8b"
LONG_EVIDENCE_S = 300.0            # the top bin of nsp5_method_figures.py's evidence lengths
DURATION_BINS = (("under 2 min", 0.0, 120.0), ("2-10 min", 120.0, 600.0), ("10-30 min", 600.0, 1800.0),
                 ("30-60 min", 1800.0, 3600.0), ("over 60 min", 3600.0, math.inf))
EVIDENCE_LABELS = ("under 1 s", "1-30 s", "30 s-5 min", "over 5 min")
LVB_GROUPS = {"15": "8-15 s", "60": "15-60 s", "600": "3-10 min", "3600": "15-60 min"}
HEADER = (f"  {'group':<31}{'n':>5}{'videos':>7}{'unif.':>7}{'GMM-EVA':>8}{'M2':>6}   "
          f"{'M2 - uniform [95% CI]':<27}{'M2 - GMM-EVA [95% CI]'}")


@dataclass(frozen=True)
class Data:
    keys: list            # every question of the five benchmarks
    rows: dict            # arm label (U, G, M) -> key -> logged row
    correct: dict         # arm label -> key -> 0/1
    bench_of: dict        # key -> benchmark
    n_opts: dict          # key -> number of options
    duration: dict        # key -> seconds (M2's logged duration_s)
    spans: dict           # key -> evidence intervals, LVBench and CG-Bench only
    cats: dict            # key -> official categories (LVBench may carry several)
    extra: dict           # key -> {"lvb_level" | "cg_group" | "vmme_split" | "lvb_group": label}


def load() -> Data:
    r = {lab: {k: x for b in BENCH for k, x in rows(arm, b).items()} for lab, arm in ARMS.items()}
    keys = sorted(r["M"])
    assert set(keys) == set(r["U"]) == set(r["G"]) and len(keys) == 10745, "arms are not complete"
    items = {b: bench.load(b) for b in BENCH if b != "lvbench"}
    bench_of = {k: ("lvbench" if ">>>" in k else k.split(":", 1)[0]) for k in keys}
    n_opts = {k: 4 for k in keys if bench_of[k] == "lvbench"}            # LVBench: always (A)-(D)
    n_opts.update({it.key: len(it.options) for its in items.values() for it in its})
    spans = {k: (tuple(r["M"][k]["span"]),) for k in keys if bench_of[k] == "lvbench"}
    spans.update({it.key: it.spans for it in items["cgbench"]})
    df = pd.read_parquet(LVB)                       # LVBench's question types live in the parquet
    cats = {f"{x.key}>>>{x.uid}": tuple(x.question_type) for x in df.itertuples(index=False)}
    field = {"videomme": "task", "lvb": "category", "mlvu": "task", "cgbench": "category"}
    cats.update({it.key: (it.meta[field[b]],) for b, its in items.items() for it in its})
    extra = {it.key: {"vmme_split": it.meta["split"]} for it in items["videomme"]}
    extra.update({it.key: {"lvb_level": it.meta["level"], "lvb_group": LVB_GROUPS[str(it.meta["duration_group"])]}
                  for it in items["lvb"]})
    extra.update({it.key: {"cg_group": "Hallucination" if it.meta["category"] == "Hallucination"
                           else it.meta["category"].split()[-1]} for it in items["cgbench"]})
    return Data(keys=keys, rows=r, correct={lab: {k: float(x["correct"]) for k, x in v.items()} for lab, v in r.items()},
                bench_of=bench_of, n_opts=n_opts, duration={k: float(r["M"][k]["duration_s"]) for k in keys},
                spans=spans, cats=cats, extra=extra)


def union_length(spans) -> float:
    total, end = 0.0, -math.inf
    for a, b in sorted(spans):
        if b > end:
            total += b - max(a, end)
            end = b
    return total


def evidence_bin(length: float) -> str:
    if length < 1:
        return "under 1 s"
    if length <= 30:
        return "1-30 s"
    return "30 s-5 min" if length <= LONG_EVIDENCE_S else "over 5 min"


def compare(d: Data, groups: list) -> tuple[list[dict], list[str]]:
    """M2 minus uniform and minus GMM-EVA's selection on each (label, keys) group."""
    recs, lines = [], [HEADER]
    for label, ks in groups:
        if not ks:
            continue
        kb = by_bench(ks)
        mu, mg = boot(d.correct["M"], d.correct["U"], kb), boot(d.correct["M"], d.correct["G"], kb)
        recs.append({"group": label, "n": mu.n, "videos": mu.videos, "uniform": round(mu.r, 2),
                     "gmm_eva": round(mg.r, 2), "m2": round(mu.a, 2),
                     "m2_minus_uniform": [round(x, 2) for x in (mu.diff, mu.lo, mu.hi)],
                     "m2_minus_gmm_eva": [round(x, 2) for x in (mg.diff, mg.lo, mg.hi)],
                     "n_by_bench": {b: len(v) for b, v in kb.items()}})
        lines.append(f"  {label[:30]:<31}{mu.n:>5}{mu.videos:>7}{mu.r:>7.1f}{mg.r:>8.1f}{mu.a:>6.1f}   "
                     f"{mu.text():<27}{mg.text()}")
    return recs, lines


def grouped(keys: list, label_of) -> list:
    """(label, keys) for every label, largest group first; label_of(key) gives a tuple of labels."""
    counts = Counter(lab for k in keys for lab in label_of(k))
    return [(lab, [k for k in keys if lab in label_of(k)]) for lab, _ in counts.most_common()]


def by_category(d: Data) -> tuple[dict, list[str]]:
    out, lines = {}, ["", "== by official question category (LVBench: a question counts in each of its types)"]
    for b in BENCH:
        ks = [k for k in d.keys if d.bench_of[k] == b]
        recs, ls = compare(d, grouped(ks, lambda k: d.cats[k]))
        out[b] = recs
        lines += ["", f"{NAMES[b]}"] + ls
    for name, b, field in (("lvb_level", "lvb", "lvb_level"), ("cg_group", "cgbench", "cg_group")):
        ks = [k for k in d.keys if d.bench_of[k] == b]
        recs, ls = compare(d, grouped(ks, lambda k: (d.extra[k][field],)))
        out[name] = recs
        lines += ["", f"{NAMES[b]}, {'level' if b == 'lvb' else 'perception / cognition / hallucination'}"] + ls
    return out, lines


def by_duration(d: Data) -> tuple[dict, list[str]]:
    groups = [(lab, [k for k in d.keys if lo <= d.duration[k] < hi]) for lab, lo, hi in DURATION_BINS]
    pooled_recs, ls = compare(d, groups)
    lines = ["", "== by video duration, five benchmarks pooled (bootstrap within each benchmark)"] + ls
    lines.append("  questions per benchmark in each bin: " + "; ".join(
        f"{r['group']}: " + ", ".join(f"{NAMES[b]} {n}" for b, n in r["n_by_bench"].items()) for r in pooled_recs))
    out = {"pooled_bins": pooled_recs}
    for name, b, field, order in (("videomme_split", "videomme", "vmme_split", ("short", "medium", "long")),
                                  ("lvb_duration_group", "lvb", "lvb_group", tuple(LVB_GROUPS.values()))):
        ks = [k for k in d.keys if d.bench_of[k] == b]
        recs, ls = compare(d, [(lab, [k for k in ks if d.extra[k][field] == lab]) for lab in order])
        out[name] = recs
        lines += ["", f"{NAMES[b]}, official duration groups"] + ls
    return out, lines


def by_evidence(d: Data) -> tuple[dict, list[str]]:
    length = {k: union_length(s) for k, s in d.spans.items()}
    out, lines = {}, ["", "== by evidence length (union of the annotated intervals)"]
    for scope, benches in (("LVBench", ("lvbench",)), ("CG-Bench", ("cgbench",)),
                           ("LVBench and CG-Bench", ("lvbench", "cgbench"))):
        ks = [k for k in d.keys if d.bench_of[k] in benches]
        recs, ls = compare(d, [(lab, [k for k in ks if evidence_bin(length[k]) == lab]) for lab in EVIDENCE_LABELS])
        out[scope] = recs
        lines += ["", scope] + ls
    ks = [k for k in d.keys if d.bench_of[k] == "cgbench"]
    n_int = {k: len(d.spans[k]) for k in ks}
    recs, ls = compare(d, [("1 clue interval", [k for k in ks if n_int[k] == 1]),
                           ("2 clue intervals", [k for k in ks if n_int[k] == 2]),
                           ("3 or more clue intervals", [k for k in ks if n_int[k] >= 3])])
    out["cgbench_clue_intervals"] = recs
    lines += ["", "== CG-Bench by number of clue intervals"] + ls
    return out, lines


def peak_options(keys: list) -> tuple[dict, dict]:
    """LVBench: the option whose per-option curve peaks highest, and a check that the per-option
    file reproduces the max-option curve M2 placed its frames on."""
    po, mo = np.load(f"{PRIOR}/curves_peropt.npz"), np.load(f"{PRIOR}/curves_maxopt.npz")
    peak, same_second, diffs = {}, 0, []
    for k in keys:
        c = po[f"lvbench:{k}"].astype(np.float64)                  # [4 options, seconds]
        z = mo[f"lvbench:{k}"].astype(np.float64)
        peak[k] = LETTERS[int(np.argmax(c.max(1)))]
        same_second += int(np.argmax(c.max(0)) == np.argmax(z))
        diffs.append(np.abs(c.max(0) - z))
    diffs = np.concatenate(diffs)
    check = {"same_peak_second": same_second, "of": len(keys), "abs_diff_median": float(np.median(diffs)),
             "abs_diff_p99": float(np.percentile(diffs, 99)), "abs_diff_max": float(diffs.max())}
    return peak, check


def steering(d: Data) -> tuple[dict, list[str]]:
    ks = [k for k in d.keys if d.bench_of[k] == "lvbench"]
    peak, check = peak_options(ks)
    pick = {lab: {k: float(d.rows[lab][k]["pred_letter"] == peak[k]) for k in ks} for lab in ARMS}
    right = [k for k in ks if peak[k] == d.rows["M"][k]["gold_letter"]]
    wrong = [k for k in ks if peak[k] != d.rows["M"][k]["gold_letter"]]
    out = {"curve_check": check, "peak_option_accuracy": round(100 * len(right) / len(ks), 2), "chance": 25.0}
    lines = ["", "== option steering, LVBench (the option whose 8B per-option curve peaks highest)",
             f"  per-option curves reproduce M2's max-option curve: same peak second on {check['same_peak_second']}"
             f" of {check['of']} questions (|difference| median {check['abs_diff_median']:.2f}, max {check['abs_diff_max']:.2f})",
             f"  the peak option is the answer on {len(right)} of {len(ks)} questions: "
             f"{out['peak_option_accuracy']:.2f}% (chance 25%)",
             f"  {'questions':<27}{'n':>5}   picks the peak option: {'unif.':>6}{'GMM-EVA':>8}{'M2':>6}   "
             f"{'M2 - uniform':<27}{'M2 - GMM-EVA'}"]
    for label, sub in (("all", ks), ("peak option is the answer", right), ("peak option is wrong", wrong)):
        kb = {"lvbench": sub}
        pu, pg = boot(pick["M"], pick["U"], kb), boot(pick["M"], pick["G"], kb)
        au, ag = boot(d.correct["M"], d.correct["U"], kb), boot(d.correct["M"], d.correct["G"], kb)
        out[label] = {"n": len(sub), "picks_peak": {"uniform": round(pu.r, 2), "gmm_eva": round(pg.r, 2), "m2": round(pu.a, 2)},
                      "picks_peak_m2_minus_uniform": [round(x, 2) for x in (pu.diff, pu.lo, pu.hi)],
                      "picks_peak_m2_minus_gmm_eva": [round(x, 2) for x in (pg.diff, pg.lo, pg.hi)],
                      "accuracy": {"uniform": round(au.r, 2), "gmm_eva": round(ag.r, 2), "m2": round(au.a, 2)},
                      "acc_m2_minus_uniform": [round(x, 2) for x in (au.diff, au.lo, au.hi)],
                      "acc_m2_minus_gmm_eva": [round(x, 2) for x in (ag.diff, ag.lo, ag.hi)]}
        lines.append(f"  {label:<27}{len(sub):>5}   {'':<23}{pu.r:>6.1f}{pg.r:>8.1f}{pu.a:>6.1f}   {pu.text():<27}{pg.text()}")
        lines.append(f"  {'':<27}{'':>5}   {'accuracy:':<23}{au.r:>6.1f}{ag.r:>8.1f}{au.a:>6.1f}   {au.text():<27}{ag.text()}")
    errs = {}
    for lab, name in (("U", "uniform"), ("G", "gmm_eva"), ("M", "m2")):
        e = [k for k in wrong if not d.correct[lab][k]]
        errs[name] = {"errors": len(e), "on_peak_option": int(sum(pick[lab][k] for k in e)),
                      "share": round(100 * float(np.mean([pick[lab][k] for k in e])), 2)}
    out["errors_on_peak_option_when_it_is_wrong"] = {**errs, "chance_share": round(100 / 3, 2)}
    lines.append("  wrong answers that are the (wrong) peak option, on the questions whose peak option is wrong "
                 "(chance 33.3%): " + ", ".join(f"{n} {v['on_peak_option']}/{v['errors']} = {v['share']:.1f}%"
                                                  for n, v in errs.items()))
    return out, lines


def coincidence(d: Data) -> tuple[dict, list[str]]:
    """How often two placements' wrong answers are the same option (all five benchmarks)."""
    out, lines = {}, ["", "== the answer model's wrong answers under two placements: share that are the same option",
                      f"  {'benchmark':<16}" + "".join(f"{p:>24}" for p in ("M2 & uniform", "M2 & GMM-EVA", "GMM-EVA & uniform"))
                      + f"{'chance':>9}"]
    pairs = (("M", "U"), ("M", "G"), ("G", "U"))
    for scope in list(BENCH) + ["pooled"]:
        ks = [k for k in d.keys if scope == "pooled" or d.bench_of[k] == scope]
        rec, cells, chance = {}, [], []
        for x, y in pairs:
            both = [k for k in ks if not d.correct[x][k] and not d.correct[y][k]
                    and d.rows[x][k]["pred_letter"] and d.rows[y][k]["pred_letter"]]
            same = int(sum(d.rows[x][k]["pred_letter"] == d.rows[y][k]["pred_letter"] for k in both))
            rec[f"{x}{y}"] = {"both_wrong": len(both), "same_option": same, "share": round(100 * same / len(both), 2)}
            cells.append(f"{same:>6}/{len(both):<6} {100 * same / len(both):5.1f}%")
            chance.append(float(np.mean([1 / (d.n_opts[k] - 1) for k in both])))
        rec["chance_share"] = round(100 * float(np.mean(chance)), 2)
        out[scope] = rec
        lines.append(f"  {NAMES.get(scope, 'pooled'):<16}" + "".join(f"{c:>24}" for c in cells) + f"{rec['chance_share']:>8.1f}%")
    return out, lines


def uniform_in_span(dur: float, spans) -> int:
    """Frames of uniform sampling on the evidence: eval_foveate.kernel_times('uniform') with the
    harness's +-0.5 s tolerance (as nsp5_method_figures.py recomputes it for LVBench)."""
    times = [dur * (i + 0.5) / N_FRAMES for i in range(N_FRAMES)]
    return sum(any(a - 0.5 <= t <= b + 0.5 for a, b in spans) for t in times)


def evidence_sort(d: Data, keys: list, u_span: dict, outcome: str) -> dict:
    """Sort questions into long evidence, evidence missed and evidence covered (by M2's frames; for
    lost questions, covered but answered wrong); for each, how many uniform covered; and the
    unprioritized cross-table."""
    covered = f"evidence covered, answered {'wrong' if outcome == 'lost' else 'right'}"
    out = {"long evidence (over 5 min)": [], "evidence missed": [], covered: []}
    cross = Counter()
    for k in keys:
        long_ = union_length(d.spans[k]) > LONG_EVIDENCE_S
        hit = d.rows["M"][k]["in_span"] > 0
        cross[("long" if long_ else "short", "covered" if hit else "missed")] += 1
        cat = "long evidence (over 5 min)" if long_ else covered if hit else "evidence missed"
        out[cat].append(k)
    return {"by_category": {c: {"n": len(v), "uniform_covered": int(sum(u_span[k] > 0 for k in v)),
                                "m2_frames_on_evidence_mean": round(float(np.mean([d.rows["M"][k]["in_span"] for k in v])), 2) if v else None}
                            for c, v in out.items()},
            "cross": {f"{a} evidence, M2 {b}": n for (a, b), n in sorted(cross.items())}}


def won_lost(d: Data) -> tuple[dict, list[str]]:
    out, lines = {}, ["", "== questions won and lost against uniform (M2 right and uniform wrong; the reverse)",
                      f"  {'benchmark':<16}{'n':>6}{'won':>6}{'lost':>6}{'net':>6}{'both right':>12}{'both wrong':>12}"]
    for scope in list(BENCH) + ["pooled"]:
        ks = [k for k in d.keys if scope == "pooled" or d.bench_of[k] == scope]
        m, u = d.correct["M"], d.correct["U"]
        c = {"n": len(ks), "won": sum(m[k] > u[k] for k in ks), "lost": sum(m[k] < u[k] for k in ks),
             "both_right": sum(m[k] == u[k] == 1 for k in ks), "both_wrong": sum(m[k] == u[k] == 0 for k in ks)}
        c = {k: int(v) for k, v in c.items()}
        out[scope] = c
        lines.append(f"  {NAMES.get(scope, 'pooled'):<16}{c['n']:>6}{c['won']:>6}{c['lost']:>6}{c['won'] - c['lost']:>+6}"
                     f"{c['both_right']:>12}{c['both_wrong']:>12}")
    u_span, check = {}, []
    for k in d.keys:
        b = d.bench_of[k]
        if b == "lvbench":
            u_span[k] = uniform_in_span(float(d.rows["U"][k]["duration_s"]), d.spans[k])
        elif b == "cgbench":
            u_span[k] = d.rows["U"][k]["in_span"]
            check.append(uniform_in_span(float(d.rows["U"][k]["duration_s"]), d.spans[k]) == u_span[k])
    out["uniform_in_span_recompute_check_cgbench"] = {"agree": int(sum(check)), "of": len(check)}
    lines.append(f"  (uniform's frames on the evidence: logged on CG-Bench, recomputed on LVBench; the recomputation "
                 f"agrees with CG-Bench's log on {sum(check)} of {len(check)} questions)")
    for scope, benches in (("LVBench", ("lvbench",)), ("CG-Bench", ("cgbench",)), ("LVBench and CG-Bench", ("lvbench", "cgbench"))):
        ks = [k for k in d.keys if d.bench_of[k] in benches]
        lost = [k for k in ks if d.correct["M"][k] < d.correct["U"][k]]
        won = [k for k in ks if d.correct["M"][k] > d.correct["U"][k]]
        out[f"sorted {scope}"] = {"lost": evidence_sort(d, lost, u_span, "lost"),
                                  "won": evidence_sort(d, won, u_span, "won")}
        for name, sub in (("lost", lost), ("won", won)):
            s = out[f"sorted {scope}"][name]
            lines.append(f"  {scope}, {name} ({len(sub)}): " + "; ".join(
                f"{c} {v['n']} (uniform covered {v['uniform_covered']})" for c, v in s["by_category"].items()))
    return out, lines


def main() -> None:
    d = load()
    data, lines = {"arms": {"uniform": UNIFORM, "gmm_eva": GMM, "m2": OURS}}, [
        "W6: where M2's gain comes from. Qwen2.5-VL-7B, 64 frames; accuracy in %, differences in points "
        "with 95% video-bootstrap intervals (5,000 draws, seed 0)"]
    for name, fn in (("category", by_category), ("duration", by_duration), ("evidence", by_evidence),
                     ("steering_lvbench", steering), ("coincidence", coincidence), ("won_lost", won_lost)):
        data[name], ls = fn(d)
        lines += ls
    write_outputs("w6_breakdown", data, lines)


if __name__ == "__main__":
    main()
