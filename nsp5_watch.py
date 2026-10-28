"""Event stream for a Claude Code Monitor over the NS-P5 X1/G2 runs (one line per event).

    python3 -u nsp5_watch.py          # a Monitor event stream
    python3 -u nsp5_watch.py --once   # exit after the first event (a background task that wakes you once)

Prints when an arm completes or stalls (no row for 15 min while its process is up), when an
8B prior lands, errors or stalls, when the scheduler (nsp5_x1_resume.sh) logs an important event
(encode, fuse, coverage, torn lines, all done), and if the scheduler exits.
"""
import collections
import os
import re
import subprocess
import sys
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
L = "logs/nsp5"
TOT = {"videomme": 2700, "lvb": 1337, "mlvu": 2174, "cgbench": 3000}
ARMS = {"g2_q35_b1": 1534, "x1_cgbench_a3_kelly_placebo": 2683,  # placebo: multi-question videos only
        "q8_kelly": 1534, "q8_kelly_q3": 1534, "q8_kelly_q35": 1534, "q8_kelly_placebo": 1534,
        "x1_cgbench_q8_kelly_placebo": 2683}
for _b, _t in TOT.items():
    for _a in ("uniform", "b1", "a2_kelly", "a3_kelly", "a3_kelly_q3", "q8opt_topk", "uniform_q3", "q8_kelly", "q8_kelly_q3"):
        ARMS[f"x1_{_b}_{_a}"] = _t
for _a in ("f2c_kelly", "bolt_its_q8", "aks_ada_q8", "bolt_own", "aks_own", "kelly_clip", "kelly_blip"):
    ARMS[_a] = 1534                      # prior-art arms, pre-registered 2026-10-28
    for _b, _t in TOT.items():
        ARMS[f"x1_{_b}_{_a}"] = _t
PAT = re.compile(r"Traceback|Error|Killed|No space")
SPAT = re.compile(r"fuse|encode|COVERAGE|every arm|torn")
SCHED = f"{L}/x1_scheduler.log"
ENCODED = ("mlvu", "cgbench")


ONCE = "--once" in sys.argv


def emit(msg):
    print(msg, flush=True)
    if ONCE:
        sys.exit(0)


def rows(n):
    p = f"{L}/{n}.jsonl"
    return sum(1 for _ in open(p)) if os.path.exists(p) else None


def py_age(needle):
    out = subprocess.run(["ps", "-eo", "etimes=,comm=,args="], capture_output=True, text=True).stdout
    for line in out.splitlines():
        f = line.split(None, 2)
        if len(f) == 3 and f[1].startswith("python") and needle in f[2]:
            return int(f[0])
    return None


def count(path, rx):
    return sum(1 for l in open(path, errors="ignore") if rx.search(l)) if os.path.exists(path) else 0


def main():
    seen = {n for n, t in ARMS.items() if (rows(n) or 0) >= t}
    # Report the scheduler only if it stops during the watch; one stopped on purpose (arms split
    # into shards by hand) would otherwise end every --once watch at once.
    if subprocess.run(["pgrep", "-f", "nsp5_x1_resume.sh run"], capture_output=True).returncode:
        seen.add("sched")
    seen |= {f"p{b}" for b in ENCODED if os.path.exists(f"data/t1/priors/qwen3vl_emb_8b/{b}_img.npz")}
    err = {b: count(f"{L}/x1_prior_q8b_{b}_r3.log", PAT) for b in ENCODED}
    sb = count(SCHED, SPAT)
    launches = lambda: collections.Counter(re.findall(r" launch (\S+) ", open(SCHED, errors="ignore").read()))
    l0 = launches()
    while True:
        for n, c in launches().items():   # an arm that dies at startup is relaunched every pass
            if c - l0.get(n, 0) >= 3 and f"{n}.loop" not in seen:
                seen.add(f"{n}.loop")
                emit(f"RELAUNCH LOOP {n}: launched {c - l0.get(n, 0)} times since the watch began")
        now = time.time()
        if subprocess.run(["pgrep", "-f", "nsp5_x1_resume.sh run"], capture_output=True).returncode and "sched" not in seen:
            seen.add("sched")
            emit("SCHEDULER EXITED")
        lines = [l.rstrip() for l in open(SCHED, errors="ignore") if SPAT.search(l)]
        for l in lines[sb:]:
            emit(f"SCHEDULER: {l}")
        sb = len(lines)
        for b in ENCODED:
            if os.path.exists(f"data/t1/priors/qwen3vl_emb_8b/{b}_img.npz") and f"p{b}" not in seen:
                seen.add(f"p{b}")
                emit(f"PRIOR DONE qwen3vl_emb_8b {b}")
            log = f"{L}/x1_prior_q8b_{b}_r3.log"
            k = count(log, PAT)
            if k > err[b]:
                err[b] = k
                emit(f"PRIOR ERROR {b}")
            up = py_age(f"--name qwen3vl_emb_8b --bench {b}")
            if up and up > 1500 and os.path.exists(log) and now - os.path.getmtime(log) > 1500 and f"s{b}" not in seen:
                seen.add(f"s{b}")
                emit(f"PRIOR STALL {b}: no log write for 25+ min")
        for n, t in ARMS.items():
            c = rows(n)
            if c is None:
                continue
            if c >= t and n not in seen:
                seen.add(n)
                emit(f"EVAL DONE {n} ({c} rows)")
            up = py_age(f"--out {L}/{n}.jsonl")
            if c < t and up and up > 900 and now - os.path.getmtime(f"{L}/{n}.jsonl") > 900 and f"{n}.stall" not in seen:
                seen.add(f"{n}.stall")
                emit(f"EVAL STALL {n} at {c}/{t}: no row for 15+ min")
        time.sleep(60)


if __name__ == "__main__":
    main()
