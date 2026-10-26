"""Event stream for a Claude Code Monitor over the NS-P5 X1/G2 runs (one line per event).

    python3 -u nsp5_watch.py

Prints when an arm completes or stalls (no row for 15 min while its process is up), when an
8B prior lands, errors or stalls, when the scheduler (nsp5_x1_resume.sh) logs an important event
(encode, fuse, coverage, torn lines, all done), and if the scheduler exits.
"""
import os
import re
import subprocess
import time

os.chdir(os.path.dirname(os.path.abspath(__file__)))
L = "logs/nsp5"
TOT = {"videomme": 2700, "lvb": 1337, "mlvu": 2174, "cgbench": 3000}
ARMS = {"g2_q35_b1": 1534, "x1_cgbench_a3_kelly_placebo": 3000}
for _b, _t in TOT.items():
    for _a in ("uniform", "b1", "a2_kelly", "a3_kelly", "a3_kelly_q3", "q8opt_topk", "uniform_q3"):
        ARMS[f"x1_{_b}_{_a}"] = _t
PAT = re.compile(r"Traceback|Error|Killed|No space")
SPAT = re.compile(r"fuse|encode|COVERAGE|every arm|torn")
SCHED = f"{L}/x1_scheduler.log"
ENCODED = ("mlvu", "cgbench")


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
    seen |= {f"p{b}" for b in ENCODED if os.path.exists(f"data/t1/priors/qwen3vl_emb_8b/{b}_img.npz")}
    err = {b: count(f"{L}/x1_prior_q8b_{b}_r3.log", PAT) for b in ENCODED}
    sb = count(SCHED, SPAT)
    while True:
        now = time.time()
        if subprocess.run(["pgrep", "-f", "nsp5_x1_resume.sh run"], capture_output=True).returncode and "sched" not in seen:
            seen.add("sched")
            print("SCHEDULER EXITED")
        lines = [l.rstrip() for l in open(SCHED, errors="ignore") if SPAT.search(l)]
        for l in lines[sb:]:
            print("SCHEDULER:", l)
        sb = len(lines)
        for b in ENCODED:
            if os.path.exists(f"data/t1/priors/qwen3vl_emb_8b/{b}_img.npz") and f"p{b}" not in seen:
                seen.add(f"p{b}")
                print(f"PRIOR DONE qwen3vl_emb_8b {b}")
            log = f"{L}/x1_prior_q8b_{b}_r3.log"
            k = count(log, PAT)
            if k > err[b]:
                err[b] = k
                print(f"PRIOR ERROR {b}")
            up = py_age(f"--name qwen3vl_emb_8b --bench {b}")
            if up and up > 1500 and os.path.exists(log) and now - os.path.getmtime(log) > 1500 and f"s{b}" not in seen:
                seen.add(f"s{b}")
                print(f"PRIOR STALL {b}: no log write for 25+ min")
        for n, t in ARMS.items():
            c = rows(n)
            if c is None:
                continue
            if c >= t and n not in seen:
                seen.add(n)
                print(f"EVAL DONE {n} ({c} rows)")
            up = py_age(f"--out {L}/{n}.jsonl")
            if c < t and up and up > 900 and now - os.path.getmtime(f"{L}/{n}.jsonl") > 900 and f"{n}.stall" not in seen:
                seen.add(f"{n}.stall")
                print(f"EVAL STALL {n} at {c}/{t}: no row for 15+ min")
        time.sleep(60)


if __name__ == "__main__":
    main()
