#!/usr/bin/env python3
"""NS-P5 job runner: keeps the GPUs fed with registered eval shards and merges finished arms.

Every minute it reads logs/nsp5/runner/plan.json, a list in launch priority:

    {"arms": [{"name": "pub32_uniform", "study": "C", "bench": "lvbench", "venv": ".venv",
               "args": "--arm uniform --n_frames 32", "shards": 2, "total": 1534}, ...]}

Shard 0 writes logs/nsp5/<name>.jsonl and shard k > 0 <name>.shard<k>.jsonl (the convention of
the earlier run records); the harness resumes from its own file. Only arms of a pre-registered
study belong in the plan, and only once its registration is on GitHub.

GPU rule. A GPU takes up to MAX_MINE of my evals while nobody else uses it, each launch needing
FREE_MIN MiB free after RESERVE MiB is set aside for every launch of the last ten minutes. A GPU
whose used memory my evals cannot explain is someone else's too: there I keep at most
SHARED_MINE evals and leave SHARED_FREE MiB free (the owner's 70 GB rule of 2026-11-02).
GPUs listed in logs/nsp5/runner/skip_gpus (e.g. "1,3") get no new launches. A free slot goes to
the GPU with the fewest of my evals. Launches pause while the container uses more than CPU_LIMIT
of its 128 cores (measured over the last minute from the cgroup); an eval decodes video on about
3 cores (2026-11-05), so GPU memory is usually the limit.

A shard killed by a signal goes back to pending. Once no shard of an arm runs or waits, the arm
is merged: every line parses, keys are unique across files (a duplicate must agree on the
answer) and the count equals "total"; shard files are kept as <file>.merged. An arm short of
its total is relaunched (shards resume) up to RETRIES times, then reported as an anomaly.
Status goes to logs/nsp5/runner/status.json each minute, events to stdout.

    cd research/TimeLens
    setsid nohup python3 -u nsp5_runner.py >> logs/nsp5/runner/runner.log 2>&1 < /dev/null &
"""
import json
import os
import shlex
import subprocess
import time

TL = os.path.dirname(os.path.abspath(__file__))
L = "logs/nsp5"
R = f"{L}/runner"
PLAN, STATUS, SKIP = f"{R}/plan.json", f"{R}/status.json", f"{R}/skip_gpus"
MAX_MINE, SHARED_MINE = 6, 2
FREE_MIN, SHARED_FREE, RESERVE, PER_EVAL = 40000, 60000, 30000, 30000   # MiB
CPU_STAT, CPU_LIMIT = "/sys/fs/cgroup/cpu.stat", 110                    # cores of the 128 quota
RETRIES, LOOP_S = 2, 60


def cpu_usec():
    """The container's cumulative CPU time in microseconds, or None if unreadable."""
    try:
        for line in open(CPU_STAT):
            if line.startswith("usage_usec"):
                return int(line.split()[1])
    except OSError:
        pass
    return None


def log(msg: str) -> None:
    print(f"{time.strftime('%F %T')} {msg}", flush=True)


def evals() -> dict:
    """{--out path: (pid, GPU)} for every eval_lvbench_foveate.py process on the machine."""
    out = {}
    for pid in os.listdir("/proc"):
        if not pid.isdigit():
            continue
        try:
            argv = [a.decode() for a in open(f"/proc/{pid}/cmdline", "rb").read().split(b"\0") if a]
            env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
        except OSError:
            continue
        if not any(a.endswith("eval_lvbench_foveate.py") for a in argv) or "--out" not in argv:
            continue
        gpu = next((e.decode().split("=", 1)[1] for e in env
                    if e.startswith(b"CUDA_VISIBLE_DEVICES=")), "")
        out[argv[argv.index("--out") + 1]] = (int(pid), gpu)
    return out


def gpus() -> list:
    """[(index, used MiB, free MiB)] from nvidia-smi."""
    q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.free",
                        "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=60)
    return [tuple(int(x) for x in line.split(",")) for line in q.stdout.strip().splitlines()]


def files(arm: dict) -> list:
    return [f"{L}/{arm['name']}.jsonl"] + [f"{L}/{arm['name']}.shard{k}.jsonl"
                                           for k in range(1, arm["shards"])]


def unique_rows(paths: list) -> list:
    """Rows whose key was not seen before, over the files that exist. A key answered
    differently in two files raises ValueError."""
    rows, ans = [], {}
    for path in paths:
        if not os.path.exists(path):
            continue
        for line in open(path, encoding="utf-8"):
            if not line.strip():
                continue
            r = json.loads(line)
            if r["key"] in ans:
                if ans[r["key"]] != r["pred_letter"]:
                    raise ValueError(f"{r['key']} answered differently in two files")
                continue
            ans[r["key"]] = r["pred_letter"]
            rows.append(line.rstrip("\n"))
    return rows


def merged(arm: dict) -> bool:
    """True when the arm's main file alone holds its full population."""
    main = files(arm)[0]
    if not os.path.exists(main) or any(os.path.exists(p) for p in files(arm)[1:]):
        return False
    try:
        return len(unique_rows([main])) == arm["total"]
    except ValueError:
        return False


def merge(arm: dict) -> int:
    """Merge the shard files into the main file if the population is complete; returns the count."""
    rows = unique_rows(files(arm))
    if len(rows) == arm["total"] and arm["shards"] > 1:
        main = files(arm)[0]
        with open(main + ".tmp", "w", encoding="utf-8") as fh:
            fh.write("".join(r + "\n" for r in rows))
        os.replace(main + ".tmp", main)
        for path in files(arm)[1:]:
            if os.path.exists(path):
                os.rename(path, path + ".merged")
    return len(rows)


def launch(arm: dict, k: int, gpu: int) -> subprocess.Popen:
    out = files(arm)[k]
    argv = [f"{arm['venv']}/bin/python", "-u", "eval_lvbench_foveate.py", "--bench", arm["bench"]]
    argv += shlex.split(arm["args"])
    if arm["shards"] > 1:
        argv += ["--shard", f"{k}/{arm['shards']}"]
    argv += ["--out", out]
    env = {key: v for key, v in os.environ.items() if key != "VIRTUAL_ENV"}
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), PYTHONPATH=".")
    with open(out[:-len(".jsonl")] + ".log", "a", encoding="utf-8") as logf:
        return subprocess.Popen(argv, cwd=TL, env=env, stdout=logf, stderr=subprocess.STDOUT,
                                stdin=subprocess.DEVNULL, start_new_session=True)


def main() -> None:
    os.chdir(TL)
    os.makedirs(R, exist_ok=True)
    children = {}     # out path -> Popen launched by this process
    state = {}        # out path -> "pending" | "running" | "exited"
    attempts = {}     # arm name -> relaunch rounds
    done, anomalies = set(), []
    recent = {}       # GPU -> launch times
    last_cpu = (cpu_usec(), time.time())
    log(f"runner started, pid {os.getpid()}")
    while True:
        try:
            plan = json.load(open(PLAN))["arms"]
        except (OSError, ValueError, KeyError) as ex:
            log(f"plan unreadable: {ex!r}")
            time.sleep(LOOP_S)
            continue
        running = evals()
        for out, p in list(children.items()):
            rc = p.poll()
            if rc is None:
                continue
            del children[out]
            state[out] = "pending" if rc < 0 else "exited"
            log(f"exit {out} rc={rc}" + (" (signal: back to pending)" if rc < 0 else ""))
        skip = set()
        if os.path.exists(SKIP):
            skip = {int(x) for x in open(SKIP).read().replace(",", " ").split() if x.isdigit()}

        wanted = []       # (arm, shard) waiting for a GPU, in plan order
        arms_status = []
        for arm in plan:
            name, paths = arm["name"], files(arm)
            if name in done or merged(arm):
                done.add(name)
                arms_status.append({"name": name, "study": arm.get("study", ""), "done": True})
                continue
            for p in paths:
                if p not in running and state.get(p, "running") == "running":
                    state[p] = "pending"      # never seen, or a running shard that vanished
            run = [k for k, p in enumerate(paths) if p in running]
            pend = [k for k, p in enumerate(paths) if p not in running and state[p] == "pending"]
            if not run and not pend:          # every shard has exited: merge or relaunch
                try:
                    n = merge(arm)
                except ValueError as ex:
                    n = -1
                    if name not in anomalies:
                        anomalies.append(name)
                        log(f"ANOMALY {name}: {ex}")
                if n == arm["total"]:
                    done.add(name)
                    log(f"MERGED {name}: {n}/{arm['total']}")
                    arms_status.append({"name": name, "study": arm.get("study", ""), "done": True})
                    continue
                if n >= 0 and attempts.get(name, 0) < RETRIES:
                    attempts[name] = attempts.get(name, 0) + 1
                    log(f"{name}: {n}/{arm['total']} after every shard exited; relaunch round "
                        f"{attempts[name]}")
                    for p in paths:
                        state[p] = "pending"
                    pend = list(range(len(paths)))
                elif n >= 0 and name not in anomalies:
                    anomalies.append(name)
                    log(f"ANOMALY {name}: {n}/{arm['total']} after {RETRIES} relaunch rounds")
            if name not in anomalies:
                wanted += [(arm, k) for k in pend]
            arms_status.append({"name": name, "study": arm.get("study", ""), "done": False,
                                "running": run, "pending": pend})

        now = time.time()
        u = cpu_usec()
        cores = 0.0
        if u is not None and last_cpu[0] is not None and now > last_cpu[1]:
            cores = (u - last_cpu[0]) / ((now - last_cpu[1]) * 1e6)
        last_cpu = (u, now)
        per_gpu = {}
        for _, g in running.values():
            per_gpu[g] = per_gpu.get(g, 0) + 1
        gpu_status = []
        for i, used, free in gpus():
            mine = per_gpu.get(str(i), 0)
            recent[i] = [t for t in recent.get(i, []) if now - t < 600]
            shared = used > mine * PER_EVAL + 4000          # memory my evals cannot explain
            room = free - len(recent[i]) * RESERVE
            if i in skip:
                ok = False
            elif shared:
                ok = mine < SHARED_MINE and room >= SHARED_FREE
            else:
                ok = mine < MAX_MINE and room >= FREE_MIN
            gpu_status.append({"gpu": i, "used": used, "free": free, "mine": mine,
                               "shared": shared, "skip": i in skip, "takes": ok})
        # one launch per eligible GPU per minute, the least loaded GPU first, while CPU allows
        for g in sorted((s for s in gpu_status if s["takes"]), key=lambda s: (s["mine"], -s["free"])):
            if not wanted or cores > CPU_LIMIT:
                break
            arm, k = wanted.pop(0)
            out = files(arm)[k]
            children[out] = launch(arm, k, g["gpu"])
            state[out] = "running"
            recent[g["gpu"]].append(now)
            log(f"launch {arm['name']} shard {k}/{arm['shards']} on GPU {g['gpu']} "
                f"(pid {children[out].pid})")
        status = {"time": time.strftime("%F %T"), "pid": os.getpid(), "gpus": gpu_status,
                  "cpu_cores": round(cores, 1), "running": len(running), "pending": len(wanted),
                  "anomalies": anomalies, "arms": arms_status}
        with open(STATUS + ".tmp", "w") as fh:
            json.dump(status, fh, indent=1)
        os.replace(STATUS + ".tmp", STATUS)
        time.sleep(LOOP_S)


if __name__ == "__main__":
    main()
