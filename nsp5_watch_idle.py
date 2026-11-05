#!/usr/bin/env python3
"""Wake-up watcher for NS-P5's autonomous loop (the owner's instruction of 2026-11-05: keep the
GPUs busy, register and launch the next study when one idles). Run it as a background task; it
exits, which re-invokes the session that started it, on the first of:

  STUDY_DONE  every arm of a study in the runner's plan is merged (each study reported once);
  ANOMALY     the runner reports an anomaly, or its status is older than STALE_MIN minutes;
  IDLE        a GPU outside skip_gpus has used under 1 GB for IDLE_MIN minutes while the runner
              has nothing pending (time to register the next study); a GPU idle that long while
              shards are pending is reported as ANOMALY instead;
  QUEUE_LOW   at most LOW_PENDING shards wait to launch, reported once per plan size (time to
              register the next study before GPUs idle);
  HEARTBEAT   HEARTBEAT_H hours without any of these.

It prints the event and a short report: each GPU's memory and evals, and each study's progress.
Reported studies and anomalies are remembered in logs/nsp5/runner/watch_state.json.

    cd research/TimeLens && python3 -u nsp5_watch_idle.py
"""
import json
import os
import subprocess
import sys
import time

TL = os.path.dirname(os.path.abspath(__file__))
R = "logs/nsp5/runner"
STATUS, SKIP, STATE = f"{R}/status.json", f"{R}/skip_gpus", f"{R}/watch_state.json"
IDLE_MIN, STALE_MIN, HEARTBEAT_H, POLL_S, IDLE_MIB, LOW_PENDING = 10, 10, 6, 60, 1000, 6


def load(path: str, default):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return default


def report(event: str, detail: str, status: dict) -> None:
    print(f"{time.strftime('%F %T')} {event}: {detail}")
    for g in status.get("gpus", []):
        print(f"  GPU {g['gpu']}: used {g['used']} MiB, my evals {g['mine']}"
              f"{', shared' if g['shared'] else ''}{', skipped' if g['skip'] else ''}")
    studies = {}
    for a in status.get("arms", []):
        s = studies.setdefault(a.get("study") or "?", [0, 0])
        s[0] += a["done"]
        s[1] += 1
    for s, (d, n) in sorted(studies.items()):
        print(f"  study {s}: {d}/{n} arms merged")
    print(f"  runner: {status.get('running', '?')} evals running, {status.get('pending', '?')} "
          f"shards pending, anomalies {status.get('anomalies', [])}")
    sys.stdout.flush()


def main() -> None:
    os.chdir(TL)
    seen = load(STATE, {"reported_studies": [], "reported_anomalies": []})
    start, idle_since = time.time(), {}
    while True:
        status = load(STATUS, {})
        now = time.time()
        # the runner must be alive and writing
        try:
            age = (now - os.path.getmtime(STATUS)) / 60
        except OSError:
            age = 1e9
        if age > STALE_MIN:
            report("ANOMALY", f"runner status is {age:.0f} min old (runner down?)", status)
            return
        new_anom = [a for a in status.get("anomalies", []) if a not in seen["reported_anomalies"]]
        if new_anom:
            seen["reported_anomalies"] += new_anom
            json.dump(seen, open(STATE, "w"))
            report("ANOMALY", f"runner anomalies {new_anom}", status)
            return
        # a study whose arms are all merged
        studies = {}
        for a in status.get("arms", []):
            studies.setdefault(a.get("study") or "?", []).append(a["done"])
        finished = [s for s, d in studies.items() if all(d) and s not in seen["reported_studies"]]
        if finished:
            seen["reported_studies"] += finished
            json.dump(seen, open(STATE, "w"))
            report("STUDY_DONE", f"studies {finished} complete", status)
            return
        # few shards left to launch: the next study should be registered now
        n_arms = len(status.get("arms", []))
        if n_arms and status.get("pending", 0) <= LOW_PENDING and seen.get("low_queue_for") != n_arms:
            seen["low_queue_for"] = n_arms
            json.dump(seen, open(STATE, "w"))
            report("QUEUE_LOW", f"{status.get('pending', 0)} shards left to launch", status)
            return
        # idle GPUs
        skip = set()
        if os.path.exists(SKIP):
            skip = {int(x) for x in open(SKIP).read().replace(",", " ").split() if x.isdigit()}
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True, timeout=60)
        for line in q.stdout.strip().splitlines():
            i, used = (int(x) for x in line.split(","))
            if used < IDLE_MIB and i not in skip:
                idle_since.setdefault(i, now)
            else:
                idle_since.pop(i, None)
        long_idle = [i for i, t in idle_since.items() if now - t >= IDLE_MIN * 60]
        if long_idle:
            if status.get("pending", 0) > 0:
                report("ANOMALY", f"GPUs {long_idle} idle {IDLE_MIN}+ min while shards are pending", status)
            else:
                report("IDLE", f"GPUs {long_idle} idle {IDLE_MIN}+ min and nothing pending", status)
            return
        if now - start >= HEARTBEAT_H * 3600:
            report("HEARTBEAT", f"{HEARTBEAT_H} h without an event", status)
            return
        time.sleep(POLL_S)


if __name__ == "__main__":
    main()
