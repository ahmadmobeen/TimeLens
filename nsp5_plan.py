#!/usr/bin/env python3
"""Add registered arms to nsp5_runner.py's plan. Every arm printed by `bash nsp5_x1_resume.sh arms`
whose name matches the pattern joins logs/nsp5/runner/plan.json under the given study label,
after the arms already there (the runner launches in plan order). Run it only once the study's
registration is on GitHub. Shards per arm follow the benchmark's size.

    python3 nsp5_plan.py C 'pub32_'
"""
import json
import os
import re
import subprocess
import sys

TL = os.path.dirname(os.path.abspath(__file__))
PLAN = "logs/nsp5/runner/plan.json"
SHARDS = {"lvbench": 2, "videomme": 3, "lvb": 2, "mlvu": 2, "cgbench": 3}


def main() -> None:
    os.chdir(TL)
    study, pattern = sys.argv[1], re.compile(sys.argv[2])
    listing = subprocess.run(["bash", "nsp5_x1_resume.sh", "arms"], capture_output=True,
                             text=True, check=True).stdout
    plan = json.load(open(PLAN)) if os.path.exists(PLAN) else {"arms": []}
    have = {a["name"] for a in plan["arms"]}
    added = []
    for line in listing.splitlines():
        name, bench, venv, prereq, args, total = line.split("|")
        if not pattern.search(name) or name in have:
            continue
        if prereq and not os.path.exists(prereq):
            sys.exit(f"{name}: missing {prereq}; nothing added")
        added.append({"name": name, "study": study, "bench": bench, "venv": venv,
                      "args": args, "shards": SHARDS[bench], "total": int(total)})
    plan["arms"] += added
    os.makedirs(os.path.dirname(PLAN), exist_ok=True)
    with open(PLAN + ".tmp", "w") as fh:
        json.dump(plan, fh, indent=1)
    os.replace(PLAN + ".tmp", PLAN)
    print(f"study {study}: {len(added)} arms added: {' '.join(a['name'] for a in added)}")


if __name__ == "__main__":
    main()
