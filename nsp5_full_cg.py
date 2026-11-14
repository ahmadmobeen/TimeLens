#!/usr/bin/env python3
"""NS-P5 Study Z: CG-Bench's full release. Every main-paper arm on the 8,256 questions outside the
mini set (bench cgbench_fresh, Study S's set), for a runner on its own node (default
logs/nsp5/runner_c). Each arm is the CG-Bench mini arm of the same name with the new questions'
curves; nothing else changes. Arms whose curves do not exist yet are listed and left out; run the
script again once they exist (it never adds an arm twice).

    python3 nsp5_full_cg.py            # dry run: what would be added, what is waiting
    python3 nsp5_full_cg.py --add      # write them into the runner's plan, in launch order
"""
import glob
import json
import os
import re
import subprocess
import sys

TL = os.path.dirname(os.path.abspath(__file__))
PLAN = os.environ.get("NSP5_RUNNER_DIR", "logs/nsp5/runner_c") + "/plan.json"
TOTAL = 8256
THREADS = "--cpu_threads 4 --prefetch 2"

# launch order: Table 1 (longest first), Fig. 4(c, d), Table 2, Fig. 4(b)
ORDER = [
    # Table 1, answer models (uniform and M2 each)
    "h_uniform_q25_72b", "h_q8_kelly_eps0_q25_72b", "h_uniform_q25_32b", "h_q8_kelly_eps0_q25_32b",
    "k_uniform_gemma4_31b", "k_q8_kelly_eps0_gemma4_31b", "uniform_q3", "q8_kelly_eps0_q3",
    # Table 1, published selectors as released
    "bolt_own", "aks_own",
    "l_uniform_plm8b", "l_q8_kelly_eps0_plm8b", "g2_q35_uniform", "q8_kelly_eps0_q35",
    # Fig. 4(c, d): frame budgets (64 and 128 frames for M2 and GMM-EVA's selection exist from Study S)
    "n16_q8_kelly_eps0", "n16_f2c_topk", "n32_q8_kelly_eps0", "n32_f2c_topk",
    "n16_uniform", "n32_uniform", "n128_uniform",
    # Table 2 (ablation)
    "f2c_kelly_eps0", "cal_a2_kelly_eps0", "kelly_clip_eps0", "cal_kelly_blip_eps0", "q8opt_topk_nr",
    "bolt_its_q8", "cal_q8_its_fit", "m_qframe128_q8", "m_qframe_all_q8", "aks_ada_q8",
    "q8_kelly_eps0_tau179", "q8_kelly_eps0_tau716", "q8_kelly_eps010", "q8_kelly_eps025", "q8_kelly_nr",
    # Fig. 4(b): the other temperatures
    "q8_kelly_eps0_tau100", "q8_kelly_eps0_tau250", "q8_kelly_eps0_tau500", "q8_kelly_eps0_tau1000",
]
SHARDS = {"h_uniform_q25_72b": 4, "h_q8_kelly_eps0_q25_72b": 4}


def definitions() -> dict:
    """name -> (venv, args) for every CG-Bench mini arm: the registered listing first, then the
    runner plans (arms added straight to a plan, e.g. Studies G2, H, K, L, M, U)."""
    defs = {}
    listing = subprocess.run(["bash", "nsp5_x1_resume.sh", "arms"], capture_output=True, text=True,
                             check=True).stdout
    for line in listing.splitlines():
        name, bench, venv, _, args, _ = line.split("|")
        if bench == "cgbench":
            defs.setdefault(name, (venv, args))
    for path in sorted(glob.glob("logs/nsp5/runner*/plan*.json")):
        try:
            arms = json.load(open(path))["arms"]
        except (OSError, ValueError, KeyError):
            continue
        for a in arms:
            if a.get("bench") == "cgbench":
                defs.setdefault(a["name"], (a.get("venv", ".venv"), a["args"]))
    return defs


def fresh_arm(base: str, venv: str, args: str) -> tuple:
    """The new questions' arm, and the curves file it needs ('' for uniform sampling)."""
    a = args.replace("/cgbench_curves_", "/cgbench_fresh_curves_")
    a = re.sub(r"\s*--(cpu_threads|prefetch) \S+", "", a).strip() + " " + THREADS
    assert "cgbench_curves" not in a, a
    m = re.search(r"--curves (\S+)", a)
    arm = {"name": f"x1_cgbench_fresh_{base}", "study": "Z", "bench": "cgbench_fresh", "venv": venv,
           "args": a, "shards": SHARDS.get(base, 3), "total": TOTAL, "exclusive": True}
    return arm, (m.group(1) if m else "")


def main() -> None:
    os.chdir(TL)
    defs = definitions()
    plan = json.load(open(PLAN)) if os.path.exists(PLAN) else {"arms": []}
    have = {a["name"] for a in plan["arms"]}
    added, waiting, missing = [], [], []
    for base in ORDER:
        src = f"x1_cgbench_{base}"
        if src not in defs:
            missing.append(src)
            continue
        arm, curves = fresh_arm(base, *defs[src])
        if arm["name"] in have:
            continue
        if curves and not os.path.exists(curves):
            waiting.append(f"{arm['name']} (needs {curves})")
            continue
        added.append(arm)
    for a in added:
        print(f"  {a['name']:<48} {a['venv']:<10} {a['args']}")
    print(f"{len(added)} arms to add; waiting for curves: {len(waiting)}; no mini definition: {missing or 'none'}")
    for w in waiting:
        print("  waiting:", w)
    if "--add" in sys.argv[1:] and added:
        plan["arms"] += added
        os.makedirs(os.path.dirname(PLAN), exist_ok=True)
        with open(PLAN + ".tmp", "w") as fh:
            json.dump(plan, fh, indent=1)
        os.replace(PLAN + ".tmp", PLAN)
        print(f"added to {PLAN}")


if __name__ == "__main__":
    main()
