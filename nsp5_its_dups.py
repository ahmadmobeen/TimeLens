"""How often BOLT's inverse transform sampling would put two of the 64 frames in one second (BOLT
repeats the frame; the harness reads them off inside that second instead, the adaptation
registered in METHODS-LOG, 2026-10-28). Uses the harness's own curve_times: a frame time that is
not a whole second came from a doubled second.

    .venv/bin/python nsp5_its_dups.py
"""
import json

import numpy as np

from eval_lvbench_foveate import curve_times

L, P = "logs/nsp5", "data/t1/priors"
POWER = {"lvb": 2.5}                     # BOLT's exponent; 3 elsewhere


def share(arm, curves, bench):
    z = np.load(curves)
    pre = "lvbench:" if bench == "lvbench" else ""
    hit = tot = 0
    for r in map(json.loads, open(f"{L}/{arm}.jsonl", encoding="utf-8")):
        t = curve_times(z[pre + r["key"]], r["duration_s"], 64, "its", 1.0, 0.0,
                        power=POWER.get(bench, 3.0))
        hit += any(x != int(x) for x in t)
        tot += 1
    return hit, tot


def main():
    for b in ("lvbench", "videomme", "lvb", "mlvu", "cgbench"):
        pre, tag = ("", "") if b == "lvbench" else (f"x1_{b}_", f"{b}_")
        for arm, curves in (("bolt_own", f"{P}/clip_l14/{tag}curves_stem.npz"),
                            ("bolt_its_q8", f"{P}/qwen3vl_emb_8b/{tag}curves_maxopt.npz")):
            h, n = share(pre + arm, curves, b)
            print(f"{b:<9} {arm:<12} {h:5d} of {n:5d} questions ({100 * h / n:.1f}%) have a doubled second")


if __name__ == "__main__":
    main()
