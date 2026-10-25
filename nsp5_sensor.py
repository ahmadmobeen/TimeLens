"""S0 readout: does the answerer's confidence say whether it has seen the evidence?

Inputs: three arms of eval_lvbench_foveate.py with p_letters logged, on the same
questions -- evidence seen (oracle gauss_s15), evidence not seen at the same
concentration (offpeak_s60), and uniform.

  .venv/bin/python nsp5_sensor.py --seen logs/nsp5/s0_gauss_s15.jsonl \\
      --unseen logs/nsp5/s0_offpeak_s60.jsonl --uniform logs/nsp5/s0_uniform.jsonl

Gate (pre-registered): paired win rate of confidence, seen vs unseen, >= 0.70.
"""
from __future__ import annotations

import argparse

import numpy as np

from nsp5_compare import boot_ci, load
from probe_hidden_states import auroc


def conf(r: dict) -> float:
    return float(max(r["p_letters"]))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--seen", required=True)
    ap.add_argument("--unseen", required=True)
    ap.add_argument("--uniform", required=True)
    args = ap.parse_args()
    S, U, Q = load(args.seen), load(args.unseen), load(args.uniform)
    keys = sorted(k for k in set(S) & set(U) & set(Q)
                  if "p_letters" in S[k] and "p_letters" in U[k] and "p_letters" in Q[k])
    cs, cu = np.array([conf(S[k]) for k in keys]), np.array([conf(U[k]) for k in keys])
    win = (cs > cu) + 0.5 * (cs == cu)
    lo, hi = boot_ci(win)
    print(f"n = {len(keys)} questions with all three arms")
    print(f"accuracy: seen {np.mean([S[k]['correct'] for k in keys]):.3f} | unseen "
          f"{np.mean([U[k]['correct'] for k in keys]):.3f} | uniform "
          f"{np.mean([Q[k]['correct'] for k in keys]):.3f}")
    print(f"\nGATE  paired win rate, conf(seen) > conf(unseen): {win.mean():.3f} "
          f"[{lo:.3f}, {hi:.3f}]   (>= 0.70 usable, <= 0.60 not)")
    lab = np.r_[np.ones(len(cs)), np.zeros(len(cu))]
    print(f"      pooled AUROC, seen vs unseen: {auroc(np.r_[cs, cu], lab):.3f}")

    print("\nlikelihood ratio P(conf | seen) / P(conf | unseen) by confidence bin")
    edges = np.quantile(np.r_[cs, cu], [0, .2, .4, .6, .8, 1.0])
    for a, b in zip(edges[:-1], edges[1:]):
        ps = np.mean((cs >= a) & (cs <= b))
        pu = np.mean((cu >= a) & (cu <= b))
        print(f"  conf [{a:.2f}, {b:.2f}]   seen {ps:.2f}   unseen {pu:.2f}   ratio "
              f"{ps / pu if pu else float('inf'):.2f}")

    cq = np.array([conf(Q[k]) for k in keys])
    ok = np.array([Q[k]["correct"] for k in keys], float)
    print(f"\nuniform arm: AUROC(conf -> correct) {auroc(cq, ok):.3f}; accuracy among the "
          f"30 % most confident {ok[cq >= np.quantile(cq, .7)].mean():.3f} vs "
          f"{ok.mean():.3f} overall")


if __name__ == "__main__":
    main()
