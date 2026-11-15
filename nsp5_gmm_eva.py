#!/usr/bin/env python3
"""GMM-EVA's event-aware visual allocation (Lu et al., PRCV 2026, arXiv 2607.12557), re-implemented
from the paper for NS-P5, since no code is released.

1. One score s_t per second becomes a weight w_t = (s_t - min s) / sum_u (s_u - min s) (Eq. 2).
2. A Gaussian mixture over the time axis is fitted by EM with these weights (Eqs. 3-6):
   C = floor(T / 30) components start at mu_j = j T / (C + 1), sigma_j = T / (2C), pi_j = 1 / C
   (Eq. 4). After each M-step, two components whose 1-sigma intervals overlap with IoU > 0.7 merge
   (Eq. 7). EM stops when the weighted log-likelihood changes by less than 1e-5, or after 1000
   iterations.
3. The selected frames, in descending order of score, are each assigned to their most probable
   component (Eq. 8). The first frame of each component is primary (256 tokens in the paper), every
   other frame secondary (64 tokens): a 4 : 1 ratio of visual tokens.

Not specified in the paper, chosen here: the 1-sigma IoU is the IoU of the intervals
[mu - sigma, mu + sigma]; all pairs above the threshold merge greedily, largest IoU first, after each
M-step; sigma is floored at 0.5 s; a component whose weight falls below 1e-9 is dropped; frames tied
in score keep their time order.

    .venv/bin/python nsp5_gmm_eva.py <curves.npz> [...]   # writes <curves>_gmm.npz beside each file
"""
from __future__ import annotations

import sys
from multiprocessing import Pool

import numpy as np

DT_C, EPS_M, EPS_L, I_MAX, SIG_MIN = 30.0, 0.7, 1e-5, 1000, 0.5


def weights(s) -> np.ndarray:
    s = np.asarray(s, float)
    w = s - s.min()
    total = w.sum()
    return w / total if total > 0 else np.full(len(s), 1.0 / len(s))


def _log_gauss(t: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> np.ndarray:
    z = (t[:, None] - mu[None, :]) / sig[None, :]
    return -0.5 * z * z - np.log(sig)[None, :] - 0.5 * np.log(2 * np.pi)


def _merge(pi: np.ndarray, mu: np.ndarray, sig: np.ndarray) -> tuple:
    """Merge pairs whose 1-sigma intervals overlap with IoU > EPS_M, largest IoU first (Eq. 7)."""
    while len(mu) > 1:
        lo, hi = mu - sig, mu + sig
        inter = np.clip(np.minimum(hi[:, None], hi[None, :]) - np.maximum(lo[:, None], lo[None, :]),
                        0, None)
        union = np.maximum(hi[:, None], hi[None, :]) - np.minimum(lo[:, None], lo[None, :])
        iou = inter / np.maximum(union, 1e-12)
        np.fill_diagonal(iou, 0.0)
        i, j = np.unravel_index(int(np.argmax(iou)), iou.shape)
        if iou[i, j] <= EPS_M:
            break
        p = pi[i] + pi[j]
        m = (mu[i] * pi[i] + mu[j] * pi[j]) / p
        v = (pi[i] * sig[i] ** 2 + pi[j] * sig[j] ** 2) / p + pi[i] * pi[j] * (mu[i] - mu[j]) ** 2 / p ** 2
        keep = [k for k in range(len(mu)) if k not in (i, j)]
        pi, mu, sig = np.append(pi[keep], p), np.append(mu[keep], m), np.append(sig[keep], np.sqrt(v))
    return pi, mu, sig


def fit(s) -> tuple:
    """(pi, mu, sigma, iterations) of the weighted mixture over the seconds of one video."""
    w = weights(s)
    T = len(w)
    t = np.arange(T, dtype=float)
    C = max(1, int(T // DT_C))
    j = np.arange(1, C + 1, dtype=float)
    pi, mu, sig = np.full(C, 1.0 / C), j * T / (C + 1), np.full(C, max(T / (2 * C), SIG_MIN))
    prev, it = -np.inf, 0
    for it in range(1, I_MAX + 1):
        lg = np.log(pi)[None, :] + _log_gauss(t, mu, sig)                 # T x C
        mx = lg.max(1, keepdims=True)
        lse = mx[:, 0] + np.log(np.exp(lg - mx).sum(1))
        ll = float((w * lse).sum())
        if abs(ll - prev) < EPS_L:
            break
        prev = ll
        g = np.exp(lg - lse[:, None])                                       # responsibilities
        nk = (w[:, None] * g).sum(0)
        ok = nk > 1e-9
        g, nk = g[:, ok], nk[ok]
        pi = nk / nk.sum()
        mu = (w[:, None] * g * t[:, None]).sum(0) / nk
        var = (w[:, None] * g * (t[:, None] - mu[None, :]) ** 2).sum(0) / nk
        sig = np.sqrt(np.maximum(var, SIG_MIN ** 2))
        pi, mu, sig = _merge(pi, mu, sig)
    return pi, mu, sig, it


def tiers(times, s, params) -> np.ndarray:
    """1 for a primary frame and 0 for a secondary one, for frames at `times` (seconds) on the score
    curve s, with params = (pi, mu, sigma) from fit(s): the frames in descending order of score,
    each assigned to its most probable component, the first frame of each component primary
    (GMM-EVA, Eq. 8 and Sec. 3.3)."""
    pi, mu, sig = (np.asarray(x, float) for x in params)
    times = np.asarray(times, float)
    secs = np.clip(np.floor(times).astype(int), 0, len(s) - 1)
    comp = (np.log(pi)[None, :] + _log_gauss(secs.astype(float), mu, sig)).argmax(1)
    order = sorted(range(len(secs)), key=lambda k: (-float(s[secs[k]]), times[k]))
    seen, out = set(), np.zeros(len(secs), int)
    for k in order:
        if comp[k] not in seen:
            seen.add(int(comp[k]))
            out[k] = 1
    return out


def token_shares(tier: np.ndarray, budget: float, ratio: float = 4.0) -> np.ndarray:
    """Visual tokens per frame: a primary gets `ratio` times a secondary, and the shares sum to the
    budget (the paper's 256 : 64 tiers, scaled to the harness's shared budget)."""
    w = np.where(np.asarray(tier) > 0, ratio, 1.0)
    return budget * w / w.sum()


def _one(job: tuple) -> tuple:
    key, s = job
    pi, mu, sig, it = fit(s)
    return key, np.stack([pi, mu, sig]), it


def main(paths: list) -> None:
    for path in paths:
        z = np.load(path)
        jobs = [(k, z[k].astype(float)) for k in z.files]
        with Pool(min(96, len(jobs))) as pool:
            res = pool.map(_one, jobs, chunksize=4)
        out = {k: p for k, p, _ in res}
        its = [it for _, _, it in res]
        comps = [p.shape[1] for p in out.values()]
        dst = path[:-4] + "_gmm.npz"
        np.savez_compressed(dst, **out)
        print(f"{dst}: {len(out)} questions; components median {int(np.median(comps))} "
              f"(range {min(comps)}-{max(comps)}); EM iterations median {int(np.median(its))}, "
              f"max {max(its)}, at the cap {sum(i >= I_MAX for i in its)}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1:])
