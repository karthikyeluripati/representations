"""Sanity simulation (CPU, synthetic): does RepExp-style elliptical-bonus selection change the
violation rate among selected responses, and what determines the sign?

Not a benchmark and not evidence about real agents. It checks whether H1 ("representation-based
exploration increases violations") is a substantive empirical claim or a mechanical consequence of
how diversity selection reweights strategy modes.

Pool: N responses per prompt drawn from K strategy clusters in R^d; one cluster is "violating",
sampled with base rate p. Selection: greedy max of h^T Sigma^-1 h (RepExp's elliptical bonus,
Sigma = lambda*I + sum of selected h h^T), vs uniform random selection. All seeds fixed.
"""
import argparse, json
import numpy as np


def pool(rng, n, k, d, p_viol, spread, sep):
    centers = rng.normal(size=(k, d)) * sep
    probs = np.r_[p_viol, np.full(k - 1, (1 - p_viol) / (k - 1))]
    lab = rng.choice(k, size=n, p=probs)
    h = centers[lab] + rng.normal(size=(n, d)) * spread
    return h - h.mean(0), lab == 0


def elliptical(h, m, lam=1.0):
    d = h.shape[1]
    inv = np.eye(d) / lam
    chosen, free = [], np.ones(len(h), bool)
    for _ in range(m):
        b = np.einsum("ij,jk,ik->i", h, inv, h)
        b[~free] = -np.inf
        i = int(np.argmax(b))
        chosen.append(i); free[i] = False
        v = inv @ h[i]
        inv -= np.outer(v, v) / (1 + h[i] @ v)  # Sherman-Morrison
    return np.array(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/alignment_explore/sim_elliptical.json")
    ap.add_argument("--prompts", type=int, default=400)
    a = ap.parse_args()
    n, m, k, d = 64, 8, 4, 32
    rows = []
    for sep in (0.5, 1.0, 3.0):
        for p in (0.02, 0.05, 0.10, 0.25, 0.50, 0.75):
            rng = np.random.default_rng(0)
            r_sel, r_rand = [], []
            for _ in range(a.prompts):
                h, v = pool(rng, n, k, d, p, 1.0, sep)
                r_sel.append(v[elliptical(h, m)].mean())
                r_rand.append(v[rng.choice(n, m, replace=False)].mean())
            rows.append(dict(sep=sep, p_viol=p, random=float(np.mean(r_rand)),
                             elliptical=float(np.mean(r_sel)),
                             delta=float(np.mean(r_sel) - np.mean(r_rand))))
            print(f"sep={sep:<4} p={p:<5} random={rows[-1]['random']:.3f} "
                  f"elliptical={rows[-1]['elliptical']:.3f} delta={rows[-1]['delta']:+.3f}")
    import os
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(dict(config=dict(n=n, m=m, k=k, d=d, prompts=a.prompts, seed=0), rows=rows),
              open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()
