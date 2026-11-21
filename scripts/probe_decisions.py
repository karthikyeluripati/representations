"""Phase 3 (B–D): can frozen LM representations rank candidate actions by branch success better than
the LM's own action preference (mean log-prob)?

Protocol (see docs/experiment_log.md, "Phase 3 probes"):
  * Labels: Monte-Carlo branch success rate (n=8 rollouts of the frozen rollout policy).
  * Objective: within-state pairwise logistic ranking. Pairs with unequal success only. Pair weight = |Δp|
    (reliability), normalised so every *state* has total weight 1, so comparisons are not treated as
    independent samples.
  * Preprocessing (PCA, standardisation, score z-scaling): fit on training rows only.
  * Hyperparameters: GroupKFold by task inside the training tasks. Small fixed grids.
  * Splits: train on `fit`; alpha for the combined score is tuned on `dev` only; `family_holdout` is
    evaluated once. Nested task-grouped CV over `fit` gives an extra out-of-fold estimate.
  * Backbone frozen: only stored embeddings are used.

Usage: python scripts/probe_decisions.py --decisions results/decisions_v1_3b --out results/phase3/probes_3b.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from dare.evaluation.metrics import cluster_bootstrap_ci, pairwise_ranking_accuracy  # noqa: E402
from dare.utils.logging import read_jsonl  # noqa: E402

torch.set_num_threads(8)
VERBS = ["go", "take", "put", "open", "close", "examine", "use", "heat", "cool", "clean", "toggle",
         "inventory", "look", "slice"]


# ----------------------------------------------------------------------------- data
def load(D, action_text_emb=None):
    rows = [r for f in sorted(glob.glob(os.path.join(D, "decisions*.jsonl"))) for r in read_jsonl(f)]
    rows.sort(key=lambda r: (r["state_id"], r["candidate_action"]))
    cache, act, ctx, lp, lps = {}, [], [], [], []
    for r in rows:
        if r["state_id"] not in cache:
            cache[r["state_id"]] = np.load(os.path.join(D, r["representation_file"]))
        z = cache[r["state_id"]]
        i = list(z["candidates"]).index(r["candidate_action"])
        act.append(z["actions"][i]); ctx.append(z["context"]); lp.append(z["logp_mean"][i]); lps.append(z["logp_sum"][i])
    sid = sorted({r["state_id"] for r in rows})
    d = {
        "rows": rows,
        "act": np.asarray(act, np.float32), "ctx": np.asarray(ctx, np.float32),
        "logp": np.asarray(lp), "logp_sum": np.asarray(lps),
        "y": np.array([r["branch_success_rate"] for r in rows]),
        "yrem": -np.array([r["estimated_remaining_actions"] for r in rows]),
        "state": np.array([sid.index(r["state_id"]) for r in rows]),
        "task": np.array([r["task_id"] for r in rows]),
        "scene": np.array([r["task_id"].split("/")[0].split("-")[-1] for r in rows]),
        "family": np.array([r["task_family"] for r in rows]),
        "split": np.array([r["fit_split"] for r in rows]),
        "is_choice": np.array([r["is_policy_choice"] for r in rows]),
        "cand": np.array([r["candidate_action"] for r in rows]),
    }
    v = np.zeros((len(rows), len(VERBS) + 1), np.float32)
    for k, r in enumerate(rows):
        w = r["candidate_action"].split()[0]
        v[k, VERBS.index(w) if w in VERBS else len(VERBS)] = 1
    d["verb"] = v
    if action_text_emb and os.path.exists(action_text_emb):
        z = np.load(action_text_emb)
        m = dict(zip(list(z["candidates"]), z["emb"]))
        d["act_text"] = np.asarray([m[c] for c in d["cand"]], np.float32)
    return d


def pairs(idx, y, state):
    """Within-state ordered pairs (hi, lo) with y[hi] > y[lo]; weights |Δy| normalised per state."""
    hi, lo, w = [], [], []
    by = collections.defaultdict(list)
    for i in idx:
        by[state[i]].append(i)
    for s, ii in by.items():
        P = [(a, b, y[a] - y[b]) for a in ii for b in ii if y[a] > y[b]]
        if not P:
            continue
        tot = sum(p[2] for p in P)
        for a, b, dlt in P:
            hi.append(a); lo.append(b); w.append(dlt / tot)
    return np.array(hi, int), np.array(lo, int), np.array(w)


# ----------------------------------------------------------------------------- models
class Prep:
    """PCA (k comps) + standardisation, fit on training rows only."""

    def __init__(self, k):
        self.k = k

    def fit(self, X):
        k = min(self.k, X.shape[1], len(X) - 1)
        self.pca = PCA(k, random_state=0).fit(X) if X.shape[1] > k else None
        Z = self.pca.transform(X) if self.pca else X
        self.mu, self.sd = Z.mean(0), Z.std(0) + 1e-6
        return self

    def __call__(self, X):
        Z = self.pca.transform(X) if self.pca else X
        return ((Z - self.mu) / self.sd).astype(np.float32)


def fit_linear(F, tr, d, C):
    hi, lo, w = pairs(tr, d["y"], d["state"])
    X = np.vstack([F[hi] - F[lo], F[lo] - F[hi]])
    lab = np.r_[np.ones(len(hi)), np.zeros(len(hi))]
    clf = LogisticRegression(C=C, fit_intercept=False, max_iter=5000).fit(X, lab, sample_weight=np.r_[w, w])
    return lambda G: G @ clf.coef_[0]


class MLP(torch.nn.Module):
    def __init__(self, k, h=64, p=0.2):
        super().__init__()
        self.f = torch.nn.Sequential(torch.nn.Dropout(p), torch.nn.Linear(k, h), torch.nn.ReLU(),
                                     torch.nn.Dropout(p), torch.nn.Linear(h, 1))

    def forward(self, a, c=None):
        return self.f(a).squeeze(-1)


class Proj(torch.nn.Module):
    """Learned low-rank state-action interaction: s = w·a + <P a, Q c>."""

    def __init__(self, k, kc, r=8):
        super().__init__()
        self.w = torch.nn.Linear(k, 1, bias=False)
        self.P = torch.nn.Linear(k, r, bias=False)
        self.Q = torch.nn.Linear(kc, r, bias=False)

    def forward(self, a, c):
        return self.w(a).squeeze(-1) + (self.P(a) * self.Q(c)).sum(-1)


def fit_torch(make, A, Cx, tr, d, wd, seeds=5, epochs=300, lr=3e-3):
    hi, lo, w = pairs(tr, d["y"], d["state"])
    A_t, C_t = torch.tensor(A), torch.tensor(Cx) if Cx is not None else None
    hi_t, lo_t, w_t = torch.tensor(hi), torch.tensor(lo), torch.tensor(w, dtype=torch.float32)
    models = []
    for s in range(seeds):
        torch.manual_seed(s)
        m = make()
        opt = torch.optim.AdamW(m.parameters(), lr=lr, weight_decay=wd)
        for _ in range(epochs):
            m.train()
            sc = m(A_t, C_t)
            loss = (w_t * torch.nn.functional.softplus(-(sc[hi_t] - sc[lo_t]))).sum() / w_t.sum()
            opt.zero_grad(); loss.backward(); opt.step()
        m.eval()
        with torch.no_grad():  # per-seed scale fixed on TRAINING rows only, so scores are call-independent
            tt = torch.as_tensor(np.asarray(tr)); o = m(A_t[tt], C_t[tt] if C_t is not None else None).numpy()
        models.append((m, o.mean(), o.std() + 1e-8))

    def score(A2, C2):
        with torch.no_grad():
            out = [(m(torch.tensor(A2), torch.tensor(C2) if C2 is not None else None).numpy() - mu) / sd
                   for m, mu, sd in models]
        return np.mean(out, 0)
    return score


def build(kind, hp, d, tr):
    """Fit a predictor on training rows `tr`; return score_fn(rows_idx) -> scores for those rows."""
    if kind == "linear":
        feat = d[hp["feat"]]
        prep = Prep(hp["k"]).fit(feat[tr])
        f = fit_linear(prep(feat), tr, d, hp["C"])
        Fall = prep(feat)
        return lambda idx: f(Fall[idx])
    if kind in ("mlp", "mlp_ctx"):
        pa = Prep(hp["k"]).fit(d["act"][tr]); A = pa(d["act"])
        if kind == "mlp_ctx":
            pc = Prep(hp["k"]).fit(d["ctx"][tr]); A = np.hstack([A, pc(d["ctx"])])
        sf = fit_torch(lambda: MLP(A.shape[1]), A, None, tr, d, hp["wd"])
        return lambda idx: sf(A[idx], None)
    if kind == "proj":
        pa = Prep(hp["k"]).fit(d["act"][tr]); pc = Prep(hp["k"]).fit(d["ctx"][tr])
        A, Cx = pa(d["act"]), pc(d["ctx"])
        sf = fit_torch(lambda: Proj(A.shape[1], Cx.shape[1], hp["r"]), A, Cx, tr, d, hp["wd"])
        return lambda idx: sf(A[idx], Cx[idx])
    raise ValueError(kind)


GRIDS = {
    "linear": [{"feat": "act", "k": k, "C": C} for k in (16, 64) for C in (0.01, 0.1, 1.0)],
    "mlp": [{"k": 64, "wd": wd} for wd in (1e-3, 1e-2, 1e-1)],
    "mlp_ctx": [{"k": 32, "wd": wd} for wd in (1e-2, 1e-1)],
    "proj": [{"k": 64, "r": 8, "wd": wd} for wd in (1e-2, 1e-1)],
    "verb": [{"feat": "verb", "k": 10 ** 6, "C": C} for C in (0.1, 1.0)],
    "act_text": [{"feat": "act_text", "k": k, "C": C} for k in (16, 64) for C in (0.01, 0.1, 1.0)],
}
KIND = {"linear": "linear", "verb": "linear", "act_text": "linear", "mlp": "mlp", "mlp_ctx": "mlp_ctx",
        "proj": "proj"}


def group_folds(idx, groups, n):
    g = sorted(set(groups[idx]))
    rng = np.random.RandomState(0)
    rng.shuffle(g)
    fold_of = {x: i % n for i, x in enumerate(g)}
    return [(idx[[fold_of[groups[i]] != f for i in idx]], idx[[fold_of[groups[i]] == f for i in idx]])
            for f in range(n)]


def oof_scores(name, hp, d, idx, groups, n=5):
    out = np.zeros(len(d["y"]))
    for tr, te in group_folds(idx, groups, n):
        out[te] = build(KIND[name], hp, d, tr)(te)
    return out


def select(name, d, tr, groups, n=4):
    best, table = None, []
    for hp in GRIDS[name]:
        s = oof_scores(name, hp, d, tr, groups, n)
        acc = pairwise_ranking_accuracy(s[tr], d["y"][tr], d["state"][tr])
        table.append((hp, acc))
        if best is None or acc > best[1]:
            best = (hp, acc)
    return best[0], table


# ----------------------------------------------------------------------------- evaluation
def _pair_correct(score, tgt, idx, st, gap=0.0):
    """Per informative ordered pair (tgt[hi] - tgt[lo] > gap, same state): 1 / 0.5 (tie) / 0, plus row hi."""
    hi, lo, _ = pairs(idx, tgt, st)
    keep = tgt[hi] - tgt[lo] > gap
    hi, lo = hi[keep], lo[keep]
    ds = score[hi] - score[lo]
    return (ds > 0) + 0.5 * (ds == 0), hi


def evaluate(score, ref, d, idx, n_boot=2000, seed=0):
    """Pooled within-state pairwise accuracy of `score`, and the PAIRED difference vs `ref` (LM logp) on
    the same pairs; 95% percentile CIs from a task-cluster bootstrap (whole tasks resampled)."""
    st, task = d["state"], d["task"]
    T = sorted(set(task[idx]))
    tix = {t: i for i, t in enumerate(T)}
    W = np.random.default_rng(seed).multinomial(len(T), np.ones(len(T)) / len(T), size=n_boot)  # [B, tasks]
    out = {}
    # succ_gap2: only pairs whose success rates differ by >= 2 of 8 rollouts (more reliable labels)
    for tname, tgt, gap in (("succ", d["y"], 0.0), ("succ_gap2", d["y"], 0.2), ("rem", d["yrem"], 0.0)):
        cs, hi = _pair_correct(score, tgt, idx, st, gap)
        cr, _ = _pair_correct(ref, tgt, idx, st, gap)
        if len(hi) == 0:
            out[f"{tname}_acc"] = out[f"{tname}_diff_vs_logp"] = (float("nan"),) * 3
            continue
        g = np.array([tix[task[h]] for h in hi], int)
        n_t = np.bincount(g, minlength=len(T)).astype(float)
        cs_t = np.bincount(g, cs, minlength=len(T)); cr_t = np.bincount(g, cr, minlength=len(T))
        den = W @ n_t
        ok = den > 0
        a_b = (W @ cs_t)[ok] / den[ok]; r_b = (W @ cr_t)[ok] / den[ok]
        a, r = cs_t.sum() / n_t.sum(), cr_t.sum() / n_t.sum()
        out[f"{tname}_acc"] = (float(a), *map(float, np.percentile(a_b, [2.5, 97.5])))
        out[f"{tname}_diff_vs_logp"] = (float(a - r), *map(float, np.percentile(a_b - r_b, [2.5, 97.5])))
        if tname == "succ_gap2":
            out["n_pairs_gap2"] = int(len(hi))
        if tname == "succ":
            out["p_boot_diff_le_0"] = float(np.mean(a_b - r_b <= 0))
            out["n_pairs"] = int(len(hi))
            out["n_informative_states"] = int(len(set(st[hi])))
            out["n_task_clusters"] = int(len(set(task[hi])))
    return out


def disagreement(score, ref, d, idx):
    """On informative pairs where the probe and LM logp order the two actions differently, who is right?"""
    hi, lo, _ = pairs(idx, d["y"], d["state"])
    dp = d["y"][hi] - d["y"][lo]
    ps, pr = score[hi] - score[lo], ref[hi] - ref[lo]
    dis = np.sign(ps) != np.sign(pr)
    out = {"pairs": int(len(hi)), "disagree": int(dis.sum()),
           "probe_right_when_disagree": float((ps[dis] > 0).mean()) if dis.any() else float("nan")}
    for lo_, hi_ in ((0, 0.26), (0.26, 1.01)):
        m = dis & (dp >= lo_) & (dp < hi_)
        out[f"probe_right_when_disagree_dp[{lo_},{hi_})"] = [float((ps[m] > 0).mean()) if m.any() else None, int(m.sum())]
    # overruling the policy's own choice (top-logp candidate)
    ch = d["is_choice"]
    m = dis & (ch[hi] | ch[lo])
    out["disagreements_involving_policy_choice"] = int(m.sum())
    out["probe_right_there"] = float((ps[m] > 0).mean()) if m.any() else None
    return out


def zfit(s, idx):
    mu, sd = s[idx].mean(), s[idx].std() + 1e-8
    return lambda x: (x - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--action-text-emb", default=None)
    ap.add_argument("--models", default="linear,mlp,mlp_ctx,proj,verb")
    ap.add_argument("--n-perm", type=int, default=20)
    args = ap.parse_args()
    t0 = time.time()
    d = load(args.decisions, args.action_text_emb)
    models = args.models.split(",")
    if "act_text" in d and "act_text" not in models:
        models.append("act_text")
    N = len(d["y"])
    I = {s: np.where(d["split"] == s)[0] for s in ("fit", "dev", "family_holdout")}
    fit_scenes = set(d["scene"][I["fit"]])
    I["dev_scene_disjoint"] = np.array([i for i in I["dev"] if d["scene"][i] not in fit_scenes], int)
    I["holdout_scene_disjoint"] = np.array([i for i in I["family_holdout"] if d["scene"][i] not in fit_scenes], int)
    logp = d["logp"]
    R = {"n_rows": N, "protocol": __doc__.strip().splitlines()[2:10], "grids": GRIDS, "models": {}}

    # ---- baselines with no fitted parameters
    for name, s in (("lm_logp_mean", logp), ("lm_logp_sum", d["logp_sum"])):
        R["models"][name] = {sp: evaluate(s, logp, d, I[sp]) for sp in ("fit", "dev", "family_holdout")}

    for name in models:
        print(f"[{time.time()-t0:.0f}s] {name}", flush=True)
        M = {}
        # (1) nested task-grouped CV over fit: outer 5 folds, hyperparameters selected in inner 4 folds
        oof, oof_comb = np.zeros(N), {}
        chosen = []
        for tr, te in group_folds(I["fit"], d["task"], 5):
            hp, _ = select(name, d, tr, d["task"], 4)
            chosen.append(hp)
            sf = build(KIND[name], hp, d, tr)
            oof[te] = sf(te)
            oof_comb[tuple(te)] = (zfit(sf(tr), np.arange(len(tr))), zfit(logp, tr))
        M["fit_nestedcv"] = evaluate(oof, logp, d, I["fit"])
        M["fit_nestedcv"]["hp_chosen_per_fold"] = chosen
        M["fit_nestedcv"]["disagreement"] = disagreement(oof, logp, d, I["fit"])
        # scene-grouped variant (memorisation control): groups = room/scene instead of task
        hp_s, _ = select(name, d, I["fit"], d["task"], 4)
        oof_sc = oof_scores(name, hp_s, d, I["fit"], d["scene"], 5)
        M["fit_scene_grouped_cv"] = evaluate(oof_sc, logp, d, I["fit"])
        # (2) final model: select on fit, train on all fit, evaluate dev / holdout
        hp, table = select(name, d, I["fit"], d["task"], 4)
        M["selected_hp"], M["cv_table"] = hp, [[h, a] for h, a in table]
        sf = build(KIND[name], hp, d, I["fit"])
        s_all = sf(np.arange(N))
        zp, zl = zfit(s_all, I["fit"]), zfit(logp, I["fit"])
        for sp in ("dev", "family_holdout", "dev_scene_disjoint", "holdout_scene_disjoint"):
            if len(I[sp]):
                M[sp] = evaluate(s_all, logp, d, I[sp])
        M["family_holdout"]["disagreement"] = disagreement(s_all, logp, d, I["family_holdout"])
        M["dev"]["disagreement"] = disagreement(s_all, logp, d, I["dev"])
        # (3) combined score z(logp) + alpha * z(probe); alpha tuned on dev only
        alphas = [0, 0.25, 0.5, 1, 2, 4]
        dev_acc = {a: pairwise_ranking_accuracy((zl(logp) + a * zp(s_all))[I["dev"]], d["y"][I["dev"]],
                                                d["state"][I["dev"]]) for a in alphas}
        a_best = max(alphas, key=lambda a: (dev_acc[a], -a))  # ties -> smaller alpha (closer to logp)
        comb = zl(logp) + a_best * zp(s_all)
        M["combined"] = {"alpha_dev_curve": dev_acc, "alpha": a_best,
                         "family_holdout": evaluate(comb, logp, d, I["family_holdout"])}
        # combined on fit-OOF with the dev-chosen alpha (alpha never saw fit-OOF labels)
        comb_oof = np.zeros(N)
        for te, (zp_f, zl_f) in oof_comb.items():
            te = np.array(te)
            comb_oof[te] = zl_f(logp[te]) + a_best * zp_f(oof[te])
        M["combined"]["fit_nestedcv"] = evaluate(comb_oof, logp, d, I["fit"])
        R["models"][name] = M
        print(json.dumps({k: (v["succ_acc"], v["succ_diff_vs_logp"]) for k, v in M.items()
                          if isinstance(v, dict) and "succ_acc" in v}, default=float), flush=True)

    # ---- state-only control: context embedding cannot rank within a state (constant); test state VALUE instead
    from scipy.stats import spearmanr
    from sklearn.linear_model import Ridge
    sv = collections.defaultdict(list)
    for i in range(N):
        sv[d["state"][i]].append(i)
    st_ids = sorted(sv)
    Xs = np.array([d["ctx"][sv[s][0]] for s in st_ids]); ys = np.array([d["y"][sv[s]].mean() for s in st_ids])
    sps = np.array([d["split"][sv[s][0]] for s in st_ids])
    pr = Prep(32).fit(Xs[sps == "fit"])
    rg = Ridge(alpha=10.0).fit(pr(Xs[sps == "fit"]), ys[sps == "fit"])
    R["state_only_control"] = {
        "within_state_acc": "0.5 by construction (identical score for all candidates of a state)",
        "state_value_spearman": {sp: float(spearmanr(rg.predict(pr(Xs[sps == sp])), ys[sps == sp]).statistic)
                                 for sp in ("dev", "family_holdout")},
        "n_states": {sp: int((sps == sp).sum()) for sp in ("dev", "family_holdout")}}

    # ---- label-permutation null for the linear probe (labels shuffled within state, on fit only)
    if args.n_perm and "linear" in models:
        hp = R["models"]["linear"]["selected_hp"]
        rng = np.random.RandomState(0)
        null = []
        y0 = d["y"].copy()
        for _ in range(args.n_perm):
            yp = y0.copy()
            for s in set(d["state"][I["fit"]]):
                ii = np.where(d["state"] == s)[0]
                yp[ii] = rng.permutation(y0[ii])
            d["y"] = yp
            sf = build("linear", hp, d, I["fit"])
            d["y"] = y0
            s_all = sf(np.arange(N))
            null.append(pairwise_ranking_accuracy(s_all[I["family_holdout"]], y0[I["family_holdout"]],
                                                  d["state"][I["family_holdout"]]))
        R["linear_permutation_null_holdout_acc"] = {"mean": float(np.mean(null)),
                                                    "p95": float(np.percentile(null, 95)), "n": args.n_perm}
    R["runtime_s"] = time.time() - t0
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump(R, f, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    print(f"wrote {args.out} in {R['runtime_s']:.0f}s")


if __name__ == "__main__":
    main()
