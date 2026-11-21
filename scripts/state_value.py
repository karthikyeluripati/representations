"""State-value feasibility (pivot Stage 1–2): is future task success decodable from frozen LM *state*
representations beyond simple, deployable baselines, and does it transfer to unseen tasks/families?

Target ("state promise"), per decision state s at step t:
  v_mean(s)   = mean over the 4 sampled candidates a of p̂(success | s, a), where p̂ is from 8 rollouts:
                take a, then follow the frozen LM policy at T=0.5 until won or the 50-step horizon,
                counted from episode reset. Success before a FIXED horizon; no efficiency term.
  v_choice(s) = p̂(success | s, policy's greedy choice)  (secondary target).
Labels use future rollouts (that is what a value label is); no feature does.

Deployable features (all observable at decision time):
  step       : t (remaining budget = 50 - t, so it is collinear)
  observable : step, step², family one-hot (parsed from the objective template; an unseen family is all
               zeros), #admissible, LM policy entropy / top-1 margin / max mean-logp over ALL admissible
               actions (logged), history-derived progress proxies (holding an object, fraction of objective
               words seen in observations, invalid-action count, repeated-observation rate)
  emb        : frozen last-layer context embedding at the final prompt token (stored at collection)

Predictors: A step (isotonic), B observable (ridge), C emb (PCA + ridge), D emb (small MLP),
            E observable + emb (ridge).
Evaluation protocols (development, no held-out family used):
  cv_task   : nested 5-fold task-grouped CV over fit tasks (unseen tasks, seen families)
  lofo      : leave-one-family-out over the 5 development families (fit + dev) (unseen families)
Final test, run once: train on fit + dev, test on `family_holdout` (pick_two_obj_and_place).
Metrics: Spearman, MSE, ECE, within-step-bin pairwise ranking, step<40 subset, paired Δ vs best
non-embedding baseline (chosen on development CV, not on test); task-cluster bootstrap CIs.

Usage: python scripts/state_value.py --decisions results/decisions_v1_3b --out results/phase3/state_value_3b.json
"""
from __future__ import annotations

import argparse
import collections
import glob
import json
import os
import re
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import spearmanr  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.isotonic import IsotonicRegression  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402

from dare.environments.alfworld_env import TASK_FAMILIES  # noqa: E402
from dare.evaluation.metrics import expected_calibration_error, pairwise_ranking_accuracy  # noqa: E402
from dare.utils.logging import load_run, read_jsonl  # noqa: E402

torch.set_num_threads(8)
HORIZON = 50
FAMS = [f for f in TASK_FAMILIES]
STOP = {"the", "a", "an", "in", "on", "with", "and", "put", "some", "it", "examine", "look", "at", "under",
        "two", "clean", "hot", "cool", "find", "then", "place", "of", "to", "into"}


def family_from_objective(obj: str) -> str | None:
    """Map the goal sentence to a task family by its template (observable at deployment)."""
    o = obj.lower()
    if "two" in o:
        return "pick_two_obj_and_place"
    if o.startswith("examine") or o.startswith("look at"):
        return "look_at_obj_in_light"
    if "clean" in o:
        return "pick_clean_then_place_in_recep"
    if "hot" in o or "heat" in o:
        return "pick_heat_then_place_in_recep"
    if "cool" in o:
        return "pick_cool_then_place_in_recep"
    return "pick_and_place_simple"


def load(D):
    rows = [r for f in sorted(glob.glob(os.path.join(D, "decisions*.jsonl"))) for r in read_jsonl(f)]
    trs = {}
    for sd in sorted(glob.glob(os.path.join(D, "source*"))):
        _, t = load_run(sd)
        for x in t:
            trs[(x["episode_id"], x["step"])] = x
    by = collections.defaultdict(list)
    for r in rows:
        by[r["state_id"]].append(r)
    S = []
    for sid, rs in sorted(by.items()):
        r0 = rs[0]
        ep, t = r0["history_ref"]["episode_id"], r0["step"]
        cur = trs[(ep, t)]
        hist = [trs[(ep, k)] for k in range(t)]
        lp = np.array(cur["lm_logp"], float)
        p = np.exp(lp - lp.max()); p /= p.sum()
        srt = np.sort(lp)[::-1]
        obs_hist = [cur["observation"]] + [h["next_observation"] for h in hist]
        holding = 0
        for h in hist:
            if h["next_observation"].startswith("You pick up"):
                holding = 1
            if h["next_observation"].startswith("You put") or h["next_observation"].startswith("You move"):
                holding = 0
        words = [w for w in re.findall(r"[a-z]+", r0["objective"].lower()) if w not in STOP]
        seen = " ".join(obs_hist[1:] + [cur["observation"]]).lower()
        fam_obs = family_from_objective(r0["objective"])
        z = np.load(os.path.join(D, r0["representation_file"]))
        S.append({
            "state_id": sid, "task": r0["task_id"], "family": r0["task_family"], "split": r0["fit_split"],
            "step": t, "v_mean": float(np.mean([r["branch_success_rate"] for r in rs])),
            "v_choice": float([r["branch_success_rate"] for r in rs if r["is_policy_choice"]][0]),
            "n_roll": int(sum(r["rollout_count"] for r in rs)),
            "fam_obs": fam_obs,
            "obs_feats": [t / HORIZON, (t / HORIZON) ** 2, len(cur["candidates"]) / 50,
                          float(-(p * np.log(p + 1e-12)).sum()), float(srt[0] - srt[1]) if len(srt) > 1 else 0.0,
                          float(srt[0]), holding,
                          float(np.mean([w in seen for w in words])) if words else 0.0,
                          sum(h["next_observation"] == "Nothing happens." for h in hist) / max(t, 1),
                          1 - len(set(obs_hist)) / len(obs_hist)]
                         + [float(fam_obs == f) for f in FAMS if f != "pick_two_obj_and_place"],
            "ctx": z["context"].astype(np.float32),
        })
    fam_ok = np.mean([s["fam_obs"] == s["family"] for s in S])
    return S, float(fam_ok)


OBS_NAMES = ["step", "step2", "n_admissible", "policy_entropy", "top1_margin", "max_logp", "holding",
             "objective_words_seen", "invalid_rate", "repeat_obs_rate"] + \
            [f"fam:{f}" for f in FAMS if f != "pick_two_obj_and_place"]


# ----------------------------------------------------------------------------- predictors
class Std:
    def fit(self, X):
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-6
        return self

    def __call__(self, X):
        return (X - self.mu) / self.sd


def fit_predict(kind, hp, S, tr, te):
    y = np.array([s["v_mean"] for s in S])
    step = np.array([s["step"] for s in S], float)
    O = np.array([s["obs_feats"] for s in S], float)
    C = np.array([s["ctx"] for s in S])
    if kind == "A_step":
        m = IsotonicRegression(increasing=False, out_of_bounds="clip").fit(step[tr], y[tr])
        return m.predict(step[te])
    if kind == "B_obs":
        st = Std().fit(O[tr])
        return Ridge(alpha=hp["alpha"]).fit(st(O[tr]), y[tr]).predict(st(O[te]))
    if kind in ("C_emb", "E_obs_emb", "D_mlp"):
        k = min(hp["k"], len(tr) - 1)
        pca = PCA(k, random_state=0).fit(C[tr])
        st = Std().fit(pca.transform(C[tr]))
        Z = lambda ii: st(pca.transform(C[ii]))  # noqa: E731
        if kind == "C_emb":
            return Ridge(alpha=hp["alpha"]).fit(Z(tr), y[tr]).predict(Z(te))
        if kind == "E_obs_emb":
            so = Std().fit(O[tr])
            X = lambda ii: np.hstack([so(O[ii]), Z(ii)])  # noqa: E731
            return Ridge(alpha=hp["alpha"]).fit(X(tr), y[tr]).predict(X(te))
        Xtr, Xte = torch.tensor(Z(tr), dtype=torch.float32), torch.tensor(Z(te), dtype=torch.float32)
        ytr = torch.tensor(y[tr], dtype=torch.float32)
        preds = []
        for seed in range(5):
            torch.manual_seed(seed)
            net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(Xtr.shape[1], 32), torch.nn.ReLU(),
                                      torch.nn.Linear(32, 1))
            opt = torch.optim.AdamW(net.parameters(), lr=3e-3, weight_decay=hp["wd"])
            for _ in range(300):
                net.train()
                loss = torch.nn.functional.binary_cross_entropy_with_logits(net(Xtr).squeeze(-1), ytr)
                opt.zero_grad(); loss.backward(); opt.step()
            net.eval()
            with torch.no_grad():
                preds.append(torch.sigmoid(net(Xte).squeeze(-1)).numpy())
        return np.mean(preds, 0)
    raise ValueError(kind)


GRIDS = {"A_step": [{}], "B_obs": [{"alpha": a} for a in (1.0, 10.0, 100.0)],
         "C_emb": [{"k": k, "alpha": a} for k in (16, 64) for a in (10.0, 100.0, 1000.0)],
         "D_mlp": [{"k": 32, "wd": wd} for wd in (1e-2, 1e-1)],
         "E_obs_emb": [{"k": k, "alpha": a} for k in (16, 64) for a in (10.0, 100.0, 1000.0)]}


def folds(idx, groups, n, seed=0):
    g = sorted(set(groups[idx]))
    np.random.RandomState(seed).shuffle(g)
    f = {x: i % n for i, x in enumerate(g)}
    return [(idx[np.array([f[groups[i]] != k for i in idx])], idx[np.array([f[groups[i]] == k for i in idx])])
            for k in range(n)]


def select(kind, S, tr, task):
    y = np.array([s["v_mean"] for s in S])
    best = None
    for hp in GRIDS[kind]:
        oof = np.zeros(len(S))
        for a, b in folds(tr, task, 4):
            oof[b] = fit_predict(kind, hp, S, a, b)
        err = float(np.mean((oof[tr] - y[tr]) ** 2))
        if best is None or err < best[1]:
            best = (hp, err)
    return best[0]


# ----------------------------------------------------------------------------- metrics
def bins_of(step):
    return np.minimum(np.asarray(step) // 10, 4)


def metrics(pred, idx, S, ref=None, n_boot=2000):
    y = np.array([s["v_mean"] for s in S])
    yc = np.array([s["v_choice"] for s in S])
    step = np.array([s["step"] for s in S])
    task = np.array([s["task"] for s in S])
    p = np.clip(pred, 0, 1)

    def stats(ii, pp):
        ii = idx[ii]
        sp = spearmanr(pp[ii], y[ii]).statistic if len(set(y[ii])) > 1 and len(set(pp[ii])) > 1 else np.nan
        wb = pairwise_ranking_accuracy(pp[ii], y[ii], bins_of(step[ii]))  # matched-step (same 10-step bin)
        early = ii[step[ii] < 40]
        sp40 = spearmanr(pp[early], y[early]).statistic if len(early) > 3 and len(set(y[early])) > 1 and len(set(pp[early])) > 1 else np.nan
        return np.array([sp, np.mean((pp[ii] - y[ii]) ** 2), wb, sp40])
    base = stats(np.arange(len(idx)), p)
    T = sorted(set(task[idx]))
    tpos = {t: np.where(task[idx] == t)[0] for t in T}
    rng = np.random.default_rng(0)
    B, D = [], []
    pr = np.clip(ref, 0, 1) if ref is not None else None
    for _ in range(n_boot):
        ii = np.concatenate([tpos[T[j]] for j in rng.integers(0, len(T), len(T))])
        s1 = stats(ii, p)
        B.append(s1)
        if pr is not None:
            D.append(s1 - stats(ii, pr))
    B = np.array(B)
    ci = lambda M, j: [float(np.nanpercentile(M[:, j], 2.5)), float(np.nanpercentile(M[:, j], 97.5))]  # noqa: E731
    names = ["spearman", "mse", "within_step_bin_pairacc", "spearman_step_lt40"]
    out = {n: [float(base[j]), *ci(B, j)] for j, n in enumerate(names)}
    out["ece"] = float(expected_calibration_error(p[idx], y[idx], 5))
    out["spearman_vs_v_choice"] = float(spearmanr(p[idx], yc[idx]).statistic)
    out["n_states"], out["n_tasks"] = int(len(idx)), int(len(T))
    out["n_states_step_lt40"] = int((step[idx] < 40).sum())
    if pr is not None:
        D = np.array(D)
        out["paired_delta_vs_ref"] = {n: [float(base[j] - stats(np.arange(len(idx)), pr)[j]), *ci(D, j)]
                                      for j, n in enumerate(names)}
    return out


def run_protocol(name, S, kinds, splits):
    """splits: list of (train_idx, test_idx). Hyperparameters selected inside each training set."""
    task = np.array([s["task"] for s in S])
    preds = {k: np.full(len(S), np.nan) for k in kinds}
    chosen = collections.defaultdict(list)
    for tr, te in splits:
        for k in kinds:
            hp = select(k, S, tr, task)
            chosen[k].append(hp)
            preds[k][te] = fit_predict(k, hp, S, tr, te)
    return preds, dict(chosen)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--decisions", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--final-test", action="store_true", help="also evaluate on family_holdout (run once)")
    a = ap.parse_args()
    S, fam_ok = load(a.decisions)
    N = len(S)
    split = np.array([s["split"] for s in S]); fam = np.array([s["family"] for s in S])
    task = np.array([s["task"] for s in S]); y = np.array([s["v_mean"] for s in S])
    step = np.array([s["step"] for s in S])
    kinds = ["A_step", "B_obs", "C_emb", "D_mlp", "E_obs_emb"]
    R = {"n_states": N, "family_from_objective_accuracy": fam_ok, "obs_features": OBS_NAMES,
         "target_summary": {"mean": float(y.mean()), "frac_zero": float((y == 0).mean()),
                            "frac_step_ge_40": float((step >= 40).mean()),
                            "mean_value_step_ge_40": float(y[step >= 40].mean()),
                            "rollouts_per_state": sorted({s["n_roll"] for s in S})},
         "label_noise_floor_mse": float(np.mean(y * (1 - y) / np.array([s["n_roll"] for s in S]))),
         "protocols": {}}
    fit = np.where(split == "fit")[0]
    devfam = np.where(split != "family_holdout")[0]
    protos = {"cv_task": [(tr, te) for tr, te in folds(fit, task, 5)],
              "lofo": [(devfam[fam[devfam] != f], devfam[fam[devfam] == f]) for f in sorted(set(fam[devfam]))]}
    if a.final_test:
        protos["final_test_family_holdout"] = [(devfam, np.where(split == "family_holdout")[0])]
    for pname, sp in protos.items():
        preds, chosen = run_protocol(pname, S, kinds, sp)
        idx = np.concatenate([te for _, te in sp])
        # strongest non-embedding baseline, chosen on DEVELOPMENT CV (cv_task MSE), never on the test set
        if pname == "cv_task":
            ref_kind = min(["A_step", "B_obs"], key=lambda k: np.mean((np.clip(preds[k][idx], 0, 1) - y[idx]) ** 2))
            R["ref_baseline_chosen_on_cv_task"] = ref_kind
        ref_kind = R["ref_baseline_chosen_on_cv_task"]
        P = {}
        for k in kinds:
            P[k] = metrics(preds[k], idx, S, ref=preds[ref_kind] if k != ref_kind else None)
            P[k]["hp"] = chosen[k]
            # step-only (isotonic) is piecewise constant, so it ties within step bins; the matched-step
            # comparison that matters is embedding vs the observable-feature baseline B_obs
            if k in ("C_emb", "D_mlp", "E_obs_emb"):
                P[k]["paired_delta_vs_B_obs"] = metrics(preds[k], idx, S, ref=preds["B_obs"])["paired_delta_vs_ref"]
        # per-family breakdown for lofo (transfer to each unseen family)
        if pname == "lofo":
            P["per_family_spearman"] = {f: {k: float(spearmanr(preds[k][fam == f], y[fam == f]).statistic)
                                            for k in kinds} for f in sorted(set(fam[devfam]))}
        R["protocols"][pname] = P
        print(pname, {k: (round(P[k]["spearman"][0], 2), round(P[k]["mse"][0], 4),
                          round(P[k]["within_step_bin_pairacc"][0], 2)) for k in kinds}, flush=True)
    # B_obs coefficients (interpretability), fit on development families
    st = Std().fit(np.array([S[i]["obs_feats"] for i in devfam]))
    rg = Ridge(alpha=10.0).fit(st(np.array([S[i]["obs_feats"] for i in devfam])), y[devfam])
    R["B_obs_coefficients_dev"] = dict(zip(OBS_NAMES, map(float, rg.coef_)))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=1)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
