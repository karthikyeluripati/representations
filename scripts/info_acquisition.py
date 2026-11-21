"""Information-acquisition diagnostics (offline, CPU). Stages 1–3 of docs/information_acquisition_novelty.md.

Stage 1: action taxonomy on source trajectories (observable rules only, never episode success):
  D redundant/invalid : next observation is "Nothing happens." or identical to an earlier observation in
                        the episode (checked first)
  A information       : examine, look, inventory (no task-relevant world change)
  C mixed             : go to, open (changes location/receptacle state and reveals contents)
  B execution         : take, move (= put), heat, cool, clean, slice, use, toggle
  E ambiguous         : close, help, anything else
  new_relevant_info   : the next observation mentions an objective keyword not seen in any earlier
                        observation of the episode
  entropy             : H(softmax(mean log-prob)) over ALL admissible actions (logged at every step).
                        This is entropy of the LM's action-score distribution, not of the greedy policy itself.
Stage 2: observational comparison of information vs other actions, exactly matched on
         (model, family, 10-step bin, pre-entropy tercile, progress tercile); task-cluster bootstrap.
Stage 3: matched-STATE counterfactual test on branch data: does pre-action entropy predict when
         information-type candidates are worth more (higher branch success) than the alternatives?
         Pairwise-ranking test: does entropy × action-type add to logp + action-type?

Usage: python scripts/info_acquisition.py --out results/phase4/info_acq.json
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
from scipy.stats import spearmanr  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402

from dare.evaluation.metrics import auroc, pairwise_ranking_accuracy  # noqa: E402
from dare.utils.logging import load_run, read_jsonl  # noqa: E402

INFO, MIXED = {"examine", "look", "inventory"}, {"go", "open"}
EXEC = {"take", "move", "put", "heat", "cool", "clean", "slice", "use", "toggle"}
STOP = {"the", "a", "an", "in", "on", "with", "and", "put", "some", "it", "examine", "look", "at", "under",
        "two", "clean", "hot", "cool", "find", "then", "place", "of", "to", "into"}
MODELS = {"3b": "results/decisions_v1_3b", "7b": "results/decisions_v1_7b"}


def ent(lp):
    lp = np.asarray(lp, float)
    p = np.exp(lp - lp.max()); p /= p.sum()
    return float(-(p * np.log(p + 1e-12)).sum()), float(np.log(len(lp)))


def action_type(a):
    v = a.split()[0]
    return "A" if v in INFO else "C" if v in MIXED else "B" if v in EXEC else "E"


def keywords(obj):
    return [w for w in re.findall(r"[a-z]+", obj.lower()) if w not in STOP]


def transitions(D, model):
    eps, trs = [], []
    for sd in sorted(glob.glob(os.path.join(D, "source*"))):
        e, t = load_run(sd)
        eps += e; trs += t
    by = collections.defaultdict(list)
    for t in trs:
        by[t["episode_id"]].append(t)
    out = []
    for e in eps:
        steps = sorted(by[e["episode_id"]], key=lambda t: t["step"])
        # the intro contains "Your task is to: <objective>"; drop it, or every keyword counts as already seen
        intro_env = steps[0]["observation"].split("Your task is to:")[0]
        seen_obs, seen_text = [steps[0]["observation"]], intro_env.lower()
        kw = keywords(e["objective"])
        n_steps = e["n_steps"]
        for i, t in enumerate(steps):
            nxt = t["next_observation"]
            H, Hmax = ent(t["lm_logp"])
            Ha = ent(steps[i + 1]["lm_logp"]) if i + 1 < len(steps) else (np.nan, np.nan)
            redundant = nxt == "Nothing happens." or nxt in seen_obs
            typ = "D" if redundant else action_type(t["selected_action"])
            new_kw = [w for w in kw if w in nxt.lower() and w not in seen_text]
            out.append({
                "model": model, "task": t["task_id"], "family": t["task_family"], "step": t["step"],
                "episode_success": bool(e["success"]), "n_steps": n_steps,
                "remaining_to_success": (n_steps - t["step"]) if e["success"] else None,
                "type": typ, "raw_type": action_type(t["selected_action"]), "verb": t["selected_action"].split()[0],
                "H": H, "Hn": H / Hmax if Hmax > 0 else 0.0, "H_after": Ha[0],
                "Hn_after": Ha[0] / Ha[1] if Ha[1] and Ha[1] > 0 else np.nan,
                "new_relevant_info": len(new_kw) > 0,
                "progress": float(np.mean([w in seen_text for w in kw])) if kw else 0.0,
                "repeat_action": t["selected_action"] in [s["selected_action"] for s in steps[:i]],
                "tokens": t["tokens"]["total"], "tokens_computed": t["tokens"]["prompt_computed"] + t["tokens"]["candidates"],
                "env_s": t["time_s"]["env"], "lm_s": t["time_s"]["lm"],
                "action": t["selected_action"], "next_obs": nxt[:120],
                "split": None,
            })
            seen_obs.append(nxt); seen_text += " " + nxt.lower()
    return out


def boot_ci(fn, clusters, n=2000, seed=0):
    groups = collections.defaultdict(list)
    for i, c in enumerate(clusters):
        groups[c].append(i)
    keys = list(groups); rng = np.random.default_rng(seed); vals = []
    for _ in range(n):
        ii = np.concatenate([groups[keys[j]] for j in rng.integers(0, len(keys), len(keys))])
        v = fn(ii)
        if v is not None and not np.isnan(v):
            vals.append(v)
    return [float(fn(np.arange(len(clusters)))), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))]


def matched_diff(rows, treat, outcome, strata_keys):
    """Exact stratification: mean over strata (weighted by #treated) of mean(outcome|treated) - mean(|control)."""
    def fn(ii):
        S = collections.defaultdict(lambda: [[], []])
        for i in ii:
            r = rows[i]
            v = outcome(r)
            if v is None or (isinstance(v, float) and np.isnan(v)):
                continue
            S[tuple(r[k] for k in strata_keys)][0 if treat(r) else 1].append(v)
        num = den = 0.0
        for a, b in S.values():
            if a and b:
                num += len(a) * (np.mean(a) - np.mean(b)); den += len(a)
        return num / den if den else np.nan
    return fn


def stage2(rows):
    for r in rows:
        r["step_bin"] = min(r["step"] // 10, 4)
    R = {}
    for m in MODELS:
        rr = [r for r in rows if r["model"] == m and not np.isnan(r["Hn_after"])]
        Hs = np.array([r["Hn"] for r in rr]); P = np.array([r["progress"] for r in rr])
        hq, pq = np.quantile(Hs, [1 / 3, 2 / 3]), np.quantile(P, [1 / 3, 2 / 3])
        for r in rr:
            r["H_bin"] = int(np.searchsorted(hq, r["Hn"])); r["prog_bin"] = int(np.searchsorted(pq, r["progress"]))
        cl = [r["task"] for r in rr]
        keys = ("family", "step_bin", "H_bin", "prog_bin")
        res = {"n_transitions": len(rr), "type_counts": dict(collections.Counter(r["type"] for r in rr))}
        dH = lambda r: r["Hn_after"] - r["Hn"]  # noqa: E731
        for name, treat, ctrl in (
                ("info(A)_vs_exec(B)", lambda r: r["type"] == "A", lambda r: r["type"] == "B"),
                ("info_or_mixed(A+C)_vs_exec(B)", lambda r: r["type"] in "AC", lambda r: r["type"] == "B"),
                ("nonredundant_info(A+C)_vs_redundant(D)", lambda r: r["type"] in "AC", lambda r: r["type"] == "D")):
            sub = [r for r in rr if treat(r) or ctrl(r)]
            c = [r["task"] for r in sub]
            res[name] = {
                "n_treated": sum(treat(r) for r in sub), "n_control": sum(ctrl(r) for r in sub),
                "d_entropy_change": boot_ci(matched_diff(sub, treat, dH, keys), c),
                "d_new_relevant_info": boot_ci(matched_diff(sub, treat, lambda r: float(r["new_relevant_info"]), keys), c),
                "d_episode_success(observational)": boot_ci(matched_diff(sub, treat, lambda r: float(r["episode_success"]), keys), c),
            }
        # does entropy reduction identify RELEVANT information among information-type actions?
        info = [r for r in rr if r["raw_type"] in "AC"]
        y = np.array([r["new_relevant_info"] for r in info]); c = [r["task"] for r in info]
        feats = {"neg_entropy_change": np.array([-(dH(r)) for r in info]),
                 "pre_entropy": np.array([r["Hn"] for r in info]),
                 "neg_step": np.array([-r["step"] for r in info]),
                 "not_repeat_action": np.array([float(not r["repeat_action"]) for r in info])}
        res["AUROC_predict_new_relevant_info_among_info_actions"] = {
            k: boot_ci(lambda ii, f=f: auroc(f[ii], y[ii]), c) for k, f in feats.items()}
        res["base_rate_new_relevant_info_among_info_actions"] = float(y.mean())
        res["spearman_entropy_change_vs_new_info"] = float(spearmanr(feats["neg_entropy_change"], y).statistic)
        R[m] = res
    return R


def stage3(rows):
    """Branch data: 4 candidates per state with 8-rollout success. Matched-state test."""
    R = {}
    tr_by = collections.defaultdict(dict)
    for r in rows:
        tr_by[(r["model"], r["task"])][r["step"]] = r
    for m, D in MODELS.items():
        dec = [x for f in sorted(glob.glob(os.path.join(D, "decisions*.jsonl"))) for x in read_jsonl(f)]
        src = {}
        for sd in sorted(glob.glob(os.path.join(D, "source*"))):
            e, t = load_run(sd)
            for x in t:
                src[(x["episode_id"], x["step"])] = x
        by = collections.defaultdict(list)
        for x in dec:
            by[x["state_id"]].append(x)
        S = []
        for sid, xs in by.items():
            x0 = xs[0]; ep, t = x0["history_ref"]["episode_id"], x0["step"]
            cur = src[(ep, t)]
            prev = [src[(ep, k)]["selected_action"] for k in range(t)]
            H, Hmax = ent(cur["lm_logp"])
            lp = dict(zip(cur["candidates"], cur["lm_logp"]))
            for x in xs:
                a = x["candidate_action"]; ty = action_type(a)
                S.append({"state": sid, "task": x["task_id"], "family": x["task_family"], "split": x["fit_split"],
                          "step": t, "Hn": H / Hmax, "logp": lp[a], "type": ty,
                          "info": ty in "AC", "repeat": a in prev, "y": x["branch_success_rate"]})
        st = collections.defaultdict(list)
        for i, s in enumerate(S):
            st[s["state"]].append(i)
        # (a) per-state value of information: mean y(info candidates) - mean y(non-info), states with both
        adv, H, step, task, fam, spl = [], [], [], [], [], []
        for sid, ii in st.items():
            a = [S[i]["y"] for i in ii if S[i]["info"] and not S[i]["repeat"]]
            b = [S[i]["y"] for i in ii if not S[i]["info"]]
            if a and b and len({S[i]["y"] for i in ii}) > 1:
                adv.append(np.mean(a) - np.mean(b)); H.append(S[ii[0]]["Hn"]); step.append(S[ii[0]]["step"])
                task.append(S[ii[0]]["task"]); fam.append(S[ii[0]]["family"]); spl.append(S[ii[0]]["split"])
        adv, H, step = map(np.array, (adv, H, step))
        res = {"n_rows": len(S), "candidate_type_counts": dict(collections.Counter(s["type"] + ("r" if s["repeat"] else "") for s in S)),
               "n_informative_states_with_info_and_noninfo": int(len(adv)),
               "n_tasks": len(set(task)),
               "mean_VOI_proxy(info - noninfo success)": boot_ci(lambda ii: float(np.mean(adv[ii])), task),
               "spearman(pre_entropy, VOI_proxy)": boot_ci(lambda ii: float(spearmanr(H[ii], adv[ii]).statistic), task),
               "spearman(step, VOI_proxy)": boot_ci(lambda ii: float(spearmanr(step[ii], adv[ii]).statistic), task)}
        # partial: entropy vs VOI controlling step (rank residuals)
        from scipy.stats import rankdata

        def partial(ii):
            a, b, c = rankdata(H[ii]), rankdata(adv[ii]), rankdata(step[ii])
            X = np.c_[np.ones_like(c), c]
            ra = a - X @ np.linalg.lstsq(X, a, rcond=None)[0]; rb = b - X @ np.linalg.lstsq(X, b, rcond=None)[0]
            return float(np.corrcoef(ra, rb)[0, 1])
        res["partial_spearman(entropy, VOI | step)"] = boot_ci(partial, task)
        # (b) pairwise ranking: logp; logp+type; logp+type+entropy×type. Fit on fit split, test on dev+holdout.
        y = np.array([s["y"] for s in S]); g = np.array([S_i for S_i in [s["state"] for s in S]])
        X0 = np.array([[s["logp"]] for s in S])
        X1 = np.array([[s["logp"], s["info"], s["repeat"], s["type"] == "B"] for s in S], float)
        X2 = np.c_[X1, X1[:, 1] * np.array([s["Hn"] for s in S]), X1[:, 2] * np.array([s["Hn"] for s in S]),
                   X1[:, 3] * np.array([s["Hn"] for s in S])]
        spl_r = np.array([s["split"] for s in S]); task_r = np.array([s["task"] for s in S])

        def fitpair(X, idx):
            P = [(i, j) for sid, ii in st.items() for i in ii for j in ii if i in idx and y[i] > y[j]]
            hi, lo = np.array([p[0] for p in P]), np.array([p[1] for p in P])
            w = np.abs(y[hi] - y[lo])
            Z = np.vstack([X[hi] - X[lo], X[lo] - X[hi]])
            mu, sd = X[list(idx)].mean(0), X[list(idx)].std(0) + 1e-9
            Zs = Z / sd
            clf = LogisticRegression(C=1.0, fit_intercept=False, max_iter=5000).fit(
                Zs, np.r_[np.ones(len(hi)), np.zeros(len(hi))], sample_weight=np.r_[w, w])
            return lambda XX: (XX / sd) @ clf.coef_[0]
        fit_idx = set(np.where(spl_r == "fit")[0])
        out = {}
        for nm, X in (("logp_only", X0), ("logp+type", X1), ("logp+type+entropy_x_type", X2)):
            f = fitpair(X, fit_idx); s = f(X)
            for sp in ("dev", "family_holdout"):
                ii = np.where(spl_r == sp)[0]
                out[f"{nm}:{sp}"] = boot_ci(lambda jj, ii=ii, s=s: pairwise_ranking_accuracy(s[ii][jj], y[ii][jj], g[ii][jj]),
                                            task_r[ii])
        # paired: entropy interaction vs logp+type
        f1, f2 = fitpair(X1, fit_idx), fitpair(X2, fit_idx)
        s1, s2 = f1(X1), f2(X2)
        for sp in ("dev", "family_holdout"):
            ii = np.where(spl_r == sp)[0]
            out[f"paired_delta(entropy_x_type - logp+type):{sp}"] = boot_ci(
                lambda jj, ii=ii: pairwise_ranking_accuracy(s2[ii][jj], y[ii][jj], g[ii][jj])
                - pairwise_ranking_accuracy(s1[ii][jj], y[ii][jj], g[ii][jj]), task_r[ii])
        res["pairwise_ranking"] = out
        R[m] = res
    return R


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--audit-sample", type=int, default=40)
    a = ap.parse_args()
    rows = []
    for m, D in MODELS.items():
        rows += transitions(D, m)
    R = {"stage1": {}}
    for m in MODELS:
        rr = [r for r in rows if r["model"] == m]
        R["stage1"][m] = {
            "n_transitions": len(rr),
            "type_counts": dict(collections.Counter(r["type"] for r in rr)),
            "raw_type_counts": dict(collections.Counter(r["raw_type"] for r in rr)),
            "redundant_share_of_info_actions": float(np.mean([r["type"] == "D" for r in rr if r["raw_type"] == "A"])),
            "redundant_share_of_exec_actions": float(np.mean([r["type"] == "D" for r in rr if r["raw_type"] == "B"])),
            "new_relevant_info_rate_by_type": {t: float(np.mean([r["new_relevant_info"] for r in rr if r["type"] == t]))
                                               for t in "ABCDE" if any(r["type"] == t for r in rr)},
            "mean_tokens_per_step": float(np.mean([r["tokens"] for r in rr])),
            "mean_computed_tokens_per_step": float(np.mean([r["tokens_computed"] for r in rr])),
            "mean_env_s": float(np.mean([r["env_s"] for r in rr])), "mean_lm_s": float(np.mean([r["lm_s"] for r in rr])),
        }
    rng = np.random.default_rng(0)
    R["stage1"]["manual_audit_sample"] = [{k: rows[i][k] for k in ("model", "action", "next_obs", "type", "new_relevant_info")}
                                          for i in rng.choice(len(rows), a.audit_sample, replace=False)]
    R["stage2"] = stage2(rows)
    R["stage3"] = stage3(rows)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(R, open(a.out, "w"), indent=1, default=float)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
