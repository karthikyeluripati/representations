"""Stage 1-3 (CPU): false novelty / false equivalence of novelty keys in the WebState simulator.

python scripts/strong_paper/novelty_reliability.py --out results/strong_paper --seed 0

Steps
1. Sample random-walk histories per family (hidden variables drawn uniformly).
2. Build pairs:
   - A_render_<op>: same history, cosmetic render change (layout, decor, style, rename). Equivalent.
   - A_route / B: two histories whose *raw* current observations are identical; label by H-step trace
     equivalence (H=3) verified by execution. Equivalent -> A_route, not -> B.
   - Identifiability: a pair is *determined* if its equivalence label is the same for every hidden
     assignment consistent with both observed histories; otherwise *ambiguous* (no function of the
     observable history can be right for all completions; Bayes ceiling = max(p, 1-p)).
3. Evaluate deterministic novelty keys, string novelty, interventional probes (simulator snapshots), and a
   small learned combiner (diagnostic only) on held-out families (mail, ticket) and a held-out op (rename).
4. Exploration consequence: count-based exploration with each key; oracle states / transitions covered.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import random
import re
import sys
import time
from collections import defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import webstate as W  # noqa: E402

TRAIN_FAMS = ("settings", "shop")
TEST_FAMS = ("mail", "ticket")
H_EQ = 3

# ----------------------------------------------------------------------------------------------
# Items
# ----------------------------------------------------------------------------------------------


def hidden_space(fam):
    keys = sorted(fam.HIDDEN)
    return [dict(zip(keys, vals)) for vals in itertools.product(*(fam.HIDDEN[k] for k in keys))]


def make_item(fam, hidden, seq, cfg):
    obs, acts, st = W.render_history(fam, hidden, seq, cfg)
    return {"fam": fam.name, "hidden": hidden, "seq": list(seq), "cfg": cfg, "obs": obs, "acts": acts,
            "state": st}


def random_walk(fam, hidden, L, rng):
    s = fam.init(dict(hidden))
    seq = []
    for _ in range(L):
        acts = W.actions(fam, s)
        if not acts:
            break
        a = rng.choice(acts)
        seq.append(a)
        s = W.step(fam, s, a)
    return seq


def sample_items(fam, n, rng):
    hs = hidden_space(fam)
    seen, items = set(), []
    for _ in range(n * 3):
        hid = rng.choice(hs)
        seq = random_walk(fam, hid, rng.randint(0, 6), rng)
        k = W.h([hid, seq])
        if k in seen:
            continue
        seen.add(k)
        items.append(make_item(fam, hid, seq, W.RenderCfg()))
        if len(items) >= n:
            break
    return items


# ----------------------------------------------------------------------------------------------
# Labels and identifiability
# ----------------------------------------------------------------------------------------------


def min_div_horizon(fam, s1, s2, hmax=6):
    for H in range(0, hmax + 1):
        ok, _ = W.h_equivalent(fam, s1, s2, H)
        if not ok:
            return H
    return None


def consistent_states(fam, item):
    """Final states of all hidden assignments that reproduce the item's observable history."""
    out = []
    for hid in hidden_space(fam):
        try:
            obs, acts, st = W.render_history(fam, hid, item["seq"], item["cfg"])
        except ValueError:
            continue
        if obs == item["obs"] and acts == item["acts"]:
            out.append(st)
    return out


def identifiability(fam, i1, i2):
    c1, c2 = consistent_states(fam, i1), consistent_states(fam, i2)
    vals = [W.h_equivalent(fam, a, b, H_EQ)[0] for a in c1 for b in c2]
    p = float(np.mean(vals))
    return ("determined" if p in (0.0, 1.0) else "ambiguous"), p


# ----------------------------------------------------------------------------------------------
# Canonicalizers / keys (deployment side: text only)
# ----------------------------------------------------------------------------------------------
STYLE_RE = re.compile(r"\s*\{class=[^}]*\}")
LINE_RE = re.compile(r"^\[(\w+)\] ([^:]*?)(?:: (.*))?$")


def canon_line(ln):
    return STYLE_RE.sub("", ln).strip().lower()


def canon_obs(text):
    lines = [canon_line(l) for l in text.split("\n") if not l.startswith("[static]")]
    return lines[0] + "|" + "|".join(sorted(lines[1:]))


def canon_role(text):
    out = []
    for l in text.split("\n")[1:]:
        if l.startswith("[static]"):
            continue
        l = canon_line(l)
        m = LINE_RE.match(l)
        if not m or m.group(1) == "status":
            out.append(l)
        else:
            out.append(f"[{m.group(1)}]" + (f":{m.group(3)}" if m.group(3) is not None else ""))
    return "|".join(sorted(out))


def tracker_map(obs_list):
    m = {}
    for text in obs_list:
        for l in text.split("\n")[1:]:
            if l.startswith("[static]"):
                continue
            l = canon_line(l)
            mm = LINE_RE.match(l)
            if not mm:
                continue
            if mm.group(1) == "status":
                m["__status"] = mm.group(2)
            elif mm.group(3) is not None:
                m[mm.group(2)] = mm.group(3)
    return frozenset(m.items())


def full_hist(item):
    seq = []
    for i, o in enumerate(item["obs"]):
        seq.append(canon_obs(o))
        if i < len(item["acts"]):
            seq.append(item["acts"][i].lower())
    return tuple(seq)


KEYS = {
    "obs_exact": lambda it: it["obs"][-1],
    "obs_canon": lambda it: canon_obs(it["obs"][-1]),
    "obs_canon_role": lambda it: canon_role(it["obs"][-1]),
    "transition_canon": lambda it: (canon_obs(it["obs"][-2]) if len(it["obs"]) > 1 else "",
                                    it["acts"][-1].lower() if it["acts"] else "", canon_obs(it["obs"][-1])),
    "hist_set": lambda it: (canon_obs(it["obs"][-1]), frozenset(canon_obs(o) for o in it["obs"])),
    "tracker": lambda it: (canon_obs(it["obs"][-1]), tracker_map(it["obs"])),
    "hist_full": full_hist,
    # stronger deterministic heuristics: last-seen *field* values only (no transient status messages), and
    # a log of in-page (non-navigating) actions as a proxy for state mutations
    "tracker_fields": lambda it: (canon_obs(it["obs"][-1]),
                                  frozenset(kv for kv in tracker_map(it["obs"]) if kv[0] != "__status")),
    "mutation_log": lambda it: (canon_obs(it["obs"][-1]), tuple(sorted(
        it["acts"][i].lower() for i in range(len(it["acts"]))
        if it["obs"][i].split("\n")[0] == it["obs"][i + 1].split("\n")[0]))),
}


def hist_text(it):
    return "\n".join(o + ("\n" + it["acts"][i] if i < len(it["acts"]) else "") for i, o in enumerate(it["obs"]))


# ----------------------------------------------------------------------------------------------
# Interventional probe (simulator snapshot; compares canonical observed outcomes only)
# ----------------------------------------------------------------------------------------------


def _labels(fam, s, cfg):
    return {W.label(fam, e, cfg).lower(): e for e in W.actions(fam, s)}


def probe_same(fam, s1, cfg1, s2, cfg2, k, nocommit, cost):
    if canon_obs(W.render(fam, W.sem_obs(fam, s1), cfg1)) != canon_obs(W.render(fam, W.sem_obs(fam, s2), cfg2)):
        return False
    if k == 0:
        return True
    l1, l2 = _labels(fam, s1, cfg1), _labels(fam, s2, cfg2)
    if set(l1) != set(l2):
        return False
    for lab in sorted(l1):
        if nocommit and l1[lab] in fam.COMMITTAL:
            continue
        cost[0] += 2
        if not probe_same(fam, W.step(fam, s1, l1[lab]), cfg1, W.step(fam, s2, l2[lab]), cfg2, k - 1, nocommit, cost):
            return False
    return True


# ----------------------------------------------------------------------------------------------
# Pair construction
# ----------------------------------------------------------------------------------------------


def build_pairs(rng, n_items, pairs_per_group, n_render):
    pairs = []
    for fname, fam in W.FAMILIES.items():
        items = sample_items(fam, n_items, rng)
        groups = defaultdict(list)
        for it in items:
            groups[it["obs"][-1]].append(it)
        for gk, g in groups.items():
            cand = [(a, b) for a, b in itertools.combinations(range(len(g)), 2)]
            rng.shuffle(cand)
            for a, b in cand[:pairs_per_group]:
                i1, i2 = g[a], g[b]
                eq, wit = W.h_equivalent(fam, i1["state"], i2["state"], H_EQ)
                same_hist = i1["obs"] == i2["obs"] and i1["acts"] == i2["acts"]
                if eq and same_hist:
                    continue  # trivially identical inputs and behaviour
                ident, p = identifiability(fam, i1, i2)
                pairs.append({
                    "fam": fname, "kind": "A_route" if eq else "B", "equiv": eq, "i1": i1, "i2": i2,
                    "cluster": f"{fname}:{W.h(gk)}", "ident": ident, "p_equiv_consistent": p,
                    "same_hist": same_hist, "witness": wit,
                    "eq_h1": W.h_equivalent(fam, i1["state"], i2["state"], 1)[0],
                    "min_div_h": None if eq else min_div_horizon(fam, i1["state"], i2["state"]),
                    "nocommit_probe_h3_resolves": (not eq) and not W.h_equivalent(fam, i1["state"], i2["state"], 3, nocommit=True)[0],
                })
        decor = W.DECOR_TRAIN if fname in TRAIN_FAMS else W.DECOR_TEST
        for it in rng.sample(items, min(n_render, len(items))):
            ops = {
                "layout": W.RenderCfg(layout_seed=rng.randint(1, 10**6)),
                "decor": W.RenderCfg(decor=rng.sample(decor, 2)),
                "style": W.RenderCfg(style=True),
                "rename": W.RenderCfg(label_var=rng.choice([1, 2])),
            }
            for op, cfg in ops.items():
                it2 = make_item(fam, it["hidden"], it["seq"], cfg)
                assert it2["state"] == it["state"]
                pairs.append({"fam": fname, "kind": f"A_render_{op}", "equiv": True, "i1": it, "i2": it2,
                              "cluster": f"{fname}:{W.h([it['hidden'], it['seq']])}", "ident": "determined",
                              "p_equiv_consistent": 1.0, "same_hist": False, "witness": None,
                              "eq_h1": True, "min_div_h": None, "nocommit_probe_h3_resolves": False})
    return pairs


def split_of(p, rng_split):
    if p["fam"] in TEST_FAMS:
        return "test"
    return "dev" if rng_split[p["cluster"]] < 0.3 else "train"


# ----------------------------------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------------------------------


def subtype(p):
    if p["kind"] == "B":
        return "B_" + p["ident"]
    if p["kind"] == "A_route":
        return "A_route_" + p["ident"]
    return p["kind"]


def boot_rate(vals, clusters, rng, B=1000):
    vals, clusters = np.asarray(vals, float), np.asarray(clusters)
    uc, inv = np.unique(clusters, return_inverse=True)
    sums = np.bincount(inv, weights=vals, minlength=len(uc))
    cnts = np.bincount(inv, minlength=len(uc)).astype(float)
    est = vals.mean()
    W_ = rng.multinomial(len(uc), np.full(len(uc), 1 / len(uc)), size=B)  # cluster bootstrap weights
    bs = (W_ @ sums) / (W_ @ cnts)
    return [round(float(est), 4), round(float(np.percentile(bs, 2.5)), 4), round(float(np.percentile(bs, 97.5)), 4)]


def pair_features(p, vec_obs, vec_hist):
    i1, i2 = p["i1"], p["i2"]
    s1 = set(canon_obs(o) for o in i1["obs"])
    s2 = set(canon_obs(o) for o in i2["obs"])
    st1 = [l for o in i1["obs"] for l in o.split("\n") if l.startswith("[status]")]
    st2 = [l for o in i2["obs"] for l in o.split("\n") if l.startswith("[status]")]
    return [
        float(KEYS["obs_canon"](i1) == KEYS["obs_canon"](i2)),
        float(KEYS["obs_canon_role"](i1) == KEYS["obs_canon_role"](i2)),
        float(KEYS["hist_set"](i1) == KEYS["hist_set"](i2)),
        float(KEYS["tracker"](i1) == KEYS["tracker"](i2)),
        float(KEYS["hist_full"](i1) == KEYS["hist_full"](i2)),
        len(s1 & s2) / max(1, len(s1 | s2)),
        abs(len(i1["obs"]) - len(i2["obs"])),
        float(sorted(st1) == sorted(st2)),
        float(KEYS["transition_canon"](i1) == KEYS["transition_canon"](i2)),
        vec_obs(p), vec_hist(p),
    ]


def run_eval(pairs, rng):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import roc_auc_score
    from sklearn.tree import DecisionTreeClassifier

    rsplit = defaultdict(lambda: random.random())
    rnd = random.Random(1)
    clusters = sorted({p["cluster"] for p in pairs})
    for c in clusters:
        rsplit[c] = rnd.random()
    for p in pairs:
        p["split"] = split_of(p, rsplit)
        p["subtype"] = subtype(p)

    fit_txt = [p["i1"]["obs"][-1] for p in pairs if p["split"] == "train"]
    fit_hist = [hist_text(p["i1"]) for p in pairs if p["split"] == "train"]
    tv_o = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(fit_txt)
    tv_h = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5)).fit(fit_hist)

    def cosd(tv, a, b):
        X = tv.transform([a, b])
        return float(1 - (X[0].multiply(X[1])).sum())

    for p in pairs:
        p["d_tfidf_obs"] = cosd(tv_o, p["i1"]["obs"][-1], p["i2"]["obs"][-1])
        p["d_tfidf_hist"] = cosd(tv_h, hist_text(p["i1"]), hist_text(p["i2"]))

    # decisions: True = judged "same / not novel"
    dec = {m: [] for m in list(KEYS) + ["tfidf_obs", "tfidf_hist", "probe_k1", "probe_k2", "probe_k2_nocommit",
                                        "learned_lr", "learned_tree", "oracle"]}
    cost = {m: [] for m in ("probe_k1", "probe_k2", "probe_k2_nocommit")}

    def tune(key):
        dv = [p for p in pairs if p["split"] == "dev" and "rename" not in p["kind"]]
        best = None
        for tau in sorted({round(p[key], 4) for p in dv} | {0.0}):
            same = np.array([p[key] <= tau for p in dv])
            y = np.array([p["equiv"] for p in dv])
            bacc = 0.5 * ((same[y]).mean() + (~same[~y]).mean())
            if best is None or bacc > best[0]:
                best = (bacc, tau)
        return best[1]

    taus = {"tfidf_obs": tune("d_tfidf_obs"), "tfidf_hist": tune("d_tfidf_hist")}

    for p in pairs:
        fam = W.FAMILIES[p["fam"]]
        for m, f in KEYS.items():
            dec[m].append(f(p["i1"]) == f(p["i2"]))
        dec["tfidf_obs"].append(p["d_tfidf_obs"] <= taus["tfidf_obs"])
        dec["tfidf_hist"].append(p["d_tfidf_hist"] <= taus["tfidf_hist"])
        for m, k, nc in (("probe_k1", 1, False), ("probe_k2", 2, False), ("probe_k2_nocommit", 2, True)):
            c = [0]
            dec[m].append(probe_same(fam, p["i1"]["state"], p["i1"]["cfg"], p["i2"]["state"], p["i2"]["cfg"], k, nc, c))
            cost[m].append(c[0])
        dec["oracle"].append(p["equiv"])

    # learned combiner (diagnostic): trained on train split, non-rename ops
    F = np.array([pair_features(p, lambda q: q["d_tfidf_obs"], lambda q: q["d_tfidf_hist"]) for p in pairs])
    y = np.array([p["equiv"] for p in pairs])
    tr = np.array([p["split"] == "train" and "rename" not in p["kind"] for p in pairs])
    mu, sd = F[tr].mean(0), F[tr].std(0) + 1e-9
    lr = LogisticRegression(max_iter=2000, class_weight="balanced").fit((F[tr] - mu) / sd, y[tr])
    tree = DecisionTreeClassifier(max_depth=4, class_weight="balanced", random_state=0).fit(F[tr], y[tr])
    prob_lr = lr.predict_proba((F - mu) / sd)[:, 1]
    dec["learned_lr"] = list(prob_lr >= 0.5)
    dec["learned_tree"] = list(tree.predict(F).astype(bool))

    # metrics
    out = {"taus": taus, "n_pairs": len(pairs), "by_split": {}}
    for split in ("train", "dev", "test"):
        res = {}
        idx_split = [i for i, p in enumerate(pairs) if p["split"] == split]
        sts = sorted({pairs[i]["subtype"] for i in idx_split})
        for st in sts:
            ii = [i for i in idx_split if pairs[i]["subtype"] == st]
            cl = [pairs[i]["cluster"] for i in ii]
            equiv = pairs[ii[0]]["equiv"]
            row = {"n": len(ii), "n_clusters": len(set(cl)), "equiv": equiv,
                   "metric": "false_novelty_rate" if equiv else "false_equivalence_rate"}
            for m in dec:
                err = [(not dec[m][i]) if equiv else dec[m][i] for i in ii]
                row[m] = boot_rate(err, cl, np.random.default_rng(0), B=500)
            if not equiv:
                row["nocommit_probe_h3_resolves"] = float(np.mean([pairs[i]["nocommit_probe_h3_resolves"] for i in ii]))
                row["min_div_h_hist"] = {str(k): int(v) for k, v in zip(*np.unique([pairs[i]["min_div_h"] for i in ii], return_counts=True))}
                row["eq_at_h1_but_not_h3"] = float(np.mean([pairs[i]["eq_h1"] for i in ii]))
            row["bayes_ceiling_error"] = round(float(np.mean([min(pairs[i]["p_equiv_consistent"], 1 - pairs[i]["p_equiv_consistent"]) for i in ii])), 4)
            res[st] = row
        # summary: mean FN over A subtypes (equal weight) and FE over B subtypes
        res["_auroc_same"] = {}
        ys = y[idx_split]
        if len(set(ys)) == 2:
            for name, score in (("tfidf_obs", -np.array([p["d_tfidf_obs"] for p in pairs])),
                                ("tfidf_hist", -np.array([p["d_tfidf_hist"] for p in pairs])),
                                ("learned_lr", prob_lr)):
                res["_auroc_same"][name] = round(float(roc_auc_score(ys, score[idx_split])), 4)
        out["by_split"][split] = res
    out["probe_cost_mean_env_steps"] = {m: float(np.mean(v)) for m, v in cost.items()}
    out["lr_coef"] = dict(zip(["obs_canon", "obs_canon_role", "hist_set", "tracker", "hist_full", "hist_jaccard",
                               "len_diff", "status_equal", "transition_canon", "d_tfidf_obs", "d_tfidf_hist"],
                              [round(float(c), 3) for c in lr.coef_[0]]))
    return out


# ----------------------------------------------------------------------------------------------
# Exploration consequence
# ----------------------------------------------------------------------------------------------

SUCCESS = {
    "settings": lambda s: s["notif_saved"],
    "shop": lambda s: s["page"] == "done",
    "mail": lambda s: s["sent"] > 0,
    "ticket": lambda s: s["req"] == "approved",
}


def explore(fam, method, seed, episodes, T, cosmetic):
    rng = random.Random(seed)
    hs = hidden_space(fam)
    counts = defaultdict(int)
    states, trans = set(), set()
    first_success = None
    step_i = 0
    for ep in range(episodes):
        hid = rng.choice(hs)
        cfg = W.RenderCfg(layout_seed=rng.randint(1, 10**6), decor=rng.sample(W.DECOR_TRAIN, rng.randint(0, 2)),
                          style=rng.random() < 0.5, label_var=rng.randint(0, 2)) if cosmetic else W.RenderCfg()
        s = fam.init(dict(hid))
        it = {"obs": [W.render(fam, W.sem_obs(fam, s), cfg)], "acts": []}
        states.add(W.state_key(s))
        for _ in range(T):
            acts = W.actions(fam, s)
            if not acts:
                break
            if method == "random":
                a = rng.choice(acts)
            else:
                key = W.state_key(s) if method == "oracle_state" else KEYS[method](it)
                akey = (lambda e: e) if method == "oracle_state" else (lambda e: W.label(fam, e, cfg).lower())
                cs = [counts[(key, akey(e))] for e in acts]
                mn = min(cs)
                a = rng.choice([e for e, c in zip(acts, cs) if c == mn])
                counts[(key, akey(a))] += 1
            trans.add((W.state_key(s), a))
            s = W.step(fam, s, a)
            step_i += 1
            it["obs"].append(W.render(fam, W.sem_obs(fam, s), cfg))
            it["acts"].append(f"click({W.label(fam, a, cfg)})")
            states.add(W.state_key(s))
            if first_success is None and SUCCESS[fam.name](s):
                first_success = step_i
    return {"states": len(states), "transitions": len(trans), "first_success": first_success}


def run_explore(seeds, episodes, T):
    methods = ["random", "obs_exact", "obs_canon", "obs_canon_role", "transition_canon", "hist_set", "tracker",
               "hist_full", "oracle_state"]
    out = {}
    for cosmetic in (False, True):
        tag = "cosmetic_perturbations" if cosmetic else "clean_render"
        out[tag] = {}
        for fname, fam in W.FAMILIES.items():
            out[tag][fname] = {}
            for m in methods:
                rs = [explore(fam, m, s, episodes, T, cosmetic) for s in range(seeds)]
                st = np.array([r["states"] for r in rs], float)
                tr = np.array([r["transitions"] for r in rs], float)
                fs = [r["first_success"] for r in rs]
                out[tag][fname][m] = {
                    "states_mean": round(st.mean(), 2), "states_se": round(st.std(ddof=1) / np.sqrt(len(st)), 2),
                    "transitions_mean": round(tr.mean(), 2), "transitions_se": round(tr.std(ddof=1) / np.sqrt(len(tr)), 2),
                    "success_found_frac": float(np.mean([f is not None for f in fs])),
                    "median_steps_to_success": (float(np.median([f for f in fs if f is not None])) if any(f is not None for f in fs) else None),
                }
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="results/strong_paper")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-items", type=int, default=400)
    ap.add_argument("--pairs-per-group", type=int, default=25)
    ap.add_argument("--n-render", type=int, default=120)
    ap.add_argument("--explore-seeds", type=int, default=20)
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--T", type=int, default=8)
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    t0 = time.time()
    rng = random.Random(args.seed)
    pairs = build_pairs(rng, args.n_items, args.pairs_per_group, args.n_render)
    t1 = time.time()
    rel = run_eval(pairs, np.random.default_rng(args.seed))
    t2 = time.time()
    counts = defaultdict(lambda: defaultdict(int))
    for p in pairs:
        counts[p["split"]][p["subtype"]] += 1
    rel["counts"] = {k: dict(v) for k, v in counts.items()}
    rel["config"] = vars(args) | {"H_EQ": H_EQ, "train_fams": TRAIN_FAMS, "test_fams": TEST_FAMS,
                                  "held_out_op": "rename (never used for tuning/training)"}
    ex = run_explore(args.explore_seeds, args.episodes, args.T)
    t3 = time.time()
    rel["timing_s"] = {"build": round(t1 - t0, 1), "eval": round(t2 - t1, 1), "explore": round(t3 - t2, 1)}
    with open(os.path.join(args.out, "reliability.json"), "w") as f:
        json.dump(rel, f, indent=1, default=str)
    with open(os.path.join(args.out, "exploration.json"), "w") as f:
        json.dump(ex, f, indent=1)
    # compact pair index (reproducible from seed; histories as semantic action ids)
    with open(os.path.join(args.out, "pairs_index.jsonl"), "w") as f:
        for p in pairs:
            f.write(json.dumps({"fam": p["fam"], "kind": p["kind"], "subtype": p["subtype"], "split": p["split"],
                                "equiv": p["equiv"], "ident": p["ident"], "p_equiv_consistent": p["p_equiv_consistent"],
                                "witness": p["witness"], "min_div_h": p["min_div_h"],
                                "a": [p["i1"]["hidden"], p["i1"]["seq"], p["i1"]["cfg"].as_dict()],
                                "b": [p["i2"]["hidden"], p["i2"]["seq"], p["i2"]["cfg"].as_dict()]}, default=str) + "\n")
    print(json.dumps(rel["counts"], indent=1))
    print("timing", rel["timing_s"])


if __name__ == "__main__":
    main()
