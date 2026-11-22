"""Training sets for the matched counterfactual-SFT pilot (docs/cf_sft_pilot_protocol.md). Tokenizer-free, CPU only.

Three arms, identical size (1,349 decisions; + 84 B anchors at render = 1,433, same as CAW-v2) and identical base:
  base (1,029)  CAW-v2 data (data/caw2_v1) minus 320 records: for each of the 64 synthetic classes the records of ONE
                idempotent kind (set for even classes, remove for odd: 4 records each = 256) and 64 duplicate cycles of
                the BFCL part. Every class and every unique BFCL record is kept.
  CF    base + 320 decision points from oracle (contract-policy) trajectories on T2 TRAIN families; held-out semantic
        combinations excluded
  CTRL  base + 320 decision points from always-check (CAW rule) trajectories on the same train families, restricted to
        cells where that rule is execution-verified correct in BOTH outcome twins (matched budget; no unsafe labels;
        no counterfactual labels)
  ABL   CF with the retry-behaviour contract removed from every tool description (labels unchanged): tests whether CF
        gains come from reading the contract
Every record is a decision point of an executed trajectory whose final grade is correct (asserted). Records are deduped
and sampled with fixed per-role quotas (QUOTA), round-robin over (effect, verifier, compensation, family) inside a role.
For CF records the CAW-rule action at the same point is recorded, and whether continuing with the CAW rule fails.
Output: data/cf_sft_v1/{CF,CTRL,ABL}.jsonl + MANIFEST.json
"""
import collections
import copy
import hashlib
import json
import os
import random

import _common
from verification import semantic_cf_t2 as T
from verification.synthetic_apis import DOMAINS

OUT = os.path.join(_common.STUDY, "data", "cf_sft_v1")
K = 320
QUOTA = {"cf": {"ff_call": 20, "finish_after_success": 20, "retry_reject": 20, "retry_idem": 30, "retry_keyed": 30, "check_needed": 35,
                "post_check_finish": 25, "post_check_reissue": 25, "escalate": 45, "reconcile_retry": 25, "reconcile_reverse": 25,
                "reconcile_finish": 20},
         "ctrl": {"ff_call": 40, "finish_after_success": 40, "retry_reject": 40, "check_after_timeout": 80, "post_check_finish": 60,
                  "post_check_reissue": 60}}
assert all(sum(q.values()) == K for q in QUOTA.values())


def trajectory(policy, spec, strip=False, prefix=None, cont=None):
    """Run `policy` (or the `prefix` actions, then `cont`) on one episode; return decision points and the grade."""
    ep = T.T2Episode(spec, strip); pts = []; k = 0
    for _ in range(10):
        if ep.done:
            break
        msgs, tools = ep.messages(T.SYSTEM), ep.tools()
        a = prefix[k] if prefix is not None and k < len(prefix) else (cont or policy).act(msgs, tools)
        pts.append((copy.deepcopy(msgs), tools, a)); k += 1
        if "final_text" in a:
            ep.finish(a["final_text"]); break
        ep.call(a["name"], a.get("arguments", {}))
    return pts, ep.grade()


def canon(a):
    return json.dumps(a, sort_keys=True)


def act_class(a, w):
    if "final_text" in a:
        return "finish"
    n = a["name"]
    return "reissue" if n == w else "escalate" if n == T.ESC else "reverse" if n.startswith("reverse_") else "check"


def role(tag, s, msgs, a):
    w = T.FamilyT2(s["family"]).names[s["name"]]; c = act_class(a, w); lab = T.oracle(s)[1]; last = msgs[-1]
    if last["role"] == "user":
        return "ff_call"
    prev = next(m for m in reversed(msgs) if m["role"] == "assistant")["tool_calls"][0]["function"]["name"]
    if prev == w and not last["content"].startswith("Error"):            # after a successful write (receipt)
        return {"reverse": "reconcile_reverse", "finish": "reconcile_finish" if lab == "reconcile" else "finish_after_success"}.get(c, "other")
    if prev == w:                                                         # right after the write error
        if c == "check":
            return "check_needed" if tag == "cf" else "check_after_timeout"
        if c == "escalate":
            return "escalate"
        if c == "reissue":
            return {"reissue": "retry_reject" if s["error"] == "reject" else "retry_idem", "reissue_same_key": "retry_keyed",
                    "reconcile": "reconcile_retry"}.get(lab, "other")
        return "other"
    return {"finish": "post_check_finish", "reissue": "post_check_reissue"}.get(c, "other")   # after a read


def records(policy, sp, tag):
    pool, n_traj = {}, 0
    for s in sp:
        pts, g = trajectory(policy, s)
        assert g["correct"], (tag, s)
        n_traj += 1
        for step, (msgs, tools, a) in enumerate(pts):
            h = hashlib.sha256(canon([msgs, tools, a]).encode()).hexdigest()
            if h in pool:
                continue
            ro = role(tag, s, msgs, a); assert ro in QUOTA[tag] or ro == "other", (tag, ro, s, a)
            pool[h] = {"kind": f"{tag}_{ro}", "task_id": f"semcf_t2:{T.FamilyT2(s['family']).domain}", "tools": tools,
                       "msgs": msgs, "chosen": a, "state": f"{tag}|{h[:16]}", "spec": s, "role": ro, "step": step,
                       "strat": [s["effect"], s["verifier"], s["compensation"], s["family"]], "prefix": [p[2] for p in pts[:step]]}
    return pool, n_traj


def balanced(pool, quota, seed):
    rng = random.Random(seed); out = []
    for ro, q in quota.items():
        by = collections.defaultdict(list)
        for h in sorted(pool):
            if pool[h]["role"] == ro:
                by[tuple(pool[h]["strat"])].append(h)
        keys = sorted(by); rng.shuffle(keys)
        for kk in keys:
            rng.shuffle(by[kk])
        got = []
        while len(got) < q and any(by.values()):
            for kk in keys:
                if by[kk] and len(got) < q:
                    got.append(pool[by[kk].pop()])
        assert len(got) == q, (ro, len(got), q)
        out += got
    return out


def both_twins_correct(policy, sp):
    """Specs of the cells where `policy` is execution-verified correct in every outcome twin."""
    key = lambda s: tuple((k, s[k]) for k in T.FACTORS if k != "outcome")  # noqa: E731
    ok = collections.defaultdict(list)
    for s in sp:
        ok[key(s)].append(trajectory(policy, s)[1]["correct"])
    return [s for s in sp if all(ok[key(s)])]


def base_records():
    caw = [json.loads(l) for l in open(os.path.join(_common.STUDY, "data", "caw2_v1", "caw2.jsonl"), encoding="utf-8")]
    idx = {d: i for i, (d, *_) in enumerate(DOMAINS)}
    keep, seen, dup_pos, drop_syn = [], set(), [], 0
    for r in caw:
        if r["task_id"].startswith("synth:"):
            dom, kind = r["state"].split("|")[0], r["state"].split("|")[1]
            if kind == ("set" if idx[dom] % 2 == 0 else "remove"):
                drop_syn += 1; continue
        elif r["kind"] not in ("persist_backup", "persist_escalate"):
            h = canon([r["task_id"], r["state"], r["msgs"], r["chosen"]])
            if h in seen:
                dup_pos.append(len(keep))
            seen.add(h)
        keep.append(r)
    drop = set(dup_pos[-64:]); assert len(drop) == 64 and drop_syn == 256, (len(drop), drop_syn)
    return [r for i, r in enumerate(keep) if i not in drop]


def main():
    sp = T.specs("train", exclude_heldout_combos=True)
    cf_pool, n_cf = records(T.T2ContractPolicy(), sp, "cf")
    sp_ctrl = both_twins_correct(T.T2AlwaysCheck(), sp)
    ctrl_pool, n_ctrl = records(T.T2AlwaysCheck(), sp_ctrl, "ctrl")
    cf = balanced(cf_pool, QUOTA["cf"], 0); ctrl = balanced(ctrl_pool, QUOTA["ctrl"], 0)
    caw = T.T2AlwaysCheck()
    for r in cf:                        # counterfactual annotation: what would the CAW rule do here, and would it work?
        ca = caw.act(r["msgs"], r["tools"])
        r["caw_action"] = ca; r["disagrees_with_caw"] = canon(ca) != canon(r["chosen"])
        g = trajectory(None, r["spec"], prefix=r["prefix"] + [ca], cont=caw)[1] if r["disagrees_with_caw"] else {"correct": True}
        r["caw_continuation_correct"] = g["correct"]
    abl = []
    for r in cf:
        x = dict(r); x["tools"] = T.FamilyT2(r["spec"]["family"]).tools(r["spec"], strip_contract=True)
        x["state"] = r["state"].replace("cf|", "abl|", 1); x["kind"] = r["kind"].replace("cf_", "abl_", 1); abl.append(x)
    base = base_records(); assert len(base) == 1029, len(base)
    os.makedirs(OUT, exist_ok=True)
    man = {"version": "cf_sft_v1", "K": K, "quota": QUOTA, "n_base": len(base), "train_specs": len(sp), "cf_trajectories": n_cf,
           "ctrl_trajectories": n_ctrl, "ctrl_specs_excluded_unsafe_cells": len(sp) - len(sp_ctrl),
           "cf_pool_unique": len(cf_pool), "ctrl_pool_unique": len(ctrl_pool), "arms": {}}
    for name, add in (("CF", cf), ("CTRL", ctrl), ("ABL", abl)):
        rows = base + [{k: v for k, v in r.items() if k != "prefix"} for r in add]
        p = os.path.join(OUT, f"{name}.jsonl")
        with open(p, "w", encoding="utf-8", newline="\n") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        man["arms"][name] = {"n": len(rows), "sha256": hashlib.sha256(open(p, "rb").read()).hexdigest(),
                             "added_by_role": dict(collections.Counter(r["role"] for r in add)),
                             "added_by_effect": dict(collections.Counter(r["spec"]["effect"] for r in add)),
                             "added_by_verifier": dict(collections.Counter(r["spec"]["verifier"] for r in add)),
                             "added_families": len({r["spec"]["family"] for r in add})}
    man["cf_disagrees_with_caw"] = sum(r["disagrees_with_caw"] for r in cf)
    man["cf_disagree_and_caw_fails"] = sum(r["disagrees_with_caw"] and not r["caw_continuation_correct"] for r in cf)
    json.dump(man, open(os.path.join(OUT, "MANIFEST.json"), "w", encoding="utf-8", newline="\n"), indent=1)
    print(json.dumps(man, indent=1))


if __name__ == "__main__":
    main()
