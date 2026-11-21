"""Training-example builders and arm definitions (B0-B9, pilot Experiments 1-3).

Example records (JSON-serialisable; tools are rebuilt from task_id at train time):
  traj   {"type":"traj","task_id","msgs":[... assistant msgs may carry loss=False]}
  action {"type":"action","task_id","msgs":prefix,"action":a}
  pair   {"type":"pair","task_id","msgs":prefix,"chosen":a+,"rejected":a-,"neg":name,"state":key}
  anchor {"type":"anchor","task_id","msgs":prefix,"action":a~,"kind":"T-out|T-clean|G-clean"}
All sources are TRAIN tasks only. Outage strings come from TRAIN_ERRORS; EVAL_ERRORS are reserved for evaluation.
"""
from __future__ import annotations

import random
import re

from src.counterfactuals.construct import episode_messages, make_episode, Counter
from src.counterfactuals.policies import OutageRule, reference_policy
from src.environment.bfcl_env import ESCALATE, TRAIN_ERRORS, Episode

# ---------------- arm registry ----------------
# init: "base" = Qwen3-4B-Instruct-2507; "pi0" = stage-0 policy (base + success + outage-recovery SFT, merged)
ARMS = {
    "STAGE0":   {"init": "base", "objective": "sft_traj", "data": ["s0", "outage"], "desc": "stage-0 policy pi0 (v2-like recovery SFT)"},
    "B0":       {"init": "base", "objective": None, "desc": "frozen base model (no training)"},
    "B1":       {"init": "base", "objective": "sft_traj", "data": ["s0"], "desc": "success-only SFT"},
    "B2":       {"init": "base", "objective": "sft_traj", "data": ["s0", "repair_traj"], "desc": "success SFT + corrective demonstrations"},
    "B3":       {"init": "base", "objective": "sft_traj", "data": ["s0", "outage", "repair_traj"], "balance": True,
                 "desc": "balanced SFT: real-error repairs and injected-fault recoveries in equal number"},
    "B4":       {"init": "base", "objective": "sft_traj", "data": ["paft"], "desc": "PAFT-style perturbation-augmented SFT (programmatic, alpha=0.3)"},
    "EXP1":     {"init": "pi0", "objective": "sft_action", "data": ["action_pos", "action_twin"],
                 "desc": "balanced corrective SFT on the SAME states as EXP3 (a+ and twin-anchor actions) = ablation F"},
    "B5":       {"init": "pi0", "objective": "sft_action", "data": ["action_pos"], "desc": "localized action-only corrective SFT"},
    "EXP2":     {"init": "pi0", "objective": "dpo", "data": ["pairs"], "desc": "verified counterfactual DPO, no anchor (= B6, CSO-style one round)"},
    "B7_kl":    {"init": "pi0", "objective": "dpo", "data": ["pairs"], "anchor": "kl", "anchor_data": "generic", "lambda": 1.0,
                 "desc": "DPO + fixed KL-to-pi0 on generic clean states"},
    "B7_replay": {"init": "pi0", "objective": "dpo", "data": ["pairs"], "anchor": "nll", "anchor_data": "generic", "lambda": 1.0,
                  "desc": "DPO + generic clean replay (matched anchor count)"},
    "B8_pcgrad": {"init": "pi0", "objective": "dpo", "data": ["pairs"], "anchor": "nll", "anchor_data": "generic", "lambda": 1.0,
                  "grad_mode": "pcgrad", "desc": "DPO + generic replay with PCGrad"},
    "EXP3":     {"init": "pi0", "objective": "dpo", "data": ["pairs"], "anchor": "nll", "anchor_data": "twin", "lambda": 1.0,
                 "desc": "CACRT: verified DPO + counterfactual-twin clean anchors (fixed lambda)"},
    "EXP3_agem": {"init": "pi0", "objective": "dpo", "data": ["pairs"], "anchor": "nll", "anchor_data": "twin", "lambda": 1.0,
                  "grad_mode": "agem", "desc": "CACRT with adaptive (A-GEM-projected) anchoring - ablation E"},
    "EXP3_unverified": {"init": "pi0", "objective": "dpo", "data": ["pairs_unverified"], "anchor": "nll", "anchor_data": "twin",
                        "lambda": 1.0, "desc": "ablation B: pairs from ALL candidate states, outcome verification skipped"},
}
PILOT = ["EXP1", "EXP2", "EXP3", "B7_replay"]


# ---------------- trajectory builders ----------------
def _traj_record(task, ep, closing=True, loss_false_upto=0):
    """Episode -> SFT record. Earlier-turn history and the first `loss_false_upto` episode actions (a forced erroneous
    prefix) are context only; every later assistant action is a target."""
    msgs = episode_messages(task, ep)
    n_hist = len(msgs) - 2 * len(ep.calls)
    for m in msgs[:n_hist]:
        if m["role"] == "assistant":
            m["loss"] = False
    for k, i in enumerate(range(n_hist, len(msgs), 2)):
        msgs[i]["loss"] = k >= loss_false_upto
    if closing and not ep.escalated:
        msgs.append({"role": "assistant", "content": "Done.", "loss": True})
    return {"type": "traj", "task_id": task["id"], "msgs": msgs, "scenario": ep.scenario}


def s0_trajs(tasks):
    out = []
    for t in tasks:
        ep = Episode(t, "S0", 0, error_pool=TRAIN_ERRORS); reference_policy(ep)
        assert ep.grade()["correct"]
        out.append(_traj_record(t, ep))
    return out


def outage_trajs(tasks, seeds=(0,)):
    out = []
    for t in tasks:
        for sc in ("S1", "S2", "S3"):
            for sd in seeds:
                ep = Episode(t, sc, sd, error_pool=TRAIN_ERRORS); reference_policy(ep)
                assert ep.grade()["correct"]
                out.append(_traj_record(t, ep))
    return out


def repair_trajs(tasks_by_id, states):
    """Forced erroneous prefix (loss=False) + executed repair plan; kept only if the result passes the grader."""
    out, c = [], Counter()
    for s in states:
        if not s["retained_pairs"]:
            continue
        t = tasks_by_id[s["task_id"]]
        ep = make_episode(t, s["forced"], c)
        OutageRule().continue_from(ep, s["repair_plan"])
        if ep.grade()["correct"]:
            out.append(_traj_record(t, ep, loss_false_upto=len(s["forced"])))
    return out


def action_records(states):
    return [{"type": "action", "task_id": s["task_id"], "msgs": s["messages"], "action": s["pos_action"], "state": s["key"]}
            for s in states if s["retained_pairs"]]


def pair_records(states, verified=True):
    out = []
    for s in states:
        negs = s["retained_pairs"] if verified else list(s["neg_actions"])
        for nm in negs:
            out.append({"type": "pair", "task_id": s["task_id"], "msgs": s["messages"], "chosen": s["pos_action"],
                        "rejected": s["neg_actions"][nm], "neg": nm, "state": s["key"]})
    return out


def anchor_records(anchors):
    return [{"type": "anchor", "task_id": a["task_id"], "msgs": a["messages"], "action": a["action"], "kind": a["kind"],
             "negs": a.get("negs", {})} for a in anchors]


# ---------------- PAFT-style (programmatic) ----------------
_SYN = {"get": "fetch", "send": "dispatch", "create": "make", "add": "append", "remove": "drop", "delete": "erase",
        "list": "enumerate", "display": "show", "find": "search", "check": "verify", "set": "assign", "update": "modify",
        "book": "reserve", "cancel": "revoke", "view": "inspect", "post": "publish", "start": "begin", "lock": "secure"}


def srp_rename_map(names, rng):
    """Symbolic Representation Perturbation analogue: consistent synonym renaming of tool names."""
    m = {}
    for n in names:
        parts = re.split(r"(?<=[a-z])(?=[A-Z])|_", n)
        new = [(_SYN.get(p.lower(), p) if rng.random() < 0.8 else p) for p in parts]
        cand = "_".join(x.lower() for x in new)
        m[n] = cand if cand != n and cand not in m.values() else n + "_v2"
    return m


def paft_style(tasks, tasks_by_id, states, alpha=0.3, seed=0):
    """(1-alpha) clean S0 trajectories U alpha perturbed: EFP (feedback perturbations with corrective continuation:
    injected-outage recoveries S1/S2 and real-error repairs), SBP (fatal failure -> refusal: S3 escalation), SRP
    (consistently renamed tools). PAFT's own generator calls gpt-4o-mini; here every perturbation is executed in the
    environment instead (no paid API), so this is a faithful-in-spirit re-implementation, not the official data."""
    rng = random.Random(seed)
    clean = s0_trajs(tasks)
    out = outage_trajs(tasks)
    s1s2 = [r for r in out if r["scenario"] in ("S1", "S2")]
    s3 = [r for r in out if r["scenario"] == "S3"]
    pool_efp = s1s2 + repair_trajs(tasks_by_id, states)
    rng.shuffle(pool_efp); rng.shuffle(s3)
    n_pert = int(round(len(clean) * alpha / (1 - alpha)))
    n_efp, n_sbp = n_pert // 2, n_pert // 4
    n_srp = n_pert - n_efp - n_sbp
    srp = [{**r, "tool_rename_seed": rng.randrange(10 ** 6)} for r in rng.sample(clean, min(n_srp, len(clean)))]
    return clean + pool_efp[:n_efp] + s3[:n_sbp] + srp


def balance(groups, rng):
    n = min(len(g) for g in groups)
    return [x for g in groups for x in rng.sample(g, n)]
