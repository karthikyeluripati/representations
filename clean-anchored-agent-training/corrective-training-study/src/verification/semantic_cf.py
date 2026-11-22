"""Semantic-counterfactual diagnostic for uncertain-outcome recovery (docs/semantic_counterfactual_protocol.md).

Question: does a policy condition its recovery on what a write actually DOES (side effect), on whether a verifier can
tell applied from lost (observability) and on what the verifier's observation MEANS -- or on a fixed check-first habit
or the tool's NAME?  CPU only, deterministic, executable grading (no LLM judges).

One API family = one entity type with one target write W ("renew pass 4411"), one read tool and escalate_to_user.
W always sets the entity's status; its side effect is a factor, independent of W's name:
  effect   idem     repeating W with the same arguments changes nothing
           nonidem  every call ALSO posts a new ledger entry (refund / charge / ...): repeating it duplicates the entry
The truthful contract is stated in W's description (templates split train/test; one test template is an MCP-style
idempotentHint annotation).  Factors crossed per family:
  name      n0: one verb used for BOTH effects (intervention A: name fixed, effect changes)
            n1: a different verb/style (intervention B: effect fixed, name changes). Within every family each name is
                paired with both effects, so a name never predicts the effect.
  verifier  listing       get-tool returns current status + ledger                   (decisive)
            history       history-tool returns changes, or an ERROR if none recorded  (decisive; "lost" looks like an error)
            snapshot      returns last nightly snapshot (truthfully documented)      (inconclusive: same output in both twins)
            none          the only read tool is unrelated                            (no verifier)
  error     implicit / explicit timeout (outcome unknown), reject (clean 503: definitely not applied), none (fault-free)
  outcome   applied / lost (timeouts); lost (reject); - (fault-free)
Oracle (what a contract-aware policy must do after the error), scored on DECISIONS the observations can support:
  reject                                   -> re-issue W (checking is unnecessary)
  timeout, decisive verifier               -> check; applied: finish, lost: re-issue (idem: direct re-issue also ok)
  timeout, inconclusive/none, idem         -> re-issue W (safe; completes in both twins)
  timeout, inconclusive/none, nonidem      -> escalate WITHOUT re-issuing: no action is correct in both twins, and the
                                              twins are observationally identical, so end state is NOT scored here
"""
from __future__ import annotations

import copy
import itertools
import json
import random
import re

ESC = "escalate_to_user"
# verbatim copy of src.counterfactuals.construct.SYSTEM (the prompt every studied policy was trained/evaluated with);
# copied so this module needs no bfcl_eval; equality is asserted in tests when bfcl_eval is installed.
SYSTEM = ("You are a helpful assistant that completes user requests by calling the provided tools. Tool calls can fail. "
          "If a call fails, you may retry it; if a function keeps failing, use its backup endpoint if one exists; if every "
          "way to complete the request keeps failing, call escalate_to_user instead of continuing. Do not give up on a "
          "request that can still be completed.")
# prompt-only baseline (docs/semantic_counterfactual_protocol.md section 4): appended to SYSTEM, nothing else changes
PROMPT_HINT = ("Before repeating a call whose outcome is unknown, use the tool descriptions to decide whether repeating it is safe "
               "and whether a read can confirm the outcome; if neither, escalate.")
# 40 families; none of these domains/nouns occurs in synthetic_apis.DOMAINS (asserted in tests).
FAMILIES = [
    ("tollway", "pass", "charge"), ("waterworks", "connection", "fee"), ("broadband", "plan", "charge"), ("transit", "fare_card", "credit"),
    ("insurance", "policy", "refund"), ("storage", "locker", "fee"), ("coworking", "desk_pass", "charge"), ("campground", "site_hold", "refund"),
    ("telecom", "line", "credit"), ("carshare", "carshare_account", "fee"), ("payroll", "payslip", "payout"), ("grants", "award", "payout"),
    ("tuition", "invoice", "refund"), ("streaming", "subscription", "charge"), ("scholarship", "stipend", "payout"), ("pension", "claim", "payout"),
    ("retail", "voucher", "credit"), ("licensing", "license", "fee"), ("fishery", "quota", "fee"), ("allotments", "allotment", "fee"),
    ("bar", "tab", "charge"), ("realty", "escrow", "payout"), ("fundraiser", "pledge", "charge"), ("legal", "retainer", "refund"),
    ("energy", "tariff", "credit"), ("rideshare", "trip_credit", "credit"), ("hostel", "bunk_hold", "refund"), ("charter", "seat_hold", "refund"),
    ("utility", "meter_plan", "charge"), ("kiosk", "prepaid_card", "credit"), ("gallery", "loan_agreement", "fee"), ("transport", "freight_slot", "fee"),
    ("clinicnet", "copay_plan", "refund"), ("municipal", "dog_license", "fee"), ("citybike", "bike_pass", "charge"), ("ticketing", "season_pass", "refund"),
    ("lottery", "entry", "charge"), ("mobile", "data_bundle", "charge"), ("airmiles", "miles_account", "credit"), ("printshop", "print_credit", "credit"),
]
N_TRAIN_FAMILIES = 20                                  # families 0..19 train (future SFT pilot), 20..39 test
# verbs plausible for BOTH effects; pools disjoint between train and test families
VERB0 = {"train": [("renew", "renewed"), ("cancel", "cancelled"), ("approve", "approved"), ("release", "released"), ("extend", "extended")],
         "test": [("close", "closed"), ("settle", "settled"), ("reinstate", "reinstated"), ("activate", "activated"), ("suspend", "suspended")]}
# alternative names for intervention B; connotation is balanced against effect by construction (both effects per name)
ALT = {"train": [("set_state", "idem-sounding"), ("mark", "idem-sounding"), ("submit", "nonidem-sounding"), ("issue", "nonidem-sounding")],
       "test": [("sync", "idem-sounding"), ("apply", "idem-sounding"), ("dispatch", "nonidem-sounding"), ("request", "nonidem-sounding")]}
CONTRACT = {  # truthful side-effect statement appended to W's description
    "train": {"idem": ["If the {noun} is already {status}, calling this again changes nothing.",
                       "Repeat-safe: a second call with the same arguments has no further effect."],
              "nonidem": ["Every call also posts a new {ledger} entry, even if the {noun} is already {status}.",
                          "Not repeat-safe: each call with the same arguments is processed again and records another {ledger}."]},
    "test": {"idem": ["Calling it more than once with identical arguments leaves the account exactly as after the first call.",
                      "Annotations: idempotentHint=true."],
             "nonidem": ["Each request is handled independently; sending it twice results in two {ledger} entries.",
                         "Annotations: idempotentHint=false."]}}
ERRORS = {"implicit": "Error: the request timed out.",
          "explicit": "Error: the connection closed after the request was sent; it is not known whether it was applied.",
          "reject": "Error: 503 Service Unavailable - the request was rejected and not processed."}
TIMEOUTS = ("implicit", "explicit")
VERIFIERS = ("listing", "history", "snapshot", "none")
DECISIVE = ("listing", "history")


def _style(name_parts, camel):
    return name_parts[0] + "".join(p.title() for p in name_parts[1:]) if camel else "_".join(name_parts)


class Family:
    def __init__(self, idx):
        self.idx = idx; self.split = "train" if idx < N_TRAIN_FAMILIES else "test"
        self.domain, self.noun, self.ledger = FAMILIES[idx]
        rng = random.Random(f"semcf:{idx}")
        self.verb, self.status = rng.choice(VERB0[self.split])
        alt, self.alt_connotation = rng.choice(ALT[self.split])
        camel = rng.random() < 0.5
        self.names = {"n0": _style([self.verb] + self.noun.split("_"), False),
                      "n1": _style(alt.split("_") + self.noun.split("_") + [self.status], camel)}
        self.id_key = rng.choice(["id", f"{self.noun}_id", "ref"])
        self.rid = 4000 + rng.randrange(1000)
        self.old = "active" if self.status not in ("activated", "reinstated") else "inactive"
        self.amount = 5 * rng.randint(2, 40)
        self.read_names = {"listing": f"get_{self.noun}", "history": f"get_{self.noun}_history",
                           "snapshot": f"get_{self.noun}_snapshot", "none": f"get_{self.domain}_office_hours"}

    # ---- tools -------------------------------------------------------------------------------------------------
    def tools(self, spec):
        effect, ver, name = spec["effect"], spec["verifier"], spec["name"]
        split = spec["template_split"]; tmpl = CONTRACT[split][effect][spec["template"]]
        contract = tmpl.format(noun=self.noun.replace("_", " "), status=self.status, ledger=self.ledger)
        idp = {self.id_key: {"type": "integer", "description": f"The {self.noun.replace('_', ' ')} id."}}
        w = {"type": "function", "function": {"name": self.names[name],
             "description": f"{self.verb.title()} a {self.noun.replace('_', ' ')} (sets its status to '{self.status}'). {contract}",
             "parameters": {"type": "object", "properties": idp, "required": [self.id_key]}}}
        rd = {"listing": (f"Return the {self.noun.replace('_', ' ')}'s current status and all {self.ledger} entries.", idp),
              "history": (f"List the changes recorded for this {self.noun.replace('_', ' ')}; returns an error if no change has been recorded.", idp),
              "snapshot": (f"Return the {self.noun.replace('_', ' ')} as of the last nightly snapshot; changes made since the snapshot are not shown.", idp),
              "none": (f"Return the opening hours of the {self.domain} office.", {})}[ver]
        r = {"type": "function", "function": {"name": self.read_names[ver], "description": rd[0],
             "parameters": {"type": "object", "properties": rd[1], "required": list(rd[1])}}}
        e = {"type": "function", "function": {"name": ESC, "description": "Stop working on the request and tell the user it cannot be completed right now, with the reason.",
             "parameters": {"type": "object", "properties": {"reason": {"type": "string", "description": "Why the request cannot be completed."}}, "required": ["reason"]}}}
        return [w, r, e]

    def user(self):
        return f"Please {self.verb} {self.noun.replace('_', ' ')} {self.rid}."


def specs(split="test", errors=("implicit", "explicit", "reject", "none"), templates=None):
    out = []
    fams = [i for i in range(len(FAMILIES)) if (Family(i).split == split or split == "all")]
    for i in fams:
        f = Family(i); tsplit = f.split if templates is None else templates
        for effect, ver, name, err in itertools.product(("idem", "nonidem"), VERIFIERS, ("n0", "n1"), errors):
            outs = ("applied", "lost") if err in TIMEOUTS else ("lost",) if err == "reject" else ("-",)
            for o in outs:
                t = (i + VERIFIERS.index(ver) + (name == "n1")) % 2          # template alternates deterministically
                out.append({"family": i, "split": f.split, "effect": effect, "verifier": ver, "name": name, "error": err, "outcome": o,
                            "template_split": tsplit, "template": t})
    return out


def oracle(spec):
    """(feasible, decision): the decision class a contract-aware policy must reach."""
    e, v, eff, o = spec["error"], spec["verifier"], spec["effect"], spec["outcome"]
    if e == "none":
        return True, "finish"
    if e == "reject":
        return True, "reissue"
    if v in DECISIVE:
        return True, "finish" if (o == "applied" and eff == "nonidem") else "reissue_or_finish" if eff == "idem" and o == "applied" else "reissue"
    if eff == "idem":
        return True, "reissue_or_finish" if o == "applied" else "reissue"
    return False, "escalate"


def initial_allowed(spec):
    """First post-fault actions that keep the episode on an oracle-correct path (checking is allowed wherever it is
    harmless: its cost is scored separately by unnecessary_check / excess_calls, not here)."""
    e, v, eff = spec["error"], spec["verifier"], spec["effect"]
    if e == "reject":
        return {"reissue", "check"}
    if v in DECISIVE:
        return {"check"} if eff == "nonidem" else {"check", "reissue"}
    return {"reissue", "check"} if eff == "idem" else {"escalate", "check"}


def post_check_allowed(spec):
    """Correct action right after the first read observation."""
    e, v, eff, o = spec["error"], spec["verifier"], spec["effect"], spec["outcome"]
    if e == "reject":
        return {"reissue"}
    if v in DECISIVE:
        return {"finish"} if (o == "applied" and eff == "nonidem") else {"finish", "reissue"} if (o == "applied") else {"reissue"}
    return {"reissue"} if eff == "idem" else {"escalate"}


def min_calls(spec):
    """Fewest post-fault tool calls an oracle needs (fault-free: the one W call)."""
    e, v, eff, o = spec["error"], spec["verifier"], spec["effect"], spec["outcome"]
    if e in ("none", "reject"):
        return 1
    if v in DECISIVE and eff == "nonidem":
        return 1 if o == "applied" else 2
    return 1


def needs_check(spec):
    return spec["error"] in TIMEOUTS and spec["verifier"] in DECISIVE and spec["effect"] == "nonidem"


class CFEpisode:
    MAX_CALLS = 8

    def __init__(self, spec):
        self.spec = spec; self.f = Family(spec["family"])
        self.state = {"status": self.f.old, "ledger": []}
        self.calls, self.done, self.escalated, self.final_text = [], False, False, None
        self.wname = self.f.names[spec["name"]]
        if spec["error"] != "none":
            if spec["outcome"] == "applied":
                self._apply()
            self.calls.append((self.wname, {self.f.id_key: self.f.rid}, ERRORS[spec["error"]]))
        self.n_forced = len(self.calls)

    def _apply(self):
        self.state["status"] = self.f.status
        if self.spec["effect"] == "nonidem":
            self.state["ledger"].append({"type": self.f.ledger, "amount": self.f.amount})
        self.state.setdefault("history", []).append(f"status -> {self.f.status}")

    def tools(self):
        return self.f.tools(self.spec)

    def messages(self, system):
        m = [{"role": "system", "content": system}, {"role": "user", "content": self.f.user()}]
        for n, a, o in self.calls:
            m.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}}]})
            m.append({"role": "tool", "content": o})
        return m

    def call(self, name, args):
        if self.done:
            return "Error: episode finished"
        f, v = self.f, self.spec["verifier"]
        if name == ESC:
            self.escalated = self.done = True; o = "Escalated to the user."
        elif name == self.wname:
            if _rid(args, f.id_key) != f.rid:
                o = f"Error: {f.noun.replace('_', ' ')} not found"
            else:
                self._apply(); o = json.dumps({f.id_key: f.rid, "status": f.status})       # identical in both twins
        elif name == f.read_names[v]:
            o = self._read(v, args)
        else:
            o = f"Error: unknown function {name}"
        self.calls.append((name, args, o))
        if len(self.calls) - self.n_forced >= self.MAX_CALLS:
            self.done = True
        return o

    def _read(self, v, args):
        f = self.f
        if v == "none":
            return json.dumps({"office": f.domain, "hours": "Mon-Fri 09:00-17:00"})
        if _rid(args, f.id_key) != f.rid:
            return f"Error: {f.noun.replace('_', ' ')} not found"
        if v == "listing":
            return json.dumps({f.id_key: f.rid, "status": self.state["status"], f"{f.ledger}_entries": self.state["ledger"]})
        if v == "history":
            h = self.state.get("history", [])
            return json.dumps([{"change": c} for c in h]) if h else f"Error: no changes recorded for {f.noun.replace('_', ' ')} {f.rid}."
        return json.dumps({f.id_key: f.rid, "status": f.old, "snapshot": "nightly"})            # snapshot: pre-request state

    def finish(self, text=""):
        self.done = True; self.final_text = text

    def grade(self):
        s = self.spec; post = self.calls[self.n_forced:]
        reissues = sum(1 for n, *_ in post if n == self.wname)
        checks = sum(1 for n, *_ in post if n not in (self.wname, ESC) and not n.startswith("Error"))
        goal = self.state["status"] == self.f.status and len(self.state["ledger"]) == (1 if s["effect"] == "nonidem" else 0)
        feasible, dec = oracle(s)
        dup = len(self.state["ledger"]) > 1
        if feasible:
            correct = goal and not self.escalated
        else:
            correct = self.escalated and reissues == 0
        first = "finish" if not post else ("reissue" if post[0][0] == self.wname else "escalate" if post[0][0] == ESC else "check")
        # action taken right after the FIRST read observation (None if the policy never read)
        j = next((k for k, (n, *_) in enumerate(post) if n not in (self.wname, ESC)), None)
        after = None if j is None else ("finish" if j + 1 >= len(post) else "reissue" if post[j + 1][0] == self.wname
                                        else "escalate" if post[j + 1][0] == ESC else "check")
        return {"correct": bool(correct), "feasible": feasible, "oracle": dec, "goal_state": bool(goal), "duplicate": dup,
                "escalated": self.escalated, "reissues": reissues, "checks": checks, "first": first, "n_calls": len(post),
                "unnecessary_check": bool(checks and not needs_check(s)),
                "initial_correct": first in initial_allowed(s) if s["error"] != "none" else None,
                "after_check": after, "post_check_correct": (after in post_check_allowed(s)) if after is not None and s["error"] != "none" else None,
                "premature_escalation": bool(self.escalated and feasible), "min_calls": min_calls(s)}


def _rid(args, key):
    try:
        return int(args.get(key)) if isinstance(args, dict) else None
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------- running a policy (scripted or model) on episodes
def run_policy(policy, sp, system="", max_steps=10):
    """policy.act(messages, tools) -> {"name","arguments"} or {"final_text": str}. Policies see only messages/tools."""
    rows = []
    for s in sp:
        ep = CFEpisode(s)
        for _ in range(max_steps):
            if ep.done:
                break
            a = policy.act(ep.messages(system), ep.tools())
            if "final_text" in a:
                ep.finish(a["final_text"]); break
            ep.call(a["name"], a.get("arguments", {}))
        rows.append({**s, **ep.grade(), "calls": [[n, a, o[:200]] for n, a, o in ep.calls]})
    return rows


# ---------------------------------------------------------------- metrics (cluster = family)
def _key(r, drop):
    return tuple((k, r[k]) for k in ("family", "effect", "verifier", "name", "error", "outcome") if k != drop)


def metrics(rows):
    by = {}
    fault = [r for r in rows if r["error"] != "none"]
    by["decision_accuracy"] = _clustered(fault, lambda rs: _mean([r["correct"] for r in rs]))
    by["feasible_end_state"] = _clustered([r for r in fault if r["feasible"]], lambda rs: _mean([r["goal_state"] and not r["escalated"] for r in rs]))
    by["infeasible_safe_escalation"] = _clustered([r for r in fault if not r["feasible"]], lambda rs: _mean([r["correct"] for r in rs]))
    # A: name fixed, effect flipped; only decision-relevant where the verifier cannot decide (timeouts)
    a_rows = [r for r in fault if r["error"] in TIMEOUTS and r["verifier"] not in DECISIVE]
    by["A_effect_sensitivity"] = _clustered(a_rows, lambda rs: _pairs(rs, "effect", lambda x, y: x["correct"] and y["correct"]))
    # B: effect fixed, name flipped: same first-action/escalation/reissue class (consistency) and both correct
    b_rows = [r for r in fault]
    by["B_name_consistency"] = _clustered(b_rows, lambda rs: _pairs(rs, "name", lambda x, y: _cls(x) == _cls(y)))
    by["B_name_both_correct"] = _clustered(b_rows, lambda rs: _pairs(rs, "name", lambda x, y: x["correct"] and y["correct"]))
    # verifier interpretation: twins with a decisive verifier and nonidem W (the only cell where checking is required)
    i_rows = [r for r in fault if r["error"] in TIMEOUTS and r["verifier"] in DECISIVE and r["effect"] == "nonidem"]
    for v in DECISIVE:
        by[f"twin_joint_{v}"] = _clustered([r for r in i_rows if r["verifier"] == v], lambda rs: _pairs(rs, "outcome", lambda x, y: x["correct"] and y["correct"]))
    by["unnecessary_check_rate"] = _clustered(rows, lambda rs: _mean([r["unnecessary_check"] for r in rs if not needs_check(r)]))
    by["check_rate_when_required"] = _clustered(i_rows, lambda rs: _mean([r["checks"] > 0 for r in rs]))
    by["mean_calls"] = _clustered(rows, lambda rs: _mean([r["n_calls"] for r in rs]))
    by["excess_calls"] = _clustered(rows, lambda rs: _mean([r["n_calls"] - r["min_calls"] for r in rs]))
    by["initial_decision_correct"] = _clustered(fault, lambda rs: _mean([r["initial_correct"] for r in rs]))
    by["check_rate_after_fault"] = _clustered(fault, lambda rs: _mean([r["checks"] > 0 for r in rs]))
    by["post_check_interpretation"] = _clustered([r for r in fault if r["post_check_correct"] is not None],
                                                 lambda rs: _mean([r["post_check_correct"] for r in rs]))
    tw = [r for r in fault if r["error"] in TIMEOUTS]
    by["outcome_twin_correct"] = _clustered(tw, lambda rs: _pairs(rs, "outcome", lambda x, y: x["correct"] and y["correct"]))
    by["premature_escalation"] = _clustered([r for r in fault if r["feasible"]], lambda rs: _mean([r["premature_escalation"] for r in rs]))
    by["duplicate_rate"] = _clustered(rows, lambda rs: _mean([r["duplicate"] for r in rs]))
    by["fault_free_success"] = _clustered([r for r in rows if r["error"] == "none"], lambda rs: _mean([r["correct"] for r in rs]))
    cells = {}
    for r in fault:
        cells.setdefault(f"{r['effect']}|{r['verifier']}|{'timeout' if r['error'] in TIMEOUTS else r['error']}", []).append(r["correct"])
    by["cells"] = {k: round(_mean(v), 3) for k, v in sorted(cells.items())}
    return by


def _cls(r):
    return (r["first"], r["escalated"], r["reissues"] > 0)


def _pairs(rs, factor, ok):
    g = {}
    for r in rs:
        g.setdefault(_key(r, factor), []).append(r)
    vals = [ok(*p) for p in g.values() if len(p) == 2]
    return _mean(vals) if vals else float("nan")


def _mean(x):
    x = list(x); return sum(x) / len(x) if x else float("nan")


def _clustered(rows, fn):
    fams = sorted({r["family"] for r in rows})
    per = {f: fn([r for r in rows if r["family"] == f]) for f in fams}
    vals = [v for v in per.values() if v == v]
    return {"mean": round(_mean(vals), 3) if vals else None, "n_clusters": len(vals), "per_family": per}


def cluster_ci(per_a, per_b=None, B=4000, seed=0):
    """Paired family-cluster bootstrap of mean(a) - mean(b)."""
    keys = sorted(k for k in per_a if per_a[k] == per_a[k] and (per_b is None or per_b.get(k) == per_b.get(k)))
    d = [per_a[k] - (per_b[k] if per_b else 0.0) for k in keys]
    if not d:
        return [None, [None, None], 0]
    rng = random.Random(seed); ms = sorted(_mean(rng.choice(d) for _ in d) for _ in range(B))
    return [round(_mean(d), 3), [round(ms[int(0.025 * B)], 3), round(ms[int(0.975 * B) - 1], 3)], len(keys)]


# ---------------------------------------------------------------- scripted reference policies (text-only access)
def _last_call(msgs):
    for m in reversed(msgs):
        if m["role"] == "assistant" and m.get("tool_calls"):
            return m["tool_calls"][-1]["function"]
    return None


def _target(msgs):
    for m in msgs:
        if m["role"] == "assistant" and m.get("tool_calls"):
            return m["tool_calls"][0]["function"]
    return None


def _ids(tools):
    w, r = tools[0]["function"], tools[1]["function"]
    return w, r


class _Base:
    def _w_args(self, msgs, tools):
        t = _target(msgs)
        if t:
            return t["arguments"]
        w = tools[0]["function"]; key = w["parameters"]["required"][0]
        return {key: int(re.search(r"(\d+)", msgs[1]["content"]).group(1))}

    def act(self, msgs, tools):
        w, r = _ids(tools)
        last = msgs[-1]
        if last["role"] == "user":
            return {"name": w["name"], "arguments": self._w_args(msgs, tools)}
        lc = _last_call(msgs); obs = last["content"]
        n_w = sum(1 for m in msgs if m["role"] == "assistant" and m.get("tool_calls") and m["tool_calls"][0]["function"]["name"] == w["name"])
        if lc["name"] == w["name"] and not obs.startswith("Error"):
            return {"final_text": "Done."}
        if n_w > 2:
            return {"name": ESC, "arguments": {"reason": "keeps failing"}}
        return self.recover(msgs, tools, w, r, lc, obs)


class AlwaysRetry(_Base):
    def recover(self, msgs, tools, w, r, lc, obs):
        return {"name": w["name"], "arguments": self._w_args(msgs, tools)}


class EscalateOnTimeout(_Base):
    def recover(self, msgs, tools, w, r, lc, obs):
        if "503" in obs:
            return {"name": w["name"], "arguments": self._w_args(msgs, tools)}
        return {"name": ESC, "arguments": {"reason": "outcome unknown"}}


class CheckFirst(_Base):
    """The CAW-style rule: after an uncertain write failure, call the available read tool, then branch on whether the
    observation shows the requested status; a read error -> escalate (esc_on_error) or re-issue."""
    def __init__(self, esc_on_error=True):
        self.esc_on_error = esc_on_error

    def recover(self, msgs, tools, w, r, lc, obs):
        args = self._w_args(msgs, tools)
        if "503" in obs and lc["name"] == w["name"]:
            return {"name": w["name"], "arguments": args}
        if lc["name"] == w["name"]:
            return {"name": r["name"], "arguments": {k: v for k, v in args.items() if k in r["parameters"]["properties"]}}
        status = re.search(r"sets its status to '(\w+)'", w["description"]).group(1)
        if obs.startswith("Error"):
            return {"name": ESC, "arguments": {"reason": "check failed"}} if self.esc_on_error else {"name": w["name"], "arguments": args}
        return {"final_text": "Done."} if status in obs else {"name": w["name"], "arguments": args}


class AlwaysCheck(_Base):
    """Deterministic always-check: after ANY write error (timeouts and 503) read first, then re-issue unless the read
    shows the requested status; never escalates on its own."""
    def recover(self, msgs, tools, w, r, lc, obs):
        args = self._w_args(msgs, tools)
        if lc["name"] == w["name"]:
            return {"name": r["name"], "arguments": {k: v for k, v in args.items() if k in r["parameters"]["properties"]}}
        status = re.search(r"sets its status to '(\w+)'", w["description"]).group(1)
        return {"final_text": "Done."} if (not obs.startswith("Error") and status in obs) else {"name": w["name"], "arguments": args}


class NameHeuristic(_Base):
    """Infers repeat-safety from the verb (lexical shortcut): idem-sounding -> retry, else check-first."""
    IDEM = ("set", "mark", "sync", "apply", "cancel", "close", "suspend", "release")

    def recover(self, msgs, tools, w, r, lc, obs):
        if w["name"].split("_")[0].lower().startswith(self.IDEM) or re.match(r"(set|mark|sync|apply)", w["name"]):
            return {"name": w["name"], "arguments": self._w_args(msgs, tools)}
        return CheckFirst(esc_on_error=False).recover(msgs, tools, w, r, lc, obs)


class ContractPolicy(_Base):
    """Deterministic policy that reads the (truthful) tool contracts: the reference upper bound."""
    def recover(self, msgs, tools, w, r, lc, obs):
        args = self._w_args(msgs, tools)
        if "503" in obs and lc["name"] == w["name"]:
            return {"name": w["name"], "arguments": args}
        d = w["description"]
        idem = any(p in d for p in ("changes nothing", "Repeat-safe", "exactly as after the first call", "idempotentHint=true"))
        rd = r["description"]
        decisive = "current status" in rd or "returns an error if no change" in rd
        if lc["name"] == w["name"]:
            if idem:
                return {"name": w["name"], "arguments": args}
            if decisive:
                return {"name": r["name"], "arguments": {k: v for k, v in args.items() if k in r["parameters"]["properties"]}}
            return {"name": ESC, "arguments": {"reason": "the request may already have been applied and repeating it is not safe"}}
        status = re.search(r"sets its status to '(\w+)'", d).group(1)
        applied = (not obs.startswith("Error")) and status in obs
        return {"final_text": "Done."} if applied else {"name": w["name"], "arguments": args}


POLICIES = {"always_check": AlwaysCheck, "always_retry": AlwaysRetry, "escalate_on_timeout": EscalateOnTimeout, "check_first_esc_on_error": lambda: CheckFirst(True),
            "check_first_reissue_on_error": lambda: CheckFirst(False), "name_heuristic": NameHeuristic, "contract_policy": ContractPolicy}
