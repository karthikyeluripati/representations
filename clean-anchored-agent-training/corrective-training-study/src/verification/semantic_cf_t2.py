"""Generator template T2 for contract-conditioned recovery (docs/cf_sft_pilot_protocol.md). CPU only, deterministic.

T2 is structurally different from the frozen T1 diagnostic (semantic_cf.py, unchanged): the target write W POSTS an
amount to an account (it does not set a status); reads come in other forms; W's success response is a receipt listing
today's active transactions; some families expose a compensating reverse tool; one effect is made safe by a client
idempotency key. Splits: 30 train / 10 dev / 20 test families, with disjoint verbs, contract phrasings and error wordings.

Factors (fully crossed per family):
  effect        idem    W sets the account's pending top-up; repeating it with the same arguments changes nothing
                nonidem every call books a new top-up transaction
                keyed   like nonidem, but the original call carried a request_key; requests reusing it are applied once
  verifier      decisive (get-account or activity listing, form varies by family) | stale (documented end-of-day
                statement: identical output in both twins) | none (unrelated price list)
  compensation  absent | present (reverse_<noun>_transaction(transaction_id))
  name          n0 (verb shared by all effects) | n1 (another verb + naming style)
  error         timeout_a / timeout_b (outcome unknown) | reject (clean, not processed) | none (fault-free)
  outcome       applied / lost (timeouts); lost (reject)
Oracle (decisions an observation-limited, contract-reading policy can justify):
  reject                                        -> re-issue
  timeout, idem                                 -> re-issue (safe)
  timeout, keyed                                -> re-issue with the same request_key (safe)
  timeout, nonidem, decisive                    -> check; applied: finish, lost: re-issue
  timeout, nonidem, stale/none, compensation    -> RECONCILE: re-issue, read the receipt, reverse the duplicate if two
  timeout, nonidem, stale/none, no compensation -> escalate without re-issuing (unresolvable: not scored on end state)
"""
from __future__ import annotations

import itertools
import json
import random
import re

from verification.semantic_cf import ESC, SYSTEM, _clustered, _mean, cluster_ci  # noqa: F401  (shared helpers; T1 unchanged)

DOMAINS_T2 = [
    "canteen", "laundromat", "makerspace", "esports", "daycare", "tutoring", "golfclub", "climbing", "vending", "carwash",
    "evcharging", "spa", "karaoke", "bowling", "cafeteria", "photobooth", "skatepark", "campusclub", "rowing", "sailing",
    "archery", "fencing", "paddle", "curling", "trampoline", "escaperoom", "minigolf", "aquapark", "icerink", "foodtruck",
    "juicebar", "teahouse", "bookclub", "chessclub", "choir", "danceschool", "yoga", "pilates", "boxing", "judo",
    "swimclub", "cyclingclub", "runclub", "hikingclub", "birdclub", "robotics", "codingcamp", "artclass", "filmclub", "quizleague",
    "triviabar", "boardgamecafe", "lasertag", "paintball", "gokart", "velodrome", "ropecourse", "zipline", "planetaria", "rinkside",
]
SPLITS = {"train": range(0, 30), "dev": range(30, 40), "test": range(40, 60)}
NOUN_SUFFIX = ["wallet", "account", "balance", "card"]
VERBS = {"train": ["top_up", "load", "add_funds", "credit"], "dev": ["recharge", "refill"], "test": ["replenish", "deposit"]}
CONTRACT = {  # truthful retry-behaviour statement in W's description, per split
    "train": {"idem": ["Retry behaviour: safe. The request sets the pending top-up, so repeating it with the same amount has no additional effect.",
                       "Repeating this call with identical arguments is harmless: it only (re)sets the pending top-up."],
              "nonidem": ["Retry behaviour: unsafe. Every request books a separate top-up transaction.",
                          "Each call is booked as its own transaction, including repeats with identical arguments."],
              "keyed": ["Retry behaviour: deduplicated. Requests that reuse a request_key are applied at most once; without it, each request is booked separately.",
                        "Supply request_key to make retries safe: a repeated request_key is ignored by the server."]},
    "dev": {"idem": ["Re-sending the same request leaves the pending top-up unchanged."],
            "nonidem": ["Re-sending the same request books another top-up."],
            "keyed": ["Re-sending a request with an already used request_key has no further effect."]},
    "test": {"idem": ["The server treats a repeated identical request as the same top-up (no extra credit).",
                      "x-retry: safe"],
             "nonidem": ["A repeated identical request is credited again as a new top-up.",
                         "x-retry: unsafe"],
             "keyed": ["Identical retries are credited again unless they carry the same request_key, which the server deduplicates.",
                       "x-retry: safe-with-request_key"]}}
ERRORS = {"train": {"timeout_a": "Error: upstream did not respond in time.", "timeout_b": "Error: connection reset before a reply was received.",
                    "reject": "Error: 429 Too Many Requests - the request was refused and not processed."},
          "dev": {"timeout_a": "Error: gateway timed out waiting for the ledger.", "timeout_b": "Error: the reply was lost in transit; the request may or may not have been booked.",
                  "reject": "Error: 503 Service Unavailable - the request was rejected and not processed."},
          "test": {"timeout_a": "Error: the request timed out.", "timeout_b": "Error: the reply was lost; the top-up may or may not have been booked.",
                   "reject": "Error: 503 Service Unavailable - the request was rejected and not processed."}}
TIMEOUTS = ("timeout_a", "timeout_b")
EFFECTS = ("idem", "nonidem", "keyed")
VERIFIERS = ("decisive", "stale", "none")
# semantic combinations never present in CF-SFT training (each component is seen in other combinations)
HELDOUT_COMBOS = {("nonidem", "stale", "absent"),    # escalate learned only via 'none'
                  ("keyed", "none", "absent"), ("keyed", "none", "present"),   # keyed retry learned only with a read available
                  ("nonidem", "none", "present")}    # reconcile learned only via 'stale'


def _camel(parts):
    return parts[0] + "".join(p.title() for p in parts[1:])


class FamilyT2:
    def __init__(self, idx):
        self.idx = idx
        self.split = next(s for s, r in SPLITS.items() if idx in r)
        rng = random.Random(f"semcf_t2:{idx}")
        self.domain = DOMAINS_T2[idx]
        self.noun = f"{self.domain}_{rng.choice(NOUN_SUFFIX)}"
        vs = VERBS[self.split]; v0 = rng.choice(vs); v1 = rng.choice([v for v in vs if v != v0])
        self.names = {"n0": f"{v0}_{self.noun}", "n1": _camel((v1 + "_" + self.noun + "_amount").split("_"))}
        self.acct = 7000 + rng.randrange(1000); self.amount = 5 * rng.randint(2, 30)
        self.decisive_form = rng.choice(["account", "activity"])
        self.read_names = {"decisive": f"get_{self.noun}" if self.decisive_form == "account" else f"list_{self.noun}_activity",
                           "stale": f"get_{self.noun}_statement", "none": f"get_{self.domain}_price_list"}
        self.reverse = f"reverse_{self.noun}_transaction"
        self.key = f"rk-{self.idx:02d}-{rng.randrange(10**6):06d}"

    def user(self):
        return f"Please add {self.amount} credits to {self.noun.replace('_', ' ')} {self.acct}."

    def w_args(self, effect):
        a = {"account_id": self.acct, "amount": self.amount}
        if effect == "keyed":
            a["request_key"] = self.key
        return a

    def tools(self, spec, strip_contract=False):
        eff = spec["effect"]; h = self.noun.replace("_", " ")
        contract = CONTRACT[spec["template_split"]][eff][spec["template"] % len(CONTRACT[spec["template_split"]][eff])]
        props = {"account_id": {"type": "integer", "description": f"The {h} id."}, "amount": {"type": "integer", "description": "Credits to add."}}
        if eff == "keyed":
            props["request_key"] = {"type": "string", "description": "Client reference for this request." if strip_contract
                                    else "Client idempotency key; reuse it when retrying the same request."}
        desc = f"Add credits to a {h}; returns a receipt with today's active transactions." + ("" if strip_contract else " " + contract)
        w = {"type": "function", "function": {"name": self.names[spec["name"]], "description": desc,
             "parameters": {"type": "object", "properties": props, "required": ["account_id", "amount"]}}}
        ap = {"account_id": {"type": "integer", "description": f"The {h} id."}}
        rd = {"decisive": ((f"Return the {h}'s live state: pending top-up and all transactions booked today." if self.decisive_form == "account"
                            else f"List every transaction booked on this {h} so far, including today's."), ap),
              "stale": (f"Return the {h}'s end-of-day statement from yesterday; activity booked today is not included.", ap),
              "none": (f"Return the current price list of the {self.domain}.", {})}[spec["verifier"]]
        out = [w, {"type": "function", "function": {"name": self.read_names[spec["verifier"]], "description": rd[0],
                                                     "parameters": {"type": "object", "properties": rd[1], "required": list(rd[1])}}}]
        if spec["compensation"] == "present":
            out.append({"type": "function", "function": {"name": self.reverse, "description": "Reverse one booked top-up transaction by its id.",
                        "parameters": {"type": "object", "properties": {"transaction_id": {"type": "string", "description": "Transaction id."}},
                                       "required": ["transaction_id"]}}})
        out.append({"type": "function", "function": {"name": ESC, "description": "Stop working on the request and tell the user it cannot be completed right now, with the reason.",
                    "parameters": {"type": "object", "properties": {"reason": {"type": "string", "description": "Why the request cannot be completed."}}, "required": ["reason"]}}})
        return out


def specs(split, errors=("timeout_a", "timeout_b", "reject", "none"), exclude_heldout_combos=False):
    out = []
    for i in SPLITS[split]:
        for eff, ver, comp, name, err in itertools.product(EFFECTS, VERIFIERS, ("absent", "present"), ("n0", "n1"), errors):
            if exclude_heldout_combos and (eff, ver, comp) in HELDOUT_COMBOS:
                continue
            outs = ("applied", "lost") if err in TIMEOUTS else ("lost",) if err == "reject" else ("-",)
            for o in outs:
                out.append({"template_id": "T2", "family": i, "split": split, "effect": eff, "verifier": ver, "compensation": comp,
                            "name": name, "error": err, "outcome": o, "template_split": split,
                            "template": (i + VERIFIERS.index(ver) + (name == "n1")) % 2,
                            "heldout_combo": (eff, ver, comp) in HELDOUT_COMBOS})
    return out


def oracle(s):
    """-> (feasible, label)."""
    if s["error"] == "none":
        return True, "do_once"
    if s["error"] == "reject":
        return True, "reissue"
    if s["effect"] == "idem":
        return True, "reissue"
    if s["effect"] == "keyed":
        return True, "reissue_same_key"
    if s["verifier"] == "decisive":
        return True, "check_then_branch"
    if s["compensation"] == "present":
        return True, "reconcile"
    return False, "escalate"


def needs_check(s):
    return s["error"] in TIMEOUTS and s["effect"] == "nonidem" and s["verifier"] == "decisive"


def min_calls(s):
    lab = oracle(s)[1]
    if lab == "check_then_branch":
        return 1 if s["outcome"] == "applied" else 2
    if lab == "reconcile":
        return 2 if s["outcome"] == "applied" else 1
    return 1


class T2Episode:
    MAX_CALLS = 8

    def __init__(self, spec, strip_contract=False):
        self.spec = spec; self.f = FamilyT2(spec["family"]); self.strip = strip_contract
        self.state = {"pending": None, "tx": [], "today": True}
        self.calls, self.done, self.escalated, self.final_text = [], False, False, None
        self.wname = self.f.names[spec["name"]]; self._n = 0
        if spec["error"] != "none":
            args = self.f.w_args(spec["effect"])
            if spec["outcome"] == "applied":
                self._apply(args)
            self.calls.append((self.wname, args, ERRORS[spec["template_split"]][spec["error"]]))
        self.n_forced = len(self.calls)

    def _apply(self, args):
        eff = self.spec["effect"]
        if eff == "idem":
            self.state["pending"] = args.get("amount"); return None
        if eff == "keyed" and args.get("request_key") and any(t["key"] == args["request_key"] and t["status"] == "active" for t in self.state["tx"]):
            return next(t for t in self.state["tx"] if t["key"] == args["request_key"])["id"]
        self._n += 1; tid = f"tx{self.f.idx:02d}{self._n:02d}"
        self.state["tx"].append({"id": tid, "amount": args.get("amount"), "status": "active", "key": args.get("request_key")})
        return tid

    def tools(self):
        return self.f.tools(self.spec, self.strip)

    def messages(self, system):
        m = [{"role": "system", "content": system}, {"role": "user", "content": self.f.user()}]
        for n, a, o in self.calls:
            m.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": n, "arguments": a}}]})
            m.append({"role": "tool", "content": o})
        return m

    def _active(self):
        return [{"transaction_id": t["id"], "amount": t["amount"]} for t in self.state["tx"] if t["status"] == "active"]

    def call(self, name, args):
        if self.done:
            return "Error: episode finished"
        f, s = self.f, self.spec; args = args if isinstance(args, dict) else {}
        tnames = {t["function"]["name"] for t in self.tools()}
        if name == ESC:
            self.escalated = self.done = True; o = "Escalated to the user."
        elif name not in tnames:
            o = f"Error: unknown function {name}"
        elif name == self.wname:
            if _int(args.get("account_id")) != f.acct:
                o = "Error: account not found"
            else:
                a = {"account_id": f.acct, "amount": _int(args.get("amount"))}
                if "request_key" in args:
                    a["request_key"] = args["request_key"]
                tid = self._apply(a)
                o = json.dumps({"pending_topup": self.state["pending"]} if s["effect"] == "idem" else
                               {"transaction_id": tid, "todays_active_transactions": self._active()})
        elif name == f.reverse:
            t = next((t for t in self.state["tx"] if t["id"] == args.get("transaction_id") and t["status"] == "active"), None)
            if t is None:
                o = "Error: no active transaction with that id"
            else:
                t["status"] = "reversed"; o = json.dumps({"reversed": t["id"], "todays_active_transactions": self._active()})
        else:
            o = self._read(s["verifier"], args)
        self.calls.append((name, args, o))
        if len(self.calls) - self.n_forced >= self.MAX_CALLS:
            self.done = True
        return o

    def _read(self, v, args):
        f = self.f
        if v == "none":
            return json.dumps({"prices": {"single": 4, "ten_pack": 35}})
        if _int(args.get("account_id")) != f.acct:
            return "Error: account not found"
        if v == "stale":
            return json.dumps({"account_id": f.acct, "statement_date": "yesterday", "transactions": [], "pending_topup": None})
        if f.decisive_form == "account":
            return json.dumps({"account_id": f.acct, "pending_topup": self.state["pending"], "transactions_today": self._active()})
        return json.dumps({"account_id": f.acct, "activity": self._active() if self.spec["effect"] != "idem" else
                           ([{"type": "pending_topup", "amount": self.state["pending"]}] if self.state["pending"] else [])})

    def finish(self, text=""):
        self.done = True; self.final_text = text

    def grade(self):
        s = self.spec; post = self.calls[self.n_forced:]; f = self.f
        reissues = sum(1 for n, *_ in post if n == self.wname)
        reads = {f.read_names[v] for v in VERIFIERS}
        checks = sum(1 for n, *_ in post if n in reads)
        act = [t for t in self.state["tx"] if t["status"] == "active"]
        if s["effect"] == "idem":
            goal = self.state["pending"] == f.amount and not self.state["tx"]
        else:
            goal = len(act) == 1 and act[0]["amount"] == f.amount
        feasible, lab = oracle(s)
        correct = (goal and not self.escalated) if feasible else (self.escalated and reissues == 0)
        cls = [_cls(n, a, self) for n, a, _ in post]
        first = cls[0] if cls else "finish"
        j = next((k for k, c in enumerate(cls) if c == "check"), None)
        after = None if j is None else (cls[j + 1] if j + 1 < len(cls) else "finish")
        return {"correct": bool(correct), "feasible": feasible, "oracle": lab, "goal_state": bool(goal),
                "duplicate": len(act) > 1 or (s["effect"] == "idem" and bool(self.state["tx"])), "escalated": self.escalated,
                "reissues": reissues, "checks": checks, "first": first, "after_check": after, "n_calls": len(post),
                "unnecessary_check": bool(checks and not needs_check(s)),
                "initial_correct": (first in initial_allowed(s)) if s["error"] != "none" else None,
                "post_check_correct": (after in post_check_allowed(s)) if (after is not None and s["error"] != "none") else None,
                "premature_escalation": bool(self.escalated and feasible), "min_calls": min_calls(s)}


def _int(x):
    try:
        return int(x)
    except (TypeError, ValueError):
        return None


def _cls(n, a, ep):
    if n == ep.wname:
        return "reissue_same_key" if (ep.spec["effect"] == "keyed" and isinstance(a, dict) and a.get("request_key") == ep.f.key) else "reissue"
    if n == ESC:
        return "escalate"
    if n == ep.f.reverse:
        return "reverse"
    if n in {ep.f.read_names[v] for v in VERIFIERS}:
        return "check"
    return "other"


def initial_allowed(s):
    lab = oracle(s)[1]
    return {"reissue": {"reissue", "check", "reissue_same_key"}, "reissue_same_key": {"reissue_same_key", "check"},
            "check_then_branch": {"check"} | ({"reissue"} if s["compensation"] == "present" else set()),
            "reconcile": {"reissue", "check"}, "escalate": {"escalate", "check"}}[lab]


def post_check_allowed(s):
    lab = oracle(s)[1]; ap = s["outcome"] == "applied"
    if lab == "check_then_branch":
        return {"finish"} if ap else {"reissue"}
    if lab == "reissue":
        return {"reissue", "reissue_same_key"} | ({"finish"} if ap and s["verifier"] == "decisive" else set())
    if lab == "reissue_same_key":
        return {"reissue_same_key"} | ({"finish"} if ap and s["verifier"] == "decisive" else set())
    return {"reissue"} if lab == "reconcile" else {"escalate"}


# ---------------------------------------------------------------- runner and metrics (cluster = family)
def run_policy(policy, sp, system=SYSTEM, strip_contract=False, max_steps=10):
    rows = []
    for s in sp:
        ep = T2Episode(s, strip_contract)
        for _ in range(max_steps):
            if ep.done:
                break
            a = policy.act(ep.messages(system), ep.tools())
            if "final_text" in a:
                ep.finish(a["final_text"]); break
            ep.call(a["name"], a.get("arguments", {}))
        rows.append({**s, **ep.grade(), "calls": [[n, a, o[:200]] for n, a, o in ep.calls]})
    return rows


FACTORS = ("family", "effect", "verifier", "compensation", "name", "error", "outcome")


def _key(r, drop):
    drop = (drop,) if isinstance(drop, str) else drop
    return tuple((k, r[k]) for k in FACTORS if k not in drop)


def _groups(rs, factor, n):
    g = {}
    for r in rs:
        g.setdefault(_key(r, factor), []).append(r)
    return [v for v in g.values() if len(v) == n]


def metrics(rows):
    by = {}
    fault = [r for r in rows if r["error"] != "none"]; tw = [r for r in fault if r["error"] in TIMEOUTS]
    M = lambda name, rs, fn: by.__setitem__(name, _clustered(rs, fn))  # noqa: E731
    M("decision_accuracy", fault, lambda rs: _mean([r["correct"] for r in rs]))
    M("initial_decision_correct", fault, lambda rs: _mean([r["initial_correct"] for r in rs]))
    M("check_rate_after_fault", fault, lambda rs: _mean([r["checks"] > 0 for r in rs]))
    M("post_check_interpretation", [r for r in fault if r["post_check_correct"] is not None], lambda rs: _mean([r["post_check_correct"] for r in rs]))
    M("outcome_twin_correct", tw, lambda rs: _mean([a["correct"] and b["correct"] for a, b in _groups(rs, "outcome", 2)]))
    M("feasible_end_state", [r for r in fault if r["feasible"]], lambda rs: _mean([r["goal_state"] and not r["escalated"] for r in rs]))
    M("infeasible_safe_escalation", [r for r in fault if not r["feasible"]], lambda rs: _mean([r["correct"] for r in rs]))
    M("reconcile_twin_correct", [r for r in tw if oracle(r)[1] == "reconcile"], lambda rs: _mean([a["correct"] and b["correct"] for a, b in _groups(rs, "outcome", 2)]))
    M("keyed_retry_correct", [r for r in tw if r["effect"] == "keyed"], lambda rs: _mean([r["correct"] for r in rs]))
    # A: name fixed, effect varied over all three effects, where the verifier cannot decide: every effect must be right
    # in BOTH outcome twins (6 episodes), so any effect-blind policy scores 0 (a lucky retry in the lost twin is not credit)
    a_rows = [r for r in tw if r["verifier"] != "decisive"]
    M("A_effect_sensitivity", a_rows, lambda rs: _mean([all(x["correct"] for x in g) for g in _groups(rs, ("effect", "outcome"), 6)]))
    M("B_name_consistency", fault, lambda rs: _mean([(a["first"], a["escalated"], a["reissues"] > 0) == (b["first"], b["escalated"], b["reissues"] > 0)
                                                     for a, b in _groups(rs, "name", 2)]))
    M("B_name_both_correct", fault, lambda rs: _mean([a["correct"] and b["correct"] for a, b in _groups(rs, "name", 2)]))
    M("premature_escalation", [r for r in fault if r["feasible"]], lambda rs: _mean([r["premature_escalation"] for r in rs]))
    M("duplicate_rate", rows, lambda rs: _mean([r["duplicate"] for r in rs]))
    M("unnecessary_check_rate", [r for r in rows if not needs_check(r)], lambda rs: _mean([r["unnecessary_check"] for r in rs]))
    M("excess_calls", rows, lambda rs: _mean([r["n_calls"] - r["min_calls"] for r in rs]))
    M("fault_free_success", [r for r in rows if r["error"] == "none"], lambda rs: _mean([r["correct"] for r in rs]))
    # the same headline metrics restricted to held-out semantic combinations / to seen combinations
    for tag, sel in (("hc", lambda r: r["heldout_combo"]), ("seen", lambda r: not r["heldout_combo"])):
        M(f"{tag}_decision_accuracy", [r for r in fault if sel(r)], lambda rs: _mean([r["correct"] for r in rs]))
        M(f"{tag}_outcome_twin_correct", [r for r in tw if sel(r)], lambda rs: _mean([a["correct"] and b["correct"] for a, b in _groups(rs, "outcome", 2)]))
    cells = {}
    for r in fault:
        cells.setdefault(f"{r['effect']}|{r['verifier']}|{r['compensation']}|{'timeout' if r['error'] in TIMEOUTS else r['error']}", []).append(r["correct"])
    by["cells"] = {k: round(_mean(v), 3) for k, v in sorted(cells.items())}
    return by


# ---------------------------------------------------------------- scripted policies (text-only access)
def _parse(msgs, tools):
    w = tools[0]["function"]; r = tools[1]["function"]
    rev = next((t["function"] for t in tools if t["function"]["name"].startswith("reverse_")), None)
    first = next((m["tool_calls"][0]["function"] for m in msgs if m["role"] == "assistant" and m.get("tool_calls")), None)
    amt = int(re.search(r"add (\d+) credits", msgs[1]["content"]).group(1))
    acct = int(re.search(r"(\d+)\.\s*$", msgs[1]["content"]).group(1))
    args = first["arguments"] if first else {"account_id": acct, "amount": amt}
    hist = [(m["tool_calls"][0]["function"], msgs[i + 1]["content"]) for i, m in enumerate(msgs) if m["role"] == "assistant" and m.get("tool_calls")]
    return w, r, rev, args, amt, hist


def _shows_applied(obs, amt):
    try:
        d = json.loads(obs)
    except Exception:  # noqa: BLE001
        return False
    if not isinstance(d, dict):
        return False
    if d.get("pending_topup") == amt:
        return True
    for k in ("transactions_today", "activity", "todays_active_transactions"):
        if any((x.get("amount") == amt) for x in d.get(k) or []):
            return True
    return False


class _T2Base:
    def act(self, msgs, tools):
        w, r, rev, args, amt, hist = _parse(msgs, tools)
        if not hist:
            return {"name": w["name"], "arguments": args}
        (lc, obs) = hist[-1]
        if lc["name"] == w["name"] and not obs.startswith("Error"):
            return self.after_success(w, rev, args, amt, hist, obs)
        if sum(1 for c, _ in hist if c["name"] == w["name"]) > 2:
            return {"name": ESC, "arguments": {"reason": "keeps failing"}}
        return self.recover(w, r, rev, args, amt, hist, lc, obs)

    def after_success(self, w, rev, args, amt, hist, obs):
        return {"final_text": "Done."}


class T2AlwaysRetry(_T2Base):
    def recover(self, w, r, rev, args, amt, hist, lc, obs):
        return {"name": w["name"], "arguments": args}


class T2AlwaysCheck(_T2Base):
    """The CAW rule: a clean rejection is retried directly; after any other write error read first, then finish if the
    read shows the top-up, else re-issue."""
    def recover(self, w, r, rev, args, amt, hist, lc, obs):
        if lc["name"] == w["name"] and "not processed" in obs:
            return {"name": w["name"], "arguments": args}
        if lc["name"] == w["name"]:
            return {"name": r["name"], "arguments": {k: v for k, v in args.items() if k in r["parameters"]["properties"]}}
        return {"final_text": "Done."} if _shows_applied(obs, amt) else {"name": w["name"], "arguments": args}


class T2ContractPolicy(_T2Base):
    """Deterministic contract-reading reference (parses the known T2 templates)."""
    def _effect(self, w):
        """Exact match against the known contract sentences (all splits); no contract -> assume unsafe (MCP default)."""
        d = w["description"]
        for eff in ("keyed", "nonidem", "idem"):                         # order matters: 'x-retry: safe' prefixes the others
            if any(t in d for sp in CONTRACT.values() for t in sp[eff]):
                return eff
        return "nonidem"

    def recover(self, w, r, rev, args, amt, hist, lc, obs):
        eff = self._effect(w); decisive = "live state" in r["description"] or "so far, including today" in r["description"]
        if lc["name"] == w["name"] and ("not processed" in obs or eff in ("idem", "keyed")):
            return {"name": w["name"], "arguments": args}
        if lc["name"] == w["name"]:
            if decisive:
                return {"name": r["name"], "arguments": {k: v for k, v in args.items() if k in r["parameters"]["properties"]}}
            if rev:
                return {"name": w["name"], "arguments": args}
            return {"name": ESC, "arguments": {"reason": "the top-up may already have been booked and repeating it is not safe"}}
        return {"final_text": "Done."} if _shows_applied(obs, amt) else {"name": w["name"], "arguments": args}

    def after_success(self, w, rev, args, amt, hist, obs):
        d = json.loads(obs); act = d.get("todays_active_transactions")
        if rev and act and len(act) > 1:
            return {"name": rev["name"], "arguments": {"transaction_id": act[0]["transaction_id"]}}
        return {"final_text": "Done."}


POLICIES_T2 = {"always_retry": T2AlwaysRetry, "always_check": T2AlwaysCheck, "contract_policy": T2ContractPolicy}


# ---------------------------------------------------------------- prompt-only baselines (selected on T2 DEV only)
from verification.semantic_cf import PROMPT_HINT  # noqa: E402

PROMPTS = {
    "P0": "",
    "P1": PROMPT_HINT,                                   # the frozen one-sentence hint of the T1 evaluation
    "P2": ("Recovery rules for a write whose outcome is unknown (for example a timeout): "
           "(1) if its documentation says repeating it is safe, or it takes an idempotency key that the original call used, "
           "repeat it with exactly the same arguments; "
           "(2) otherwise, if a read tool reports the live state, check it first: if the change is there, stop; if not, repeat the write; "
           "(3) otherwise, if a tool can reverse a duplicate, repeat the write and reverse any duplicate shown in the receipt; "
           "(4) otherwise call escalate_to_user without repeating the write. "
           "A clean rejection that says the request was not processed can be retried directly. Do not check when nothing is uncertain."),
}
PROMPTS["P3"] = PROMPTS["P2"] + (" Example: issue_gift_card timed out; its description says every call issues a new card; the only read "
                                 "tool returns yesterday's report and nothing can cancel a card, so the right action is escalate_to_user.")


def system_for(prompt_id):
    return SYSTEM + (" " + PROMPTS[prompt_id] if PROMPTS[prompt_id] else "")
