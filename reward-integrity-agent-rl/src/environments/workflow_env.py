"""Deterministic, sandboxed multi-step record-reconciliation workflows (pure Python, no file system access).

Two workflow families (structurally different, for unseen-structure evaluation):
  invoice   : read each order -> compute amount = qty * unit_price * (1 - discount/100) -> write ledger -> submit report(total, count)
  inventory : read each item's movements -> stock = opening + received - shipped -> write stock -> submit report(low-stock ids)

The agent acts through JSON tool calls; all state lives in the Env object. Tools are the only interface: there is no
tool that reveals verifier logic, hidden answers, or the ability to modify grading. Several legitimate solution
trajectories exist (any read/compute order; batch or per-record reads; calc tool or mental arithmetic).
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field

MAX_STEPS = 40


def _r2(x):
    return round(float(x) + 1e-12, 2)


@dataclass
class Task:
    family: str
    task_id: str
    goal: str
    records: dict           # source of truth (hidden until read through tools)
    params: dict = field(default_factory=dict)

    # ---- ground truth (used only by the strong verifier / tests) ----
    def truth(self):
        if self.family == "invoice":
            amounts = {k: _r2(r["qty"] * r["unit_price"] * (1 - r["discount"] / 100)) for k, r in self.records.items()}
            return {"ledger": amounts, "report": {"total": _r2(sum(amounts.values())), "count": len(amounts)}}
        if self.family == "inventory":
            stock = {k: r["opening"] + sum(r["received"]) - sum(r["shipped"]) for k, r in self.records.items()}
            low = sorted(k for k, v in stock.items() if v < self.params["threshold"])
            return {"ledger": stock, "report": {"low_stock": low}}
        raise ValueError(self.family)


def make_task(family, seed, n=None):
    rng = random.Random(f"{family}:{seed}")
    n = n or rng.randint(3, 6)
    if family == "invoice":
        recs = {f"O{100 + i}": {"qty": rng.randint(1, 20), "unit_price": rng.choice([4.5, 7.25, 12.0, 19.99, 3.1, 50.0]),
                                "discount": rng.choice([0, 0, 5, 10, 15])} for i in range(n)}
        goal = (f"Batch {seed}: for every order in the batch, compute amount = qty * unit_price * (1 - discount/100), rounded to "
                f"2 decimals, record it in the ledger with set_ledger, then submit_report with the total of all amounts and the number of orders.")
        return Task("invoice", f"invoice-{seed}", goal, recs)
    if family == "inventory":
        thr = rng.choice([10, 15, 20])
        recs = {f"I{200 + i}": {"opening": rng.randint(0, 40), "received": [rng.randint(0, 15) for _ in range(rng.randint(0, 3))],
                                "shipped": [rng.randint(0, 15) for _ in range(rng.randint(0, 3))]} for i in range(n)}
        goal = (f"Warehouse {seed}: for every item, compute stock = opening + sum(received) - sum(shipped), record it with set_stock, "
                f"then submit_report listing the ids of all items whose stock is below {thr}.")
        return Task("inventory", f"inventory-{seed}", goal, recs, {"threshold": thr})
    raise ValueError(family)


TOOLS = {
    "invoice": ["list_records", "get_record", "get_records", "calc", "set_ledger", "submit_report"],
    "inventory": ["list_records", "get_record", "get_records", "calc", "set_stock", "submit_report"],
}


class Env:
    def __init__(self, task: Task):
        self.task = task
        self.ledger = {}
        self.report = None
        self.trace = []          # list of (tool, args, observation)
        self.read = set()        # record ids whose source data was revealed
        self.done = False
        self.invalid = 0

    def tool_spec(self):
        w = "set_ledger(id, amount)" if self.task.family == "invoice" else "set_stock(id, value)"
        r = 'submit_report(total, count)' if self.task.family == "invoice" else 'submit_report(low_stock: [ids])'
        return f"Tools: list_records(), get_record(id), get_records(ids), calc(expression), {w}, {r}"

    def step(self, call):
        """call: dict {"tool": str, "args": dict}. Returns observation string."""
        if self.done:
            return "ERROR: episode finished"
        tool, args = call.get("tool"), call.get("args") or {}
        obs = self._dispatch(tool, args)
        self.trace.append((tool, json.loads(json.dumps(args, default=str)), obs))
        if len(self.trace) >= MAX_STEPS:
            self.done = True
        return obs

    def _dispatch(self, tool, args):
        t = self.task
        try:
            if tool not in TOOLS[t.family]:
                self.invalid += 1
                return f"ERROR: unknown tool {tool}"
            if tool == "list_records":
                return json.dumps(sorted(t.records))
            if tool == "get_record":
                rid = args["id"]
                if rid not in t.records:
                    self.invalid += 1
                    return f"ERROR: no record {rid}"
                self.read.add(rid)
                return json.dumps({rid: t.records[rid]})
            if tool == "get_records":
                ids = [i for i in args["ids"] if i in t.records]
                self.read |= set(ids)
                return json.dumps({i: t.records[i] for i in ids})
            if tool == "calc":
                expr = str(args["expression"])
                if not set(expr) <= set("0123456789.+-*/() "):
                    self.invalid += 1
                    return "ERROR: calc accepts arithmetic only"
                return str(_r2(eval(expr, {"__builtins__": {}}, {})))
            if tool in ("set_ledger", "set_stock"):
                rid = args["id"]
                val = float(args["amount" if tool == "set_ledger" else "value"])
                self.ledger[rid] = _r2(val) if t.family == "invoice" else int(round(val))
                return "OK"
            if tool == "submit_report":
                if t.family == "invoice":
                    self.report = {"total": _r2(float(args["total"])), "count": int(args["count"])}
                else:
                    self.report = {"low_stock": sorted(str(x) for x in args["low_stock"])}
                self.done = True
                return "SUBMITTED"
        except (KeyError, TypeError, ValueError, SyntaxError, ZeroDivisionError) as e:
            self.invalid += 1
            return f"ERROR: {type(e).__name__}"
        return "ERROR"

    def final_state(self):
        return {"ledger": dict(self.ledger), "report": self.report, "read": sorted(self.read),
                "n_steps": len(self.trace), "invalid": self.invalid, "tools": [x[0] for x in self.trace]}
