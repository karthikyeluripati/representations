"""Synthetic stateful APIs for the diversity study (docs/method_synthetic.md).

Each class = one domain with one record type and four functions whose semantics are fixed but whose surface varies
(verb, naming style, argument names, output format):
  create   NON-idempotent: appends a new record with a fresh id          (e.g. register_patient, bookSlot, open_ticket)
  set      idempotent write: sets one field of an existing record         (e.g. update_patient_ward)
  remove   idempotent write: deletes a record; repeating leaves state equal ("not found" error, nothing changes)
  list     read-only: returns all records
A task asks for one write. Training decisions are verified by executing this tiny state machine:
the chosen continuation reaches the goal state, and (for twin states) the opposite decision does not.
"""
from __future__ import annotations

import copy
import json
import random

DOMAINS = [
    ("clinic", "patient", ["name", "ward"]), ("library", "loan", ["title", "member"]), ("garden", "plot", ["crop", "owner"]),
    ("fleet", "vehicle", ["plate", "driver"]), ("kitchen", "order", ["dish", "table"]), ("hotel", "reservation", ["guest", "room"]),
    ("school", "enrollment", ["student", "course"]), ("museum", "exhibit", ["artwork", "hall"]), ("gym", "membership", ["member", "plan"]),
    ("shelter", "adoption", ["pet", "family"]), ("lab", "sample", ["specimen", "freezer"]), ("cinema", "screening", ["film", "screen"]),
    ("harbor", "berth", ["vessel", "pier"]), ("bakery", "batch", ["bread", "oven"]), ("farm", "harvest", ["field", "crop"]),
    ("pharmacy", "prescription", ["drug", "patient"]), ("theater", "booking", ["show", "seat"]), ("courier", "parcel", ["recipient", "address"]),
    ("archive", "record", ["document", "shelf"]), ("vet", "appointment", ["animal", "vet"]), ("print", "job", ["file", "printer"]),
    ("parking", "permit", ["plate", "zone"]), ("bank", "transfer", ["payee", "amount"]), ("event", "registration", ["attendee", "session"]),
    ("repair", "workorder", ["device", "technician"]), ("rental", "lease", ["tenant", "unit"]), ("orchestra", "rehearsal", ["piece", "room"]),
    ("airport", "gate_assignment", ["flight", "gate"]), ("charity", "donation", ["donor", "amount"]), ("studio", "session", ["artist", "studio"]),
    ("warehouse", "shipment", ["sku", "dock"]), ("conference", "talk", ["speaker", "track"]),
    # domains 32-63 (diversity-scaling study; appended so classes 0-31 are unchanged)
    ("zoo", "feeding", ["animal", "keeper"]), ("observatory", "observation", ["target", "telescope"]), ("ferry", "crossing", ["passenger", "deck"]),
    ("salon", "styling", ["client", "stylist"]), ("winery", "barrel", ["vintage", "cellar"]), ("brewery", "brew", ["recipe", "tank"]),
    ("dentist", "visit", ["patient", "chair"]), ("tailor", "fitting", ["garment", "customer"]), ("marina", "mooring", ["boat", "slip"]),
    ("hangar", "inspection", ["aircraft", "inspector"]), ("ranch", "herd_move", ["herd", "pasture"]), ("quarry", "blast", ["site", "crew"]),
    ("mint", "coinage", ["design", "press"]), ("radio", "segment", ["host", "slot"]), ("newsroom", "story", ["headline", "editor"]),
    ("laundry", "load", ["customer", "machine"]), ("pottery", "firing", ["piece", "kiln"]), ("arcade", "tournament", ["game", "player"]),
    ("chapel", "ceremony", ["couple", "officiant"]), ("aquarium", "tank_check", ["species", "tank"]), ("botany", "specimen_log", ["plant", "greenhouse"]),
    ("forge", "commission", ["blade", "smith"]), ("tram", "service", ["route", "car"]), ("ski", "lesson", ["student", "instructor"]),
    ("vineyard", "picking", ["row", "picker"]), ("customs", "declaration", ["importer", "goods"]), ("census", "survey", ["household", "enumerator"]),
    ("lighthouse", "watch", ["keeper", "night"]), ("apiary", "hive_check", ["hive", "beekeeper"]), ("atelier", "portrait", ["sitter", "painter"]),
    ("foundry", "casting", ["mold", "alloy"]), ("planetarium", "show_run", ["program", "dome"]),
]
CREATE_VERBS = ["create", "add", "register", "submit", "book", "open", "issue", "log", "post", "file", "schedule", "place"]
SET_VERBS = ["update", "set", "change", "modify", "assign"]
REMOVE_VERBS = ["delete", "remove", "cancel", "drop"]
LIST_VERBS = ["list", "get_all", "view", "show"]
VALUES = ["Aria", "Bram", "Cora", "Dex", "Elio", "Faye", "Gus", "Hana", "Ivo", "Juno", "Kai", "Lena", "Milo", "Nia", "Otto", "Pia",
          "Quin", "Rosa", "Sven", "Tara", "Uma", "Vito", "Wren", "Xena", "Yuri", "Zoe"]


def _name(style, verb, noun, rng):
    if style == "camel":
        parts = (verb + "_" + noun).split("_")
        return parts[0] + "".join(p.title() for p in parts[1:])
    return f"{verb}_{noun}"


class SynthClass:
    def __init__(self, idx, seed=0):
        rng = random.Random(f"synth:{idx}:{seed}")
        self.domain, self.noun, self.fields = DOMAINS[idx]
        self.style = rng.choice(["snake", "snake", "camel"])
        self.out = rng.choice(["json", "text"])
        cv, sv, rv, lv = rng.choice(CREATE_VERBS), rng.choice(SET_VERBS), rng.choice(REMOVE_VERBS), rng.choice(LIST_VERBS)
        self.cv = cv
        plural = self.noun + "s"
        self.fn = {"create": _name(self.style, cv, self.noun, rng), "set": _name(self.style, sv, self.noun + "_" + self.fields[1], rng),
                   "remove": _name(self.style, rv, self.noun, rng), "list": _name(self.style, lv, plural, rng)}
        self.id_key = rng.choice(["id", f"{self.noun}_id", "ref"])
        self.desc = {"create": f"{cv.replace('_', ' ').title()} a new {self.noun} in the {self.domain} system.",
                     "set": f"{sv.title()} the {self.fields[1]} of an existing {self.noun}.",
                     "remove": f"{rv.title()} an existing {self.noun}.",
                     "list": f"Return all {plural} currently in the {self.domain} system."}

    # ---- tools (OpenAI function schema) ----
    def tools(self):
        f0, f1 = self.fields
        s = lambda d: {"type": "string", "description": d}
        return [
            {"type": "function", "function": {"name": self.fn["create"], "description": self.desc["create"], "parameters": {
                "type": "object", "properties": {f0: s(f"The {f0}."), f1: s(f"The {f1}.")}, "required": [f0, f1]}}},
            {"type": "function", "function": {"name": self.fn["set"], "description": self.desc["set"], "parameters": {
                "type": "object", "properties": {self.id_key: {"type": "integer", "description": f"The {self.noun} id."},
                                                 f1: s(f"The new {f1}.")}, "required": [self.id_key, f1]}}},
            {"type": "function", "function": {"name": self.fn["remove"], "description": self.desc["remove"], "parameters": {
                "type": "object", "properties": {self.id_key: {"type": "integer", "description": f"The {self.noun} id."}},
                "required": [self.id_key]}}},
            {"type": "function", "function": {"name": self.fn["list"], "description": self.desc["list"],
                                              "parameters": {"type": "object", "properties": {}, "required": []}}},
            {"type": "function", "function": {"name": "escalate_to_user", "description": "Stop working on the request and tell the user it cannot be completed right now, with the reason.",
                                              "parameters": {"type": "object", "properties": {"reason": {"type": "string", "description": "Why the request cannot be completed."}}, "required": ["reason"]}}},
        ]

    # ---- execution ----
    def new_state(self, rng):
        n = rng.randint(1, 3)
        recs = {}
        for i in range(n):
            recs[100 + i] = {self.fields[0]: rng.choice(VALUES) + " " + rng.choice(VALUES), self.fields[1]: rng.choice(VALUES)}
        return {"records": recs, "next": 100 + n}

    def _fmt(self, obj):
        if self.out == "json":
            return json.dumps(obj)
        return "; ".join(f"{k}={v}" for k, v in obj.items()) if isinstance(obj, dict) else str(obj)

    def execute(self, st, name, args):
        f0, f1 = self.fields
        if name == self.fn["create"]:
            rid = st["next"]; st["next"] += 1
            st["records"][rid] = {f0: args[f0], f1: args[f1]}
            return self._fmt({self.id_key: rid, f0: args[f0], f1: args[f1]})
        if name == self.fn["set"]:
            rid = int(args[self.id_key])
            if rid not in st["records"]:
                return f"Error: {self.noun} {rid} not found"
            st["records"][rid][f1] = args[f1]
            return self._fmt({self.id_key: rid, **st["records"][rid]})
        if name == self.fn["remove"]:
            rid = int(args[self.id_key])
            if rid not in st["records"]:
                return f"Error: {self.noun} {rid} not found"
            del st["records"][rid]
            return self._fmt({"removed": rid})
        if name == self.fn["list"]:
            return self._fmt({f"{self.noun}s": [{self.id_key: k, **v} for k, v in sorted(st["records"].items())]})
        return f"Error: unknown function {name}"


def make_tasks(cls, n, seed=0):    # n tasks of each kind (create, set, remove)
    """n tasks per kind: create (non-idempotent target) and set/remove (idempotent targets)."""
    rng = random.Random(f"tasks:{cls.domain}:{seed}")
    out = []
    for j in range(n):
        st = cls.new_state(rng)
        f0, f1 = cls.fields
        v0, v1 = rng.choice(VALUES) + " " + rng.choice(VALUES), rng.choice(VALUES)
        while any(r[f0] == v0 for r in st["records"].values()):
            v0 = rng.choice(VALUES) + " " + rng.choice(VALUES)
        out.append({"kind": "create", "state": st, "user": f"Please {cls.cv} a {cls.noun.replace('_', ' ')} with {f0} '{v0}' and {f1} '{v1}'.",
                    "call": {"name": cls.fn["create"], "arguments": {f0: v0, f1: v1}}, "value": v0})
        rid = rng.choice(sorted(st["records"]))
        nv = rng.choice([v for v in VALUES if v != st["records"][rid][f1]])
        out.append({"kind": "set", "state": copy.deepcopy(st), "user": f"Change the {f1} of {cls.noun} {rid} to '{nv}'.",
                    "call": {"name": cls.fn["set"], "arguments": {cls.id_key: rid, f1: nv}}})
        out.append({"kind": "remove", "state": copy.deepcopy(st), "user": f"Remove {cls.noun} {rid}.",
                    "call": {"name": cls.fn["remove"], "arguments": {cls.id_key: rid}}})
    return out


def make_read_task(cls, seed=0):
    rng = random.Random(f"read:{cls.domain}:{seed}")
    return {"kind": "list", "state": cls.new_state(rng), "user": f"Show me all {cls.noun.replace('_', ' ')}s.",
            "call": {"name": cls.fn["list"], "arguments": {}}}


def goal_state(cls, task):
    st = copy.deepcopy(task["state"]); cls.execute(st, task["call"]["name"], task["call"]["arguments"]); return st


def run(cls, task, prefix_effects, actions):
    """prefix_effects: list of (applied: bool) for the faulted first call; actions executed afterwards."""
    st = copy.deepcopy(task["state"])
    if prefix_effects:
        cls.execute(st, task["call"]["name"], task["call"]["arguments"])
    esc = False
    for a in actions:
        if "final_text" in a:
            break
        if a["name"] == "escalate_to_user":
            esc = True; break
        cls.execute(st, a["name"], a["arguments"])
    return (not esc) and st == goal_state(cls, task)
