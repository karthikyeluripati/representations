"""Deterministic execution and replay of BFCL v4 multi-turn calls.

Fixes the v2 grader defect: ground-truth call strings with POSITIONAL arguments (e.g. `cd('documents')`) were parsed
keyword-only, so the positional value was silently dropped during replay. `parse_call` here returns both and
`execute` passes both.

Environment checkpoints are deep copies of the BFCL tool-class instances (plain Python objects, no RNG or I/O), so a
restored checkpoint is bit-identical to the original; `state_hash` verifies that.
"""
from __future__ import annotations

import ast
import copy
import hashlib
import importlib
import inspect
import json

from bfcl_eval.constants.executable_backend_config import CLASS_FILE_PATH_MAPPING, STATELESS_CLASSES


def parse_call(s: str):
    """'mv(source="a", destination="b")' or "cd('x')" -> (name, kwargs, positional_args)."""
    node = ast.parse(s.strip(), mode="eval").body
    if not isinstance(node, ast.Call):
        raise ValueError(f"not a call: {s!r}")
    name = node.func.attr if isinstance(node.func, ast.Attribute) else node.func.id
    kw = {k.arg: ast.literal_eval(k.value) for k in node.keywords}
    pos = [ast.literal_eval(a) for a in node.args]
    return name, kw, pos


def bind_args(inst, name, kw, pos):
    """Map positional args to parameter names (signature-aware) so every call can be expressed as a kwargs dict,
    which is the form a tool-calling model emits."""
    cls = method_owner(inst, name)
    if cls is None or not pos:
        return dict(kw)
    sig = inspect.signature(getattr(inst[cls], name))
    params = [p for p in sig.parameters.values() if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
    if len(pos) > len(params):
        raise ValueError(f"too many positional args for {name}")
    out = {p.name: v for p, v in zip(params, pos)}
    for k, v in kw.items():
        if k in out:
            raise ValueError(f"duplicate arg {k} for {name}")
        out[k] = v
    return out


def new_instances(entry):
    inst = {}
    cfg = entry["initial_config"]
    for cls in entry["involved_classes"]:
        mod = importlib.import_module(CLASS_FILE_PATH_MAPPING[cls])
        obj = getattr(mod, cls)()
        if cls not in STATELESS_CLASSES:
            obj._load_scenario(copy.deepcopy(cfg.get(cls, {})))
        inst[cls] = obj
    return inst


def method_owner(inst, name):
    if not isinstance(name, str) or name.startswith("_"):
        return None
    for cls, obj in inst.items():
        m = getattr(obj, name, None)
        if m is not None and inspect.ismethod(m):
            return cls
    return None


def to_obs(r) -> str:
    return r if isinstance(r, str) else json.dumps(r, default=str)


def execute(inst, name, args):
    """Execute one tool call with kwargs; returns the observation string (real BFCL errors are returned, not raised)."""
    cls = method_owner(inst, name)
    if cls is None:
        return f"Error: unknown function {name}"
    try:
        return to_obs(getattr(inst[cls], name)(**(args or {})))
    except Exception as e:  # noqa: BLE001 - BFCL APIs raise on bad arguments; the agent sees the message
        return f"Error: {type(e).__name__}: {e}"


def execute_str(inst, call: str):
    name, kw, pos = parse_call(call)
    return execute(inst, name, bind_args(inst, name, kw, pos))


def run_calls(inst, calls):
    return [execute_str(inst, c) for c in calls]


def public_state(obj):
    return {k: copy.deepcopy(v) for k, v in vars(obj).items() if not k.startswith("_")}


def _canon(x, seen=None):
    """Canonical JSON-able form of a tool-class state (handles mixed-type dict keys and parent/child cycles)."""
    seen = set() if seen is None else seen
    if isinstance(x, (str, int, float, bool, type(None))):
        return x
    if id(x) in seen:
        return "<cycle>"
    seen = seen | {id(x)}
    if isinstance(x, dict):
        return {repr(k): _canon(v, seen) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_canon(v, seen) for v in x]
    if isinstance(x, (set, frozenset)):
        return sorted(repr(v) for v in x)
    if hasattr(x, "__dict__"):
        return {"__cls__": type(x).__name__, **{k: _canon(v, seen) for k, v in vars(x).items() if not k.startswith("_")}}
    return repr(x)


def state_hash(inst) -> str:
    d = {c: _canon(public_state(o)) for c, o in sorted(inst.items())}
    return hashlib.sha256(json.dumps(d, sort_keys=True, default=str).encode()).hexdigest()[:16]


def checkpoint(inst):
    return copy.deepcopy(inst)


def states_equal(a, b) -> bool:
    return set(a) == set(b) and all(public_state(a[c]) == public_state(b[c]) for c in a)


def call_to_str(name, args) -> str:
    return f"{name}(" + ", ".join(f"{k}={v!r}" for k, v in args.items()) + ")"
