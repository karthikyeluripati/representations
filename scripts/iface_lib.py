"""Shared code for the tool-interface invariance study ("When Novelty Is Only Syntax").

Environment semantics: we use ToolSchemaSpace's (TSS) own scoring convention. A schema variant is only an
interface. Every call made in the variant is decoded to a native action, and the environment state is the
multiset of executed native actions (toolschema/simenv.py: "Scoring is state-based at episode end: the
multiset of executed native actions"). Two responses are behaviorally equivalent iff their decoded native
multisets are equal. The canonical *ordered* action sequence is tracked separately: it differs from the
state for call-order permutations.

Pinned TSS commit: d266b7879fd840292770f69e260450866b7b0512 (pass --tss PATH to a checkout of it).

Two reference-set regimes:
  R1 "alias catalog": one prompt per task. The catalog is the native tools plus equivalent alias tools,
     and every response in a group shares that prompt. This is the unit faithful RepExp actually uses:
     the covariance is computed within the group of rollouts for one prompt.
  R2 "cross-schema": the same task under different full-catalog schemas (TSS rename / arg_lower / merge,
     plus our argument renaming). Only relevant to novelty memories that persist across prompts (RepExp's
     persist_covariance option, or episodic or global embedding memories).
"""
import copy, hashlib, json, os, random, re, sys

TSS_COMMIT = "d266b7879fd840292770f69e260450866b7b0512"

VERB_SYN = {"set": "configure", "get": "fetch", "start": "begin", "stop": "halt", "add": "create",
            "delete": "erase", "remove": "drop", "turn": "switch", "activate": "trigger", "run": "execute",
            "announce": "broadcast", "cancel": "abort", "schedule": "plan", "lock": "secure",
            "unlock": "unsecure", "play": "playback", "pause": "suspend", "open": "unseal", "close": "seal",
            "send": "transmit", "create": "make", "update": "modify", "show": "display", "search": "lookup",
            "navigate": "route", "arm": "enable", "disarm": "disable", "toggle": "flip", "adjust": "tune",
            "change": "alter", "move": "relocate", "record": "capture", "clean": "wash", "find": "locate",
            "list": "enumerate", "enable": "allow", "disable": "deny", "mute": "silence", "rate": "score",
            "reply": "respond", "share": "distribute", "snooze": "postpone", "skip": "bypass",
            "book": "reserve", "invite": "summon", "log": "note", "measure": "gauge", "track": "follow",
            "preheat": "prewarm", "brew": "prepare", "bake": "roast", "wash": "launder", "dry": "dehumidify",
            "dock": "park", "map": "chart", "avoid": "evade", "save": "store", "load": "retrieve",
            "reset": "restore", "check": "inspect", "view": "see", "call": "phone", "connect": "link"}
ARG_SYN = {"room": "location", "position": "placement", "state": "mode", "speed": "rate", "level": "intensity",
           "direction": "heading", "angle": "tilt_degrees", "color": "hue", "temperature": "temp_value",
           "name": "label", "duration": "length_minutes", "volume": "loudness", "channel": "station",
           "time": "at_time", "date": "on_date", "title": "heading_text", "zone": "area", "seat": "seat_id",
           "device": "target_device", "mode": "setting", "program": "cycle", "brightness": "luminosity",
           "source": "input_source", "destination": "target", "query": "search_text", "id": "identifier"}
PREAMBLES = ["I'll take care of that now.", "Sure, making those changes now.", "On it.",
             "Okay, let me do that for you.", "Got it, handling this right away.",
             "No problem, applying the requested settings.", "Alright, carrying that out.",
             "Understood. Executing the request."]


def h(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, ensure_ascii=False).encode()).hexdigest()[:16]


def load_tss(tss):
    sys.path.insert(0, tss)
    d = os.path.join(tss, "benchmarks", "synthetic", "data")
    tools = json.load(open(os.path.join(d, "tools.json"), encoding="utf-8"))
    queries = [json.loads(l) for l in open(os.path.join(d, "queries.jsonl"), encoding="utf-8")]
    by_dom = {}
    for t in tools:
        by_dom.setdefault(t["group"], []).append(t)
    return by_dom, queries


def canon_call(name, args):
    return json.dumps([name, args], sort_keys=True, ensure_ascii=False)


def state_of(native_calls):
    """Environment state = multiset of executed native actions (TSS convention)."""
    return tuple(sorted(canon_call(n, a) for n, a in native_calls))


def seq_of(native_calls):
    return tuple(canon_call(n, a) for n, a in native_calls)


# ---------------------------------------------------------------- alias layer (R1)
def verb_alias(name, taken):
    parts = name.split("_")
    cand = "_".join([VERB_SYN[parts[0]]] + parts[1:]) if parts[0] in VERB_SYN else "perform_" + name
    while cand in taken:
        cand = cand + "_alt"
    return cand


def arg_alias(p, taken):
    c = ARG_SYN.get(p, p + "_value")
    while c in taken:
        c = c + "_x"
    return c


def props(t):
    return t["parameters"].get("properties", {})


def make_alias_tools(native, kinds=("A", "AB", "AC")):
    """Return (alias_tools, alias_map). alias_map: alias_name -> dict(native, argmap alias->native, nest)."""
    taken = {t["name"] for t in native}
    out, amap = [], {}
    for t in native:
        for k in kinds:
            if k == "AC" and len(props(t)) < 2:
                continue
            base = verb_alias(t["name"], taken)
            name = base if k == "A" else base + {"AB": "_fields", "AC": "_options"}[k]
            while name in taken:
                name += "_alt"
            taken.add(name)
            t2 = copy.deepcopy(t)
            t2["name"] = name
            t2["description"] = t["description"] + " (Equivalent alias of " + t["name"] + ".)"
            argmap, nest = {p: p for p in props(t)}, None
            if k == "AB":
                used, newp, argmap = set(props(t)), {}, {}
                for p, spec in props(t).items():
                    a = arg_alias(p, used - {p})
                    used.add(a)
                    newp[a] = spec
                    argmap[a] = p
                t2["parameters"]["properties"] = newp
                t2["parameters"]["required"] = [k2 for k2, v in argmap.items()
                                                if v in t["parameters"].get("required", [])]
            if k == "AC":
                nest = "options"
                t2["parameters"] = {"type": "object", "properties": {"options": {
                    "type": "object", "description": "All arguments of the operation.",
                    "properties": props(t), "required": t["parameters"].get("required", [])}},
                    "required": ["options"]}
            t2.pop("uid", None)
            out.append(t2)
            amap[name] = {"native": t["name"], "kind": k, "argmap": argmap, "nest": nest}
    return out, amap


def encode_alias(call, amap, kind):
    """Native call -> the alias call of the given kind (oracle encoder)."""
    n, a = call
    for an, e in amap.items():
        if e["native"] == n and e["kind"] == kind:
            inv = {v: k for k, v in e["argmap"].items()}
            if e["nest"]:
                return an, {e["nest"]: dict(a)}
            return an, {inv[p]: v for p, v in a.items()}
    return None


def decode_alias(calls, native_names, amap):
    """Model calls in the R1 alias catalog -> (native calls, flags). Off-schema calls are flagged."""
    out, flags = [], []
    for n, a in calls:
        if n in native_names:
            out.append((n, a))
        elif n in amap:
            e = amap[n]
            if e["nest"]:
                if set(a) != {e["nest"]} or not isinstance(a[e["nest"]], dict):
                    flags.append(("bad_nest", n)); continue
                out.append((e["native"], dict(a[e["nest"]])))
            else:
                bad = [k for k in a if k not in e["argmap"]]
                if bad:
                    flags.append(("bad_arg", n, bad)); continue
                out.append((e["native"], {e["argmap"][k]: v for k, v in a.items()}))
        else:
            flags.append(("unknown_tool", n))
    return out, flags


# ---------------------------------------------------------------- response text
def render(calls, preamble=None, key_order=None):
    """Qwen2.5 / Hermes tool-call text. key_order: None (as given) or 'reverse'."""
    parts = [preamble] if preamble else []
    for n, a in calls:
        if key_order == "reverse":
            a = dict(reversed(list(a.items())))
        parts.append("<tool_call>\n" + json.dumps({"name": n, "arguments": a}, ensure_ascii=False)
                     + "\n</tool_call>")
    return "\n".join(parts)


TC_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.S)


def parse(text):
    calls, bad = [], 0
    for m in TC_RE.finditer(text):
        try:
            j = json.loads(m.group(1))
            calls.append((j["name"], j.get("arguments", {}) or {}))
        except Exception:
            bad += 1
    return calls, bad


# ---------------------------------------------------------------- perturbations (different transitions)
def perturb_value(call, tool, rng):
    """Similar syntax, different transition: same tool, one enum/numeric/bool argument changed."""
    n, a = call
    ps = props(tool)
    opts = []
    for p, v in a.items():
        spec = ps.get(p, {})
        if spec.get("enum") and len(spec["enum"]) > 1:
            opts.append((p, [e for e in spec["enum"] if e != v]))
        elif spec.get("type") in ("integer", "number") and isinstance(v, (int, float)):
            lo, hi = spec.get("minimum", None), spec.get("maximum", None)
            c = [v + d for d in (1, 2, 5, 10, -1, -2, -5, -10)
                 if (lo is None or v + d >= lo) and (hi is None or v + d <= hi)]
            if spec.get("type") == "integer":
                c = [int(x) for x in c]
            if c:
                opts.append((p, c))
        elif spec.get("type") == "boolean" and isinstance(v, bool):
            opts.append((p, [not v]))
    if not opts:
        return None
    p, vals = opts[rng.randrange(len(opts))]
    a2 = dict(a)
    a2[p] = vals[rng.randrange(len(vals))]
    return n, a2


def tool_library(queries):
    """Valid native calls observed in gold data, per (domain, tool)."""
    lib = {}
    for q in queries:
        for c in q["gold_calls"]:
            lib.setdefault(q["domain"], {}).setdefault(c["name"], []).append(c.get("arguments", {}))
    return lib


def swap_tool(calls, i, domain, lib, rng):
    """Different syntax, different transition: replace call i with a valid call of a different tool."""
    others = [t for t in lib[domain] if t != calls[i][0]]
    if not others:
        return None
    t = others[rng.randrange(len(others))]
    a = lib[domain][t][rng.randrange(len(lib[domain][t]))]
    out = list(calls)
    out[i] = (t, dict(a))
    return out


def qwen_tools(tools):
    return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                              "parameters": t["parameters"]}} for t in tools]
