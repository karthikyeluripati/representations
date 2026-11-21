"""Build the paired-transition benchmark from ToolSchemaSpace's synthetic tasks (deterministic).

For each sampled task we build:
  - behaviors: G (gold), S1..S4 (similar syntax, different transition: one argument value changed) and
    D1..D3 (different syntax, different transition: one call replaced by another valid tool call).
  - R1 (alias catalog, one prompt). The group holds G, S1..S3, D1..D3 and one surface variant of G per
    family:
      identical (exact duplicate), A (alias tool name), AB (alias name + renamed args),
      AC (alias name + nested args), K (reversed JSON key order), O (reversed order of the calls),
      P (paraphrased natural-language preamble).
  - R1_shuf: the same responses, with the tool catalog shown in a different order (formatting control).
  - R2 (cross-schema): G under native, A (TSS rename), B (argument rename), C (TSS arg_lower), and
    M (TSS merge-flat dispatcher). The native schema also gets S1..S4 and D1..D3.
Every response is executed and verified (verify_tool_equivalence.verify_record); a task is kept only if all
checks pass.

Usage: python scripts/construct_paired_transitions.py --tss PATH --per-domain 15 --seed 0
"""
import argparse, json, os, random, sys
sys.path.insert(0, os.path.dirname(__file__))
import iface_lib as L

FAMILIES_R1 = ["identical", "A", "AB", "AC", "K", "O", "P"]
SCHEMAS_R2 = ["R2_native", "R2_A", "R2_B", "R2_C", "R2_M"]


def _argrename(native):
    tools, amap = [], {}
    for t in native:
        used, newp, argmap = set(L.props(t)), {}, {}
        for p, spec in L.props(t).items():
            a = L.arg_alias(p, used - {p})
            used.add(a); newp[a] = spec; argmap[a] = p
        t2 = {"name": t["name"], "description": t["description"],
              "parameters": {"type": "object", "properties": newp,
                             "required": [k for k, v in argmap.items()
                                          if v in t["parameters"].get("required", [])]}}
        tools.append(t2)
        amap[t["name"]] = {"native": t["name"], "kind": "B", "argmap": argmap, "nest": None}
    return tools, amap


def build_catalogs(domain, native, seed):
    """Deterministic catalogs. Returns {key: dict(tools, encode(native_calls)->calls, exec_kind, extra)}."""
    from toolschema.operators import compile_spec
    native = [{k: v for k, v in t.items() if k != "uid"} for t in native]
    alias_tools, amap = L.make_alias_tools(native)
    r1 = native + alias_tools
    rng = random.Random(f"{seed}:{domain}:R1")
    r1a = r1[:]; rng.shuffle(r1a)
    r1b = r1[:]; random.Random(f"{seed}:{domain}:R1shuf").shuffle(r1b)
    taken = {t["name"] for t in native}
    ren = {}
    for t in native:
        ren[t["name"]] = L.verb_alias(t["name"], taken | set(ren.values()))
    cA = compile_spec(native, {"ops": [{"op": "rename", "tool": n, "to": m} for n, m in ren.items()]})
    cC = compile_spec(native, {"ops": [{"op": "arg_lower", "tool": t["name"], "into": "options",
                                         "params": list(L.props(t))}
                                        for t in native if len(L.props(t)) >= 2]})
    cM = compile_spec(native, {"ops": [{"op": "merge", "tools": [t["name"] for t in native],
                                         "into": "execute", "encoding": "flat"}]})
    bt, bmap = _argrename(native)
    nat_names = {t["name"] for t in native}

    def enc_tss(c):
        return lambda calls: c.encode_episode([{"name": n, "arguments": a} for n, a in calls])

    def enc_b(calls):
        out = []
        for n, a in calls:
            inv = {v: k for k, v in bmap[n]["argmap"].items()}
            out.append((n, {inv[p]: v for p, v in a.items()}))
        return out

    ident = lambda calls: list(calls)
    cat = {
        "R1": dict(tools=r1a, encode=ident, exec_kind="alias", extra=dict(amap=amap, passthrough=nat_names)),
        "R1_shuf": dict(tools=r1b, encode=ident, exec_kind="alias",
                        extra=dict(amap=amap, passthrough=nat_names)),
        "R2_native": dict(tools=native, encode=ident, exec_kind="alias",
                          extra=dict(amap={}, passthrough=nat_names)),
        "R2_A": dict(tools=cA.tools, encode=enc_tss(cA), exec_kind="tss", extra=dict(compiled=cA)),
        "R2_B": dict(tools=bt, encode=enc_b, exec_kind="alias", extra=dict(amap=bmap, passthrough=set())),
        "R2_C": dict(tools=cC.tools, encode=enc_tss(cC), exec_kind="tss", extra=dict(compiled=cC)),
        "R2_M": dict(tools=cM.tools, encode=enc_tss(cM), exec_kind="tss", extra=dict(compiled=cM)),
    }
    for v in cat.values():
        v["catalog_hash"] = L.h(v["tools"])
    return cat, amap, native


def build_executors(rec, by_dom):
    import verify_tool_equivalence as V
    cat, amap, native = build_catalogs(rec["domain"], by_dom[rec["domain"]], rec["seed"])
    ex = {}
    for k, c in cat.items():
        assert c["catalog_hash"] == rec["catalog_hash"][k], f"catalog drift for {k}"
        e = V.Executor(c["exec_kind"], native, c["extra"])
        if c["exec_kind"] == "alias":
            e.native_names = c["extra"]["passthrough"]
        ex[k] = e
    return ex


def make_behaviors(q, tools_by_name, lib, rng):
    G = [(c["name"], c.get("arguments", {})) for c in q["gold_calls"]]
    seen = {L.state_of(G)}
    S, D = [], []
    for _ in range(200):
        if len(S) >= 4:
            break
        i = rng.randrange(len(G))
        p = L.perturb_value(G[i], tools_by_name[G[i][0]], rng)
        if p is None:
            continue
        b = list(G); b[i] = p
        if L.state_of(b) not in seen:
            seen.add(L.state_of(b)); S.append(b)
    for _ in range(200):
        if len(D) >= 3:
            break
        i = rng.randrange(len(G))
        b = L.swap_tool(G, i, q["domain"], lib, rng)
        if b is not None and L.state_of(b) not in seen:
            seen.add(L.state_of(b)); D.append(b)
    if len(S) < 4 or len(D) < 3:
        return None
    return G, S, D


def build_record(q, by_dom, lib, seed):
    import verify_tool_equivalence as V
    rng = random.Random(f"{seed}:{q['id']}")
    cat, amap, native = build_catalogs(q["domain"], by_dom[q["domain"]], seed)
    tbn = {t["name"]: t for t in native}
    beh = make_behaviors(q, tbn, lib, rng)
    if beh is None:
        return None, "no_perturbations"
    G, S, D = beh
    P0 = L.PREAMBLES[0]
    resp = []

    def add(pk, behavior, family, calls_variant, native_calls, pre=None, key_order=None):
        text = L.render(calls_variant, pre, key_order)
        resp.append(dict(rid=f"{pk}:{behavior}:{family}", prompt_key=pk, behavior=behavior, family=family,
                         text=text, native=native_calls,
                         final_state_hash=L.h(L.state_of(native_calls)),
                         seq_hash=L.h(L.seq_of(native_calls))))

    for pk in ("R1", "R1_shuf"):
        add(pk, "G", "canonical", G, G, P0)
        add(pk, "G", "identical", G, G, P0)
        for k in ("A", "AB", "AC"):
            enc = [L.encode_alias(c, amap, k) or c for c in G]
            if any(e[0] != c[0] for e, c in zip(enc, G)):
                add(pk, "G", k, enc, G, P0)
        if any(len(a) >= 2 for _, a in G):
            add(pk, "G", "K", G, G, P0, key_order="reverse")
        if len(G) >= 2 and L.seq_of(G[::-1]) != L.seq_of(G):
            add(pk, "G", "O", G[::-1], G[::-1], P0)
        add(pk, "G", "P", G, G, L.PREAMBLES[1 + rng.randrange(len(L.PREAMBLES) - 1)])
        for j, b in enumerate(S[:3]):
            add(pk, f"S{j+1}", "control_sim", b, b, P0)
        for j, b in enumerate(D):
            add(pk, f"D{j+1}", "control_diff", b, b, P0)
    for pk in SCHEMAS_R2:
        add(pk, "G", "canonical" if pk == "R2_native" else pk[3:], cat[pk]["encode"](G), G)
    for j, b in enumerate(S):
        add("R2_native", f"S{j+1}", "control_sim", b, b)
    for j, b in enumerate(D):
        add("R2_native", f"D{j+1}", "control_diff", b, b)
    rec = dict(task_id=q["id"], domain=q["domain"], type=q["type"], n_calls=q["n_calls"], seed=seed,
               goal=q["query"], system=q["system"],
               initial_state_hash=L.h([]),  # TSS synthetic env: empty native-action multiset at reset
               catalog_hash={k: v["catalog_hash"] for k, v in cat.items()},
               tss_commit=L.TSS_COMMIT, responses=resp)
    errs = V.verify_record(rec, build_executors(rec, by_dom))
    return (rec, None) if not errs else (None, "verify:" + errs[0])


def pairs_of(rec):
    """Rows in the format required by docs/benchmark_protocol.md (representation fields filled later)."""
    out = []
    for pk in ("R1", "R1_shuf"):
        base = next(r for r in rec["responses"] if r["rid"] == f"{pk}:G:canonical")
        for r in rec["responses"]:
            if r["prompt_key"] != pk or r is base:
                continue
            out.append(_pair(rec, pk, pk, base, r))
    base = next(r for r in rec["responses"] if r["rid"] == "R2_native:G:canonical")
    for r in rec["responses"]:
        if r["prompt_key"].startswith("R2") and r is not base:
            out.append(_pair(rec, "R2_native", r["prompt_key"], base, r))
    return out


def _pair(rec, pk0, pk1, a, b):
    eq = a["final_state_hash"] == b["final_state_hash"]
    ctrl = ("identical_syntax_identical_transition" if a["text"] == b["text"] and pk0 == pk1 else
            "different_syntax_identical_transition" if eq else
            "similar_syntax_different_transition" if b["family"] == "control_sim" else
            "different_syntax_different_transition")
    return dict(task_id=rec["task_id"], initial_state_hash=rec["initial_state_hash"], goal=rec["goal"],
                original_schema=f"{pk0}@{rec['catalog_hash'][pk0]}",
                transformed_schema=f"{pk1}@{rec['catalog_hash'][pk1]}",
                family=b["family"], control=ctrl,
                original_action=a["text"], transformed_action=b["text"],
                canonical_transition=b["native"], final_state_hash=b["final_state_hash"],
                semantic_equivalence_verified=eq, model_id=None, representation_config=None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tss", required=True)
    ap.add_argument("--per-domain", type=int, default=15)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/interface_invariance")
    a = ap.parse_args()
    by_dom, queries = L.load_tss(a.tss)
    lib = L.tool_library(queries)
    os.makedirs(a.out, exist_ok=True)
    rng = random.Random(a.seed)
    kept, drop = [], {}
    for dom in sorted(by_dom):
        qs = sorted([q for q in queries if q["domain"] == dom], key=lambda q: q["id"])
        rng.shuffle(qs)
        n = 0
        for q in qs:
            if n >= a.per_domain:
                break
            rec, why = build_record(q, by_dom, lib, a.seed)
            if rec is None:
                drop[why.split(":")[0]] = drop.get(why.split(":")[0], 0) + 1
                if why.startswith("verify"):
                    print("VERIFY FAIL", why)
                continue
            kept.append(rec); n += 1
    with open(os.path.join(a.out, "benchmark.jsonl"), "w", encoding="utf-8") as f:
        for r in kept:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "pairs.jsonl"), "w", encoding="utf-8") as f:
        for r in kept:
            for p in pairs_of(r):
                f.write(json.dumps(p, ensure_ascii=False) + "\n")
    fam = {}
    for r in kept:
        for x in r["responses"]:
            fam[(x["prompt_key"][:2], x["family"])] = fam.get((x["prompt_key"][:2], x["family"]), 0) + 1
    summ = dict(n_tasks=len(kept), dropped=drop, seed=a.seed, per_domain=a.per_domain,
                tss_commit=L.TSS_COMMIT, dataset_hash=L.h([r["task_id"] for r in kept]),
                responses_by_family={f"{k[0]}:{k[1]}": v for k, v in sorted(fam.items())})
    json.dump(summ, open(os.path.join(a.out, "benchmark_summary.json"), "w"), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
