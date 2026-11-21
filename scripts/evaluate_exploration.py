"""Stage 4 (run only if Gate C passes): do representation-novelty rewards change exploration decisions on
responses the policy actually samples?

For each benchmark task and catalog (R2_native = plain native tools; R1 = native + equivalent aliases),
sample N responses from the frozen policy at T=1.0 (RepExp's sampling temperature). Every response is
executed in SimEnv. Then:
  1. Prevalence: how often a group holds responses that are behaviorally identical but textually distinct.
  2. Reward: does the faithful RepExp bonus (group of n=8, as rollout.n=8) rank surface duplicates below
     responses with a unique state?
  3. Decisions (RepExp's inference-time use): choose k of N by each method. Report unique valid states,
     redundant picks, success@k, tokens and compute.

--generate (GPU) writes samples_<tag>.jsonl and feats_samples_<tag>.npz. --analyze (CPU) writes
explore_<tag>.json.
"""
import argparse, json, os, sys, time
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import iface_lib as L
import audit_novelty_rewards as A


def generate(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import construct_paired_transitions as C
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16).cuda().eval()
    by_dom, queries = L.load_tss(a.tss)
    recs = [json.loads(l) for l in open(a.data, encoding="utf-8")][: a.max_tasks]
    gold = {q["id"]: q for q in queries}
    out_rows, feats, texts = [], [], []
    t0, gen_tokens, fwd_tokens = time.time(), 0, 0
    torch.manual_seed(a.seed)
    for ri, rec in enumerate(recs):
        cat, _, _ = C.build_catalogs(rec["domain"], by_dom[rec["domain"]], rec["seed"])
        execs = C.build_executors(rec, by_dom)
        G = [(c["name"], c.get("arguments", {})) for c in gold[rec["task_id"]]["gold_calls"]]
        gstate = L.h(L.state_of(G))
        for pk in a.catalogs:
            msgs = [{"role": "system", "content": rec["system"]}, {"role": "user", "content": rec["goal"]}]
            ptxt = tok.apply_chat_template(msgs, tools=L.qwen_tools(cat[pk]["tools"]),
                                           add_generation_prompt=True, tokenize=False)
            pids = tok(ptxt, add_special_tokens=False, return_tensors="pt")["input_ids"].cuda()
            gens = []
            with torch.no_grad():
                for _ in range(a.n // a.gen_batch):
                    gens += model.generate(pids, do_sample=True, temperature=1.0, top_p=1.0, top_k=0,
                                           max_new_tokens=a.max_new_tokens, num_return_sequences=a.gen_batch,
                                           pad_token_id=tok.pad_token_id)[:, pids.shape[1]:].tolist()
            P = pids.shape[1]
            resp_ids = []
            for s in gens:
                eos_i = [i for i, x in enumerate(s) if x in (tok.convert_tokens_to_ids("<|im_end|>"),
                                                            tok.pad_token_id)]
                s = s[: eos_i[0] + 1] if eos_i else s
                resp_ids.append(s)
            gen_tokens += sum(map(len, resp_ids))
            # RepExp features: separate forward of prompt + response (as the reward-model worker does)
            lps = []
            for c0 in range(0, len(resp_ids), a.chunk):
                chunk = resp_ids[c0:c0 + a.chunk]
                with torch.no_grad():
                    T = max(map(len, chunk)) + P
                    ids = torch.full((len(chunk), T), tok.pad_token_id, dtype=torch.long)
                    att = torch.zeros_like(ids)
                    for i, s in enumerate(chunk):
                        ids[i, :P + len(s)] = torch.tensor(pids[0].tolist() + s); att[i, :P + len(s)] = 1
                    hs = model.model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
                    fwd_tokens += int(att.sum())
                    for i, s in enumerate(chunk):
                        feats.append(hs[i, P:P + len(s)].float().mean(0).cpu().numpy().astype(np.float32))
                        lg = model.lm_head(hs[i, P - 1:P + len(s) - 1]).float()
                        lp = torch.log_softmax(lg, -1).gather(1, torch.tensor(s, device="cuda")[:, None])[:, 0]
                        lps.append(float(lp.mean()))
                    del hs
            for i, s in enumerate(resp_ids):
                text = tok.decode(s, skip_special_tokens=True)
                calls, bad = L.parse(text)
                native, rej, flags = execs[pk].execute(calls)
                st = L.h(L.state_of(native))
                out_rows.append(dict(task_id=rec["task_id"], domain=rec["domain"], prompt_key=pk, i=i,
                                     text=text, n_tokens=len(s), logp_mean=lps[i], n_calls=len(calls),
                                     parse_errors=bad, rejected=rej, flags=[list(map(str, f)) for f in flags],
                                     valid=bool(calls) and not bad and not rej and not flags,
                                     state=st, seq=L.h(L.seq_of(native)), success=st == gstate,
                                     used_alias=any(c[0] not in execs[pk].native_names for c in calls)))
                texts.append(text)
        if ri % 10 == 0:
            print(f"[{ri+1}/{len(recs)}] {time.time()-t0:.0f}s gen_tokens={gen_tokens}", flush=True)
    from sentence_transformers import SentenceTransformer
    mini = SentenceTransformer(a.text_model, device="cuda").encode(texts, batch_size=128)
    with open(os.path.join(a.outdir, f"samples_{a.tag}.jsonl"), "w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    np.savez_compressed(os.path.join(a.outdir, f"feats_samples_{a.tag}.npz"), lm_ctx=np.stack(feats),
                        minilm=mini.astype(np.float32))
    json.dump(dict(wall_s=time.time() - t0, gen_tokens=gen_tokens, fwd_tokens=fwd_tokens,
                   n_samples=len(out_rows), model=a.model, n=a.n, temperature=1.0, seed=a.seed),
              open(os.path.join(a.outdir, f"gen_cost_{a.tag}.json"), "w"), indent=1)


def analyze(a):
    rows = [json.loads(l) for l in open(os.path.join(a.outdir, f"samples_{a.tag}.jsonl"), encoding="utf-8")]
    Z = np.load(os.path.join(a.outdir, f"feats_samples_{a.tag}.npz"))
    H, M = Z["lm_ctx"], Z["minilm"]
    groups = {}
    for i, r in enumerate(rows):
        groups.setdefault((r["task_id"], r["prompt_key"]), []).append(i)
    rng = np.random.default_rng(0)
    res = {"tag": a.tag, "by_catalog": {}}
    for pk in sorted({r["prompt_key"] for r in rows}):
        prev, acc, sel = {}, {}, {}
        stats = dict(valid=[], success=[], used_alias=[], n_tokens=[])
        for (t, p), I in groups.items():
            if p != pk:
                continue
            for i in I:
                for k in stats:
                    stats[k].append(float(rows[i][k]))
            # 1. prevalence + 2. reward, on the first 8 samples (RepExp group size)
            g = I[:8]
            vg = [i for i in g if rows[i]["valid"]]
            st = [rows[i]["state"] for i in vg]
            tx = [rows[i]["text"] for i in vg]
            pairs = [(x, y) for x in range(len(vg)) for y in range(x + 1, len(vg))]
            if pairs:
                sd = np.mean([st[x] == st[y] and tx[x] != tx[y] for x, y in pairs])
                ed = np.mean([tx[x] == tx[y] for x, y in pairs])
                prev.setdefault("surface_dup_pair_frac", {})[t] = [sd]
                prev.setdefault("exact_dup_pair_frac", {})[t] = [ed]
                prev.setdefault("n_distinct_states_over_texts", {})[t] = [len(set(st)) / max(len(set(tx)), 1)]
            if len(g) >= 3:
                b = A.repexp(H[g])
                stg = [rows[i]["state"] for i in g]
                txg = [rows[i]["text"] for i in g]
                from collections import Counter
                cs, ct = Counter(stg), Counter(txg)
                surf = np.array([cs[s] > 1 and ct[x] == 1 for s, x in zip(stg, txg)])  # surface duplicates
                uniq = np.array([cs[s] == 1 for s in stg])
                if surf.any() and uniq.any():
                    m = surf | uniq
                    acc.setdefault("repexp_surfacedup_vs_unique", {})[t] = [A.pairwise_dup_acc(b[m], surf[m])]
                    for nm, fn in (("nn_cos_minilm", lambda G: A.nn_cos(M[G])),
                                   ("tfidf_nn", lambda G: A.nn_cos(A.tfidf_feats([rows[i]["text"] for i in G])))):
                        bb = fn(g)
                        acc.setdefault(nm + "_surfacedup_vs_unique", {})[t] = [A.pairwise_dup_acc(bb[m], surf[m])]
            # 3. decisions: choose k of all N samples
            k = a.k
            if len(I) < k:
                continue
            keys_s = [rows[i]["state"] if rows[i]["valid"] else f"invalid{i}" for i in I]
            keys_q = [rows[i]["seq"] if rows[i]["valid"] else f"invalid{i}" for i in I]
            picks = {
                "random": [set(rng.choice(len(I), k, replace=False)) for _ in range(32)],
                "top_logp": [set(np.argsort(-np.array([rows[i]["logp_mean"] for i in I]))[:k])],
                "repexp_greedy": A.greedy_elliptical(H[I], k),
                "farthest_lm": A.greedy_generic(H[I], k),
                "farthest_minilm": A.greedy_generic(M[I], k),
                "farthest_tfidf": A.greedy_generic(A.tfidf_feats([rows[i]["text"] for i in I]), k),
                "canon_greedy": A.greedy_count(keys_q, k, rng),
                "state_greedy": A.greedy_count(keys_s, k, rng),
            }
            for nm, ps in picks.items():
                us, red, suc = [], [], []
                for p in ps:
                    vs = [rows[I[j]]["state"] for j in p if rows[I[j]]["valid"]]
                    us.append(len(set(vs)))
                    red.append(len(vs) - len(set(vs)))
                    suc.append(any(rows[I[j]]["success"] for j in p))
                for key, v in (("unique_valid_states", us), ("redundant_picks", red), ("success_at_k", suc)):
                    sel.setdefault(nm, {}).setdefault(key, {})[t] = [float(np.mean(v))]
            sel.setdefault("_ceiling", {}).setdefault("unique_valid_states", {})[t] = [
                float(min(k, len({rows[i]["state"] for i in I if rows[i]["valid"]})))]
        # paired difference: repexp_greedy - random, repexp_greedy - state_greedy
        diffs = {}
        for other in ("random", "farthest_tfidf", "canon_greedy", "state_greedy", "top_logp"):
            for key in ("unique_valid_states", "redundant_picks", "success_at_k"):
                d = {t: [sel["repexp_greedy"][key][t][0] - sel[other][key][t][0]]
                     for t in sel["repexp_greedy"][key] if t in sel[other][key]}
                diffs[f"repexp_minus_{other}:{key}"] = A.boot_ci(d)
        res["by_catalog"][pk] = dict(
            stats={k: float(np.mean(v)) for k, v in stats.items()},
            prevalence={k: A.boot_ci(v) for k, v in prev.items()},
            n_groups_with_surface_dups=len(acc.get("repexp_surfacedup_vs_unique", {})),
            reward_dup_recognition={k: A.boot_ci(v) for k, v in acc.items()},
            selection={nm: {key: A.boot_ci(v) for key, v in d.items()} for nm, d in sel.items()},
            paired_diffs=diffs)
    res["cost"] = json.load(open(os.path.join(a.outdir, f"gen_cost_{a.tag}.json")))
    json.dump(res, open(os.path.join(a.outdir, f"explore_{a.tag}.json"), "w"), indent=1)
    print(json.dumps(res, indent=1)[:6000])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tss", required=True)
    ap.add_argument("--data", default="results/interface_invariance/benchmark.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--catalogs", nargs="+", default=["R2_native", "R1"])
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--max-tasks", type=int, default=180)
    ap.add_argument("--max-new-tokens", type=int, default=384)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--chunk", type=int, default=4)
    ap.add_argument("--gen-batch", type=int, default=8)
    ap.add_argument("--tag", default="qwen7b")
    ap.add_argument("--outdir", default="results/interface_invariance")
    ap.add_argument("--generate", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.generate:
        generate(a)
    if a.analyze:
        analyze(a)


if __name__ == "__main__":
    main()
