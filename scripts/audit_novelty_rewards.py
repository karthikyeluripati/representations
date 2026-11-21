"""Stage 3: do representation-based novelty rewards treat behaviorally equivalent tool calls as novel?

--extract (GPU): for each benchmark response, under each prompt:
  * lm_ctx:   mean over response tokens of the frozen model's final-layer state (after the final norm),
              conditioned on the full chat prompt (system + tool catalog + query). This is exactly RepExp's
              feature (verl-recipe rep_exp/workers/elliptical_reward_model_worker.py: AutoModel
              last_hidden_state, mean over response positions, response includes the EOS token).
  * lm_noctx: the same, for the response text alone (no prompt).
  * logp_sum / logp_mean / n_tokens of the response under the model.
  * minilm:   sentence embedding of the response text (all-MiniLM-L6-v2).
--analyze (CPU): bonuses and metrics, written to results/interface_invariance/audit_<model>.json.

Bonus definitions:
  repexp     FAITHFUL formula (official worker, train_elliptical.sh settings): sparse random projection to
             32 dims (sklearn SparseRandomProjection, density auto, redrawn every call ->
             randomize_sparse_matrix=True; we average over 16 draws), float64, centered within the group,
             Sigma = lam*I + sum h h^T over the group (lam = 0.01), bonus = leverage h^T Sigma^-1 h.
  repexp_l*  same, with lam scaled relative to the group's mean squared norm (sensitivity only).
  lev_full   leverage on the unprojected features, lam = 0.01.
  nn_cos     1 - max cosine to any other group member (episodic-memory style embedding novelty).
  tfidf_nn   nn_cos on char 3-5-gram TF-IDF (non-neural text baseline).
  canon      1 / count of the identical decoded ordered native-action sequence in the group.
  state      1 / count of the identical final environment state in the group (ORACLE semantic novelty).
"""
import argparse, json, os, sys
import numpy as np

sys.path.insert(0, os.path.dirname(__file__))
import iface_lib as L

R1_FAMS = ["identical", "A", "AB", "AC", "K", "O", "P"]
R2_FAMS = ["A", "B", "C", "M"]


# ----------------------------------------------------------------------------- extraction
def extract(a):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(a.model)
    model = AutoModelForCausalLM.from_pretrained(a.model, torch_dtype=torch.bfloat16,
                                                 attn_implementation="sdpa").cuda().eval()
    import construct_paired_transitions as C
    by_dom, _ = L.load_tss(a.tss)
    recs = [json.loads(l) for l in open(a.data, encoding="utf-8")]
    eos = "<|im_end|>"
    out = {k: [] for k in ("rid", "task", "lm_ctx", "lm_noctx", "logp_sum", "logp_mean", "n_tokens",
                           "prompt_tokens")}
    texts = []

    @torch.no_grad()
    def run(prompt_ids, resp_ids_list):
        P = len(prompt_ids)
        seqs = [prompt_ids + r for r in resp_ids_list]
        T = max(map(len, seqs))
        ids = torch.full((len(seqs), T), tok.pad_token_id, dtype=torch.long)
        att = torch.zeros((len(seqs), T), dtype=torch.long)
        for i, s in enumerate(seqs):
            ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
        ids, att = ids.cuda(), att.cuda()
        hs = model.model(input_ids=ids, attention_mask=att).last_hidden_state
        feats, lps = [], []
        for i, r in enumerate(resp_ids_list):
            hr = hs[i, P:P + len(r)]
            feats.append(hr.float().mean(0).cpu().numpy())
            logits = model.lm_head(hs[i, P - 1:P + len(r) - 1]).float()
            lp = torch.log_softmax(logits, -1).gather(1, torch.tensor(r, device="cuda")[:, None])[:, 0]
            lps.append(lp.cpu().numpy())
        return feats, lps

    for ri, rec in enumerate(recs):
        cat, _, _ = C.build_catalogs(rec["domain"], by_dom[rec["domain"]], rec["seed"])
        for pk in cat:
            rs = [r for r in rec["responses"] if r["prompt_key"] == pk]
            if not rs:
                continue
            msgs = [{"role": "system", "content": rec["system"]}, {"role": "user", "content": rec["goal"]}]
            ptxt = tok.apply_chat_template(msgs, tools=L.qwen_tools(cat[pk]["tools"]),
                                           add_generation_prompt=True, tokenize=False)
            pids = tok(ptxt, add_special_tokens=False)["input_ids"]
            rids = [tok(r["text"] + eos, add_special_tokens=False)["input_ids"] for r in rs]
            f_ctx, lps = run(pids, rids)
            for r, ids_, fc, lp in zip(rs, rids, f_ctx, lps):
                out["rid"].append(r["rid"]); out["task"].append(rec["task_id"])
                out["lm_ctx"].append(fc.astype(np.float32))
                out["logp_sum"].append(float(lp.sum())); out["logp_mean"].append(float(lp.mean()))
                out["n_tokens"].append(len(ids_)); out["prompt_tokens"].append(len(pids))
                texts.append(r["text"])
            # context-free representation of the same response texts (batched, no prompt)
            with torch.no_grad():
                T = max(map(len, rids))
                ids = torch.full((len(rids), T), tok.pad_token_id, dtype=torch.long)
                att = torch.zeros_like(ids)
                for i, s in enumerate(rids):
                    ids[i, :len(s)] = torch.tensor(s); att[i, :len(s)] = 1
                hs = model.model(input_ids=ids.cuda(), attention_mask=att.cuda()).last_hidden_state
                for i, s in enumerate(rids):
                    out["lm_noctx"].append(hs[i, :len(s)].float().mean(0).cpu().numpy().astype(np.float32))
        if ri % 20 == 0:
            print(f"[{ri+1}/{len(recs)}] {rec['task_id']} prompt_tokens={len(pids)}", flush=True)
    from sentence_transformers import SentenceTransformer
    st = SentenceTransformer(a.text_model, device="cuda")
    mini = st.encode(texts, batch_size=128, normalize_embeddings=False, convert_to_numpy=True)
    np.savez_compressed(a.feats, rid=np.array(out["rid"]), task=np.array(out["task"]),
                        lm_ctx=np.stack(out["lm_ctx"]), lm_noctx=np.stack(out["lm_noctx"]),
                        minilm=mini.astype(np.float32), logp_sum=np.array(out["logp_sum"]),
                        logp_mean=np.array(out["logp_mean"]), n_tokens=np.array(out["n_tokens"]),
                        prompt_tokens=np.array(out["prompt_tokens"]), text=np.array(texts, dtype=object))
    print("saved", a.feats)


# ----------------------------------------------------------------------------- bonuses
def leverage(H, lam):
    Hc = H - H.mean(0, keepdims=True)
    K = Hc @ Hc.T
    n = len(H)
    return np.clip(np.diag(K @ np.linalg.inv(lam * np.eye(n) + K)), 0, None)  # Woodbury: = h^T(lam I+H^TH)^-1 h


_PROJ = {}


def sparse_proj(d, k, seed):
    key = (d, k, seed)
    if key not in _PROJ:
        from sklearn.random_projection import SparseRandomProjection
        p = SparseRandomProjection(k, density="auto", random_state=seed).fit(np.zeros((1, d)))
        _PROJ[key] = p.components_.toarray().T.astype(np.float64)
    return _PROJ[key]


def repexp(H, lam=0.01, k=32, draws=16, rel=False):
    H = H.astype(np.float64)
    out = np.zeros(len(H))
    for s in range(draws):
        Hp = H @ sparse_proj(H.shape[1], k, s)
        l = lam * np.mean(np.sum((Hp - Hp.mean(0)) ** 2, 1)) if rel else lam
        out += leverage(Hp, max(l, 1e-12))
    return out / draws


def nn_cos(H):
    X = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    S = X @ X.T
    np.fill_diagonal(S, -np.inf)
    return 1 - S.max(1)


def inv_count(keys):
    from collections import Counter
    c = Counter(keys)
    return np.array([1.0 / c[k] for k in keys])


def tfidf_feats(texts):
    from sklearn.feature_extraction.text import TfidfVectorizer
    v = TfidfVectorizer(analyzer="char", ngram_range=(3, 5), min_df=1)
    return v.fit_transform(texts).toarray()


METHODS = {
    "repexp": lambda F, G: repexp(F["lm_ctx"][G]),
    "repexp_lrel0.1": lambda F, G: repexp(F["lm_ctx"][G], 0.1, rel=True),
    "repexp_lrel1": lambda F, G: repexp(F["lm_ctx"][G], 1.0, rel=True),
    "lev_full": lambda F, G: leverage(F["lm_ctx"][G].astype(np.float64), 0.01),
    "nn_cos_lm": lambda F, G: nn_cos(F["lm_ctx"][G]),
    "repexp_noctx": lambda F, G: repexp(F["lm_noctx"][G]),
    "nn_cos_noctx": lambda F, G: nn_cos(F["lm_noctx"][G]),
    "repexp_minilm": lambda F, G: repexp(F["minilm"][G]),
    "nn_cos_minilm": lambda F, G: nn_cos(F["minilm"][G]),
    "tfidf_nn": lambda F, G: nn_cos(tfidf_feats([F["text"][i] for i in G])),
    "canon": lambda F, G: inv_count([F["seq"][i] for i in G]),
    "state": lambda F, G: inv_count([F["state"][i] for i in G]),
}


def greedy_elliptical(H, k, lam=0.01, proj_k=32, draws=16):
    """Faithful-style greedy selection (Sherman-Morrison), averaged over projection draws by majority."""
    picks = []
    for s in range(draws):
        Hp = H.astype(np.float64) @ sparse_proj(H.shape[1], proj_k, s)
        Hp = Hp - Hp.mean(0)
        inv = np.eye(Hp.shape[1]) / lam
        chosen, free = [], np.ones(len(Hp), bool)
        for _ in range(k):
            b = np.einsum("ij,jk,ik->i", Hp, inv, Hp)
            b[~free] = -np.inf
            i = int(np.argmax(b)); chosen.append(i); free[i] = False
            v = inv @ Hp[i]; inv -= np.outer(v, v) / (1 + Hp[i] @ v)
        picks.append(set(chosen))
    return picks


def greedy_generic(H, k, metric="cos", starts=None):
    """Farthest-point selection (max-min cosine distance) for embedding novelty baselines.
    Averaged over every possible starting element, so no response (e.g. the canonical one at index 0)
    is privileged."""
    X = H / (np.linalg.norm(H, axis=1, keepdims=True) + 1e-12)
    picks = []
    for s0 in (range(len(X)) if starts is None else starts):
        chosen = [s0]
        while len(chosen) < k:
            d = 1 - (X @ X[chosen].T).max(1)
            d[chosen] = -np.inf
            chosen.append(int(np.argmax(d)))
        picks.append(set(chosen))
    return picks


def greedy_count(keys, k, rng):
    order = rng.permutation(len(keys))
    chosen, seen = [], set()
    for i in order:
        if keys[i] not in seen:
            chosen.append(i); seen.add(keys[i])
    for i in order:
        if len(chosen) >= k:
            break
        if i not in chosen:
            chosen.append(i)
    return [set(chosen[:k])]


def boot_ci(vals_by_task, B=2000, seed=0):
    tasks = list(vals_by_task)
    if not tasks:
        return None
    m = np.array([np.mean(vals_by_task[t]) for t in tasks])
    rng = np.random.default_rng(seed)
    bs = [m[rng.integers(0, len(m), len(m))].mean() for _ in range(B)]
    return [float(m.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))]


def pairwise_dup_acc(b, dup_mask):
    """P(bonus of a duplicate member < bonus of a unique member); ties count 1/2. Oracle state = 1."""
    d, u = b[dup_mask], b[~dup_mask]
    if len(d) == 0 or len(u) == 0:
        return None
    diff = d[:, None] - u[None, :]
    scale = max(np.abs(b).max(), 1e-12)
    lt = (diff < -1e-9 * scale).mean(); eq = (np.abs(diff) <= 1e-9 * scale).mean()
    return float(lt + 0.5 * eq)


def analyze(a):
    Z = np.load(a.feats, allow_pickle=True)
    recs = {json.loads(l)["task_id"]: json.loads(l) for l in open(a.data, encoding="utf-8")}
    rid_task = list(zip(Z["task"].tolist(), Z["rid"].tolist()))
    pos = {(t, r): i for i, (t, r) in enumerate(rid_task)}
    F = {k: Z[k] for k in ("lm_ctx", "lm_noctx", "minilm", "logp_mean", "logp_sum", "n_tokens")}
    F["text"] = Z["text"].tolist()
    F["seq"], F["state"] = [None] * len(rid_task), [None] * len(rid_task)
    meta = {}
    for t, rec in recs.items():
        for r in rec["responses"]:
            i = pos[(t, r["rid"])]
            F["seq"][i], F["state"][i] = r["seq_hash"], r["final_state_hash"]
            meta[i] = r
    res = {"model": a.model, "n_tasks": len(recs), "R1": {}, "R2": {}, "controls": {}}
    rng = np.random.default_rng(0)
    methods = list(METHODS)
    # ---------------- R1: within-prompt groups of n=8 (RepExp rollout.n=8)
    for pk in ("R1", "R1_shuf"):
        acc = {m: {f: {} for f in R1_FAMS} for m in methods}
        ratio = {m: {f: {} for f in R1_FAMS} for m in methods}
        waste = {m: {f: {} for f in R1_FAMS} for m in ["repexp_greedy", "farthest_lm", "farthest_minilm",
                                                       "farthest_tfidf", "canon_greedy", "state_greedy",
                                                       "random", "top_logp"]}
        lens = {m: [] for m in methods}
        for t, rec in recs.items():
            base = [f"{pk}:G:canonical"] + [f"{pk}:S{j}:control_sim" for j in (1, 2, 3)] + \
                   [f"{pk}:D{j}:control_diff" for j in (1, 2, 3)]
            base_i = [pos[(t, r)] for r in base]
            for f in R1_FAMS:
                key = (t, f"{pk}:G:{f}")
                if key not in pos:
                    continue
                G = base_i + [pos[key]]
                dup = np.zeros(8, bool); dup[0] = dup[7] = True
                for m in methods:
                    b = METHODS[m](F, G)
                    v = pairwise_dup_acc(b, dup)
                    acc[m][f].setdefault(t, []).append(v)
                    ratio[m][f].setdefault(t, []).append(float(b[dup].mean() / max(b[~dup].mean(), 1e-12)))
                    if f == "identical":
                        pass
                    lens[m].append(float(np.corrcoef(b, F["n_tokens"][G])[0, 1]) if np.std(b) > 0 else 0.0)
                # selection of k = 7 (= number of distinct behaviors): wasted if both duplicates chosen
                k = 7
                sel = {
                    "repexp_greedy": greedy_elliptical(F["lm_ctx"][G], k),
                    "farthest_lm": greedy_generic(F["lm_ctx"][G], k),
                    "farthest_minilm": greedy_generic(F["minilm"][G], k),
                    "farthest_tfidf": greedy_generic(tfidf_feats([F["text"][i] for i in G]), k),
                    "canon_greedy": greedy_count([F["seq"][i] for i in G], k, rng),
                    "state_greedy": greedy_count([F["state"][i] for i in G], k, rng),
                    "random": [set(rng.choice(8, k, replace=False)) for _ in range(16)],
                    "top_logp": [set(np.argsort(-F["logp_mean"][G])[:k])],
                }
                for m, picks in sel.items():
                    waste[m][f].setdefault(t, []).append(float(np.mean([(0 in p and 7 in p) for p in picks])))
        res[pk] = {"dup_recognition_acc": {m: {f: boot_ci(acc[m][f]) for f in R1_FAMS} for m in methods},
                   "dup_bonus_ratio": {m: {f: boot_ci(ratio[m][f]) for f in R1_FAMS} for m in methods},
                   "wasted_slot_rate_k7": {m: {f: boot_ci(waste[m][f]) for f in R1_FAMS} for m in waste},
                   "corr_bonus_ntokens_mean": {m: float(np.nanmean(lens[m])) for m in methods}}
        # by-domain spread of the faithful repexp accuracy (transformation-family x domain clusters)
        dom = {}
        for f in R1_FAMS:
            for t, v in acc["repexp"][f].items():
                dom.setdefault(f, {}).setdefault(recs[t]["domain"], []).extend(v)
        res[pk]["repexp_acc_by_domain_minmax"] = {f: [float(min(np.mean(x) for x in d.values())),
                                                      float(max(np.mean(x) for x in d.values()))]
                                                  for f, d in dom.items()}
    # ---------------- controls: tool-order stability, permutation invariance, group size, logp of variants
    c = {}
    r1 = [pos[(t, r["rid"])] for t, rec in recs.items() for r in rec["responses"] if r["prompt_key"] == "R1"]
    r1s = [pos[(t, r["rid"].replace("R1:", "R1_shuf:", 1))] for t, rec in recs.items()
           for r in rec["responses"] if r["prompt_key"] == "R1"]
    a1, a2 = F["lm_ctx"][r1], F["lm_ctx"][r1s]
    cs = np.sum(a1 * a2, 1) / (np.linalg.norm(a1, axis=1) * np.linalg.norm(a2, axis=1))
    c["tool_order_cosine_same_response"] = [float(np.mean(cs)), float(np.min(cs))]
    t0 = next(iter(recs))
    G = [pos[(t0, r)] for r in [f"R1:G:canonical", "R1:S1:control_sim", "R1:S2:control_sim",
                                 "R1:S3:control_sim", "R1:D1:control_diff", "R1:D2:control_diff",
                                 "R1:D3:control_diff", "R1:G:P"]]
    perm = np.random.default_rng(1).permutation(8)
    b0 = repexp(F["lm_ctx"][G]); b1 = repexp(F["lm_ctx"][np.array(G)[perm]])
    c["repexp_permutation_invariant_maxdiff"] = float(np.abs(b0[perm] - b1).max())
    lp = {}
    for t, rec in recs.items():
        g = F["logp_mean"][pos[(t, "R1:G:canonical")]]
        for f in R1_FAMS[1:]:
            if (t, f"R1:G:{f}") in pos:
                lp.setdefault(f, {}).setdefault(t, []).append(float(F["logp_mean"][pos[(t, f"R1:G:{f}")]] - g))
    c["delta_logp_mean_variant_minus_canonical"] = {f: boot_ci(v) for f, v in lp.items()}
    gs = {m: {} for m in ("repexp", "nn_cos_lm", "nn_cos_minilm", "tfidf_nn")}
    for t, rec in recs.items():
        for f in ("A", "P", "K"):
            if (t, f"R1:G:{f}") not in pos:
                continue
            G4 = [pos[(t, r)] for r in ["R1:G:canonical", "R1:S1:control_sim", "R1:D1:control_diff",
                                         f"R1:G:{f}"]]
            dup = np.array([1, 0, 0, 1], bool)
            for m in gs:
                gs[m].setdefault(f, {}).setdefault(t, []).append(pairwise_dup_acc(METHODS[m](F, G4), dup))
    c["group_size4_dup_acc"] = {m: {f: boot_ci(v) for f, v in d.items()} for m, d in gs.items()}
    # does repexp grade semantic distance at all? within-group bonus dispersion (CV) in the identical-free group
    cv = []
    for t, rec in recs.items():
        G = [pos[(t, r)] for r in ["R1:G:canonical", "R1:S1:control_sim", "R1:S2:control_sim",
                                    "R1:S3:control_sim", "R1:D1:control_diff", "R1:D2:control_diff",
                                    "R1:D3:control_diff"]]
        b = repexp(F["lm_ctx"][G])
        cv.append([float(b.mean()), float(b.std() / b.mean())])
    cv = np.array(cv)
    c["repexp_7distinct_mean_bonus_and_cv"] = [float(cv[:, 0].mean()), float(cv[:, 1].mean())]
    sq = []
    for t in recs:
        G = [pos[(t, r)] for r in ["R1:G:canonical", "R1:G:P"]]
        for s in range(4):
            Hp = F["lm_ctx"][G].astype(np.float64) @ sparse_proj(F["lm_ctx"].shape[1], 32, s)
            sq.append(float(np.sum((Hp[0] - Hp[1]) ** 2)))
    c["repexp_proj_sqdist_G_vs_P_median_vs_lam"] = [float(np.median(sq)), 0.01]
    res["controls"] = c
    # ---------------- R2: cross-schema persistent memory
    r2 = {m: {f: {} for f in R2_FAMS} for m in ("repexp_mem", "nn_cos_lm", "nn_cos_minilm", "tfidf_nn",
                                                 "canon", "state")}
    dist = {f: {} for f in R2_FAMS}
    for t, rec in recs.items():
        mem = [pos[(t, r)] for r in ["R2_native:G:canonical"] + [f"R2_native:S{j}:control_sim" for j in (1, 2, 3)]
               + [f"R2_native:D{j}:control_diff" for j in (1, 2, 3)]]
        novel = pos[(t, "R2_native:S4:control_sim")]
        for f in R2_FAMS:
            key = (t, f"R2_{f}:G:{f}")
            if key not in pos:
                continue
            dupi = pos[key]
            for m in r2:
                if m == "repexp_mem":
                    vals = []
                    for s in range(16):
                        P = sparse_proj(F["lm_ctx"].shape[1], 32, s)
                        M = F["lm_ctx"][mem].astype(np.float64) @ P
                        mu = M.mean(0); Mc = M - mu
                        inv = np.linalg.inv(0.01 * np.eye(32) + Mc.T @ Mc)
                        q = lambda i: float(((F["lm_ctx"][i] @ P - mu) @ inv @ (F["lm_ctx"][i] @ P - mu)))
                        vals.append(q(dupi) < q(novel))
                    v = float(np.mean(vals))
                elif m in ("canon", "state"):
                    keys = F["seq" if m == "canon" else "state"]
                    v = float((keys[dupi] in {keys[i] for i in mem}) and (keys[novel] not in {keys[i] for i in mem}))
                else:
                    E = (F["lm_ctx"] if m == "nn_cos_lm" else F["minilm"]) if m != "tfidf_nn" else None
                    if E is None:
                        X = tfidf_feats([F["text"][i] for i in mem + [dupi, novel]])
                    else:
                        X = E[mem + [dupi, novel]]
                    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
                    nov = 1 - (X[-2:] @ X[:-2].T).max(1)
                    v = float(nov[0] < nov[1]) + 0.5 * float(nov[0] == nov[1])
                r2[m][f].setdefault(t, []).append(v)
            X = F["lm_ctx"]
            dG = np.linalg.norm(X[dupi] - X[mem[0]])
            dS = np.mean([np.linalg.norm(X[i] - X[mem[0]]) for i in mem[1:4]])
            dist[f].setdefault(t, []).append(float(dG / dS))
    res["R2"] = {"dup_scored_less_novel_than_new_behavior": {m: {f: boot_ci(v) for f, v in d.items()}
                                                              for m, d in r2.items()},
                 "lm_dist_ratio_Gvariant_vs_Snative": {f: boot_ci(v) for f, v in dist.items()}}
    json.dump(res, open(a.out, "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("controls",)}, indent=1))
    for pk in ("R1",):
        print("dup_recognition_acc", pk)
        for m in methods:
            print(f"  {m:16s}", "  ".join(f"{f}:{res[pk]['dup_recognition_acc'][m][f][0]:.2f}"
                                           for f in R1_FAMS if res[pk]['dup_recognition_acc'][m][f]))
        print("wasted_slot_rate_k7", pk)
        for m, d in res[pk]["wasted_slot_rate_k7"].items():
            print(f"  {m:16s}", "  ".join(f"{f}:{d[f][0]:.2f}" for f in R1_FAMS if d[f]))
    print("R2 dup<new", json.dumps({m: {f: round(v[0], 2) for f, v in d.items() if v}
                                     for m, d in res["R2"]["dup_scored_less_novel_than_new_behavior"].items()}))
    print("R2 dist ratio", {f: [round(x, 2) for x in v] for f, v in res["R2"]["lm_dist_ratio_Gvariant_vs_Snative"].items() if v})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tss", required=True)
    ap.add_argument("--data", default="results/interface_invariance/benchmark.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--text-model", default="sentence-transformers/all-MiniLM-L6-v2")
    ap.add_argument("--feats", required=True)
    ap.add_argument("--out", default=None)
    ap.add_argument("--extract", action="store_true")
    ap.add_argument("--analyze", action="store_true")
    a = ap.parse_args()
    if a.extract:
        extract(a)
    if a.analyze:
        analyze(a)


if __name__ == "__main__":
    main()
