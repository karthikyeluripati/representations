"""Frozen causal-LM action scorer with KV-cached batched candidate scoring.

For a decision context (prompt) and a set of candidate actions, computes in one
prefix pass + one batched candidate pass:
  * log p(action tokens + end-of-turn | prompt)   (sum and mean over tokens)
  * a context embedding (last-layer hidden state at the final prompt position)
  * per-candidate embeddings (mean of last-layer hidden states over the
    candidate's tokens, i.e. RepExp-style mean pooling, contextualised on state)
and returns exact token accounting for the forward passes.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass

import numpy as np

SYSTEM_PROMPT = (
    "You are an agent acting in a text-based household environment. "
    "Each turn, reply with exactly one admissible action that makes progress toward the task."
)


@dataclass
class ScoreResult:
    logp_sum: np.ndarray      # [K]
    logp_mean: np.ndarray     # [K]
    n_action_tokens: np.ndarray  # [K]
    context_emb: np.ndarray   # [H]
    action_emb: np.ndarray    # [K, H]
    prompt_tokens: int
    candidate_tokens: int     # sum of (non-pad) candidate tokens processed
    wall_s: float
    prompt_tokens_computed: int = 0  # prompt tokens actually run through the model (rest reused from KV cache)


def static_user_prefix(intro: str, demo: str | None = None) -> str:
    """The part of the user message that is constant within an episode (demo + task intro)."""
    lines = []
    if demo:
        lines += ["Here is an example of how a similar task was solved:", demo.strip(), "",
                  "Now solve your task.", ""]
    lines += [intro.strip(), ""]
    return "\n".join(lines)


def build_prompt_messages(intro: str, history: list[tuple[str, str]], observation: str,
                          admissible: list[str], max_history: int, demo: str | None = None) -> list[dict]:
    """history: list of (action, resulting observation). Truncated to last `max_history` turns.
    demo: optional solved example transcript (train split) shown before the current task.
    The user content always starts with static_user_prefix(intro, demo)."""
    lines = [static_user_prefix(intro, demo)]
    hist = history[-max_history:] if max_history > 0 else []
    if len(history) > len(hist):
        lines.append(f"[{len(history) - len(hist)} earlier steps omitted]")
    for a, o in hist:
        lines.append(f"> {a}\n{o}")
    if hist:
        lines.append("")
    lines.append("Current observation: " + observation.strip() if history else "")
    lines.append("Admissible actions: " + ", ".join(admissible))
    lines.append("Next action:")
    return [{"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": "\n".join(l for l in lines if l is not None)}]


class LMScorer:
    def __init__(self, model_id: str, dtype: str = "float32", num_threads: int | None = None,
                 max_batch: int = 64, device: str = "auto", reuse_prefix: bool = True):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if num_threads:
            torch.set_num_threads(num_threads)
        self.torch = torch
        self.model_id = model_id
        self.tok = AutoTokenizer.from_pretrained(model_id)
        if device == "auto":
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        frac = os.environ.get("DARE_CUDA_MEM_FRACTION")  # set by scripts/run_shards.sh for multi-process runs
        if frac and device.startswith("cuda"):
            torch.cuda.set_per_process_memory_fraction(float(frac))
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=getattr(torch, dtype)).to(device)
        self.model.eval()
        self.max_batch = max_batch
        eot = self.tok.convert_tokens_to_ids("<|im_end|>")
        self.eot_id = eot if isinstance(eot, int) and eot >= 0 else self.tok.eos_token_id
        self.hidden_size = self.model.config.hidden_size
        # Static-prefix reuse: the KV cache of the episode-constant prefix (system prompt + demo + task
        # intro) is computed once, always from scratch in one pass, and the rest of the prompt is run on
        # top of it. The split point is a function of the prompt alone, so scores are a deterministic
        # function of (prompt, candidates) regardless of call history. NOTE: in bf16 this is not bitwise
        # equal to a single full pass (different reduction order), so reuse on/off are different policies;
        # `reuse_prefix` is part of the model config hash.
        self.reuse_prefix = reuse_prefix
        self._prefix_cache: dict = {}  # small FIFO: static-prefix token tuple -> KV cache

    def prompt_ids(self, messages: list[dict]) -> list[int]:
        return self.tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=True)

    def _static_boundary(self, messages: list[dict], pids: list[int], static_prefix: str) -> int:
        sys_msgs = [messages[0], {"role": "user", "content": static_prefix}]
        sids = self.tok.apply_chat_template(sys_msgs, add_generation_prompt=False, tokenize=True)
        n = min(len(sids), len(pids) - 1)
        a, b = np.asarray(sids[:n]), np.asarray(pids[:n])
        neq = np.nonzero(a != b)[0]
        lcp = int(neq[0]) if len(neq) else n
        return max(lcp - 2, 0)  # margin: never split inside a BPE merge at the static/dynamic seam

    def _get_prefix(self, ids: list[int]):
        import copy

        key = tuple(ids)
        if key not in self._prefix_cache:
            with self.torch.no_grad():
                o = self.model(self.torch.tensor([ids], device=self.device), use_cache=True, logits_to_keep=1)
            if len(self._prefix_cache) >= 4:
                self._prefix_cache.pop(next(iter(self._prefix_cache)))
            self._prefix_cache[key] = o.past_key_values
        return copy.deepcopy(self._prefix_cache[key])  # forward() extends caches in place

    def score(self, messages: list[dict], candidates: list[str], static_prefix: str | None = None) -> ScoreResult:
        torch = self.torch
        t0 = time.perf_counter()
        pids = self.prompt_ids(messages)
        P = len(pids)
        cand_ids = [self.tok.encode(c, add_special_tokens=False) + [self.eot_id] for c in candidates]
        with torch.no_grad():
            lcp = 0
            past = None
            if self.reuse_prefix and static_prefix is not None:
                lcp = self._static_boundary(messages, pids, static_prefix)
                if lcp > 0:
                    past = self._get_prefix(pids[:lcp])
            # logits_to_keep=1: never materialise the [P, vocab] logits matrix
            out = self.model(torch.tensor([pids[lcp:]], device=self.device), past_key_values=past, use_cache=True,
                             output_hidden_states=True, logits_to_keep=1)
            prefix_cache = out.past_key_values
            first_logp = torch.log_softmax(out.logits[0, -1].float(), -1).cpu()  # predicts 1st cand token
            ctx_emb = out.hidden_states[-1][0, -1].float().cpu().numpy()
            del out
            K = len(candidates)
            lsum, lmean, ntok, embs = np.zeros(K), np.zeros(K), np.zeros(K, dtype=int), np.zeros((K, self.hidden_size))
            for s in range(0, K, self.max_batch):
                chunk = cand_ids[s:s + self.max_batch]
                B, L = len(chunk), max(len(c) for c in chunk)
                ids = torch.full((B, L), self.tok.pad_token_id or 0, dtype=torch.long)
                mask = torch.zeros((B, L), dtype=torch.long)
                for i, c in enumerate(chunk):
                    ids[i, :len(c)] = torch.tensor(c)
                    mask[i, :len(c)] = 1
                cache = _repeat_cache(prefix_cache, B)
                ids, mask = ids.to(self.device), mask.to(self.device)
                attn = torch.cat([torch.ones((B, P), dtype=torch.long, device=self.device), mask], 1)
                pos = (P + torch.arange(L, device=self.device)).unsqueeze(0).expand(B, L)
                o2 = self.model(ids, past_key_values=cache, attention_mask=attn, position_ids=pos,
                                use_cache=True, output_hidden_states=True)
                logits = o2.logits  # [B,L,V]; position j predicts token j+1
                # log p(next) = logit[next] - logsumexp(logits): avoids a full [B,L,V] log_softmax copy
                nxt = torch.cat([ids[:, 1:], ids[:, :1]], 1)  # last column is a dummy, masked below
                lse = torch.logsumexp(logits.float(), -1)
                lp = (logits.gather(-1, nxt.unsqueeze(-1)).squeeze(-1).float() - lse).cpu()  # [B,L]
                h = o2.hidden_states[-1].float().cpu()
                del o2, logits
                for i, c in enumerate(chunk):
                    n = len(c)
                    toks = torch.tensor(c)
                    tok_lp = torch.empty(n)
                    tok_lp[0] = first_logp[toks[0]]
                    if n > 1:
                        tok_lp[1:] = lp[i, :n - 1]
                    k = s + i
                    lsum[k], lmean[k], ntok[k] = float(tok_lp.sum()), float(tok_lp.mean()), n
                    # embedding over the action's own tokens (exclude end-of-turn token)
                    embs[k] = h[i, :max(n - 1, 1)].mean(0).numpy()
        return ScoreResult(lsum, lmean, ntok, ctx_emb, embs, P, int(sum(len(c) for c in cand_ids)),
                           time.perf_counter() - t0, P - lcp)


def _repeat_cache(cache, B: int):
    """Return a batch-expanded *copy* of a single-sequence KV cache (prefix is reused across chunks)."""
    import copy

    c = copy.deepcopy(cache)
    if hasattr(c, "batch_repeat_interleave"):
        c.batch_repeat_interleave(B)
        return c
    raise RuntimeError("Unsupported cache type; upgrade transformers (need DynamicCache.batch_repeat_interleave)")
