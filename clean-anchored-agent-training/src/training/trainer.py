"""Minimal, explicit trainer shared by every arm (SFT / action-SFT / DPO, optional anchor, optional gradient operator).

Per optimizer step:
  g_P = grad of the primary loss (SFT or DPO) over a batch of `batch_size` examples (micro-batched)
  g_A = grad of the anchor loss over `anchor_batch_size` anchors (if the arm has anchors)
  update direction:  sum    -> g_P + lambda g_A
                     pcgrad -> PCGrad(g_P, lambda g_A)
                     agem   -> A-GEM(g_P, g_A)          (anchor used only as a constraint)
Logged every step: losses, DPO margins/accuracy, and cos(g_P, g_A) - for arms WITHOUT anchors g_A is computed on
`probe` anchors every `probe_every` steps and never applied (mechanism measurement, not training signal).
Every `eval_every` steps: decision accuracy (log pi(correct) > log pi(each alternative)) on dev corrective states and
dev twin anchors - the retention/correction training curves. Dev data is never trained on.
"""
from __future__ import annotations

import json
import math
import os
import random
import time

import torch

from src.training import objectives as O
from src.training.render import encode_action_example, encode_trajectory


# ---------------- encoding ----------------
class Encoder:
    def __init__(self, renderer, tools_fn, max_len):
        self.r, self.tools_fn, self.max_len = renderer, tools_fn, max_len
        self.dropped = 0

    def _tools_and_msgs(self, rec):
        tools = self.tools_fn(rec["task_id"])
        msgs = rec["msgs"]
        if "tool_rename_seed" in rec:              # PAFT-style SRP: consistent renaming in schema and calls
            from src.baselines.arms import srp_rename_map
            m = srp_rename_map([t["function"]["name"] for t in tools], random.Random(rec["tool_rename_seed"]))
            tools = [{**t, "function": {**t["function"], "name": m[t["function"]["name"]]}} for t in tools]
            msgs = json.loads(json.dumps(msgs))
            for x in msgs:
                for tc in x.get("tool_calls", []) or []:
                    tc["function"]["name"] = m.get(tc["function"]["name"], tc["function"]["name"])
        return tools, msgs

    def encode(self, rec):
        tools, msgs = self._tools_and_msgs(rec)
        t = rec["type"]
        if t == "traj":
            e = encode_trajectory(self.r, msgs, tools, self.max_len)
        elif t in ("action", "anchor"):
            e = encode_action_example(self.r, msgs, rec["action"], tools, self.max_len)
            if e is not None:
                e["suffix"] = True
        elif t == "pair":
            c = encode_action_example(self.r, msgs, rec["chosen"], tools, self.max_len)
            r = encode_action_example(self.r, msgs, rec["rejected"], tools, self.max_len)
            e = None if c is None or r is None else {"c": {**c, "suffix": True}, "r": {**r, "suffix": True}}
        else:
            raise ValueError(t)
        if e is None:
            self.dropped += 1
        return e

    def encode_all(self, recs):
        out = []
        for r in recs:
            e = self.encode(r)
            if e is not None:
                e["rec"] = r
                out.append(e)
        return out


def _collate(exs, pad_id, device):
    T = max(len(e["input_ids"]) for e in exs)
    ids = torch.full((len(exs), T), pad_id, dtype=torch.long)
    att = torch.zeros((len(exs), T), dtype=torch.long)
    msk = torch.zeros((len(exs), T), dtype=torch.long)
    for i, e in enumerate(exs):
        n = len(e["input_ids"])
        ids[i, :n] = torch.tensor(e["input_ids"]); att[i, :n] = 1; msk[i, :n] = torch.tensor(e["mask"])
    return ids.to(device), att.to(device), msk.to(device)


def forward_logits(model, exs, pad_id, device):
    """Returns (logits, ids, mask) aligned so that objectives.* can be applied.
    Suffix-target examples (one action after a long prefix) are LEFT-padded with explicit position ids so that every
    row's targets sit at the end, and only the last max(n_target)+1 logits are materialised (`logits_to_keep`).
    This is what makes 4B training on ~4-6k-token tool-schema prompts cheap; it is numerically the same objective."""
    if all(e.get("suffix") for e in exs):
        T = max(len(e["input_ids"]) for e in exs)
        k = max(e["n_target"] for e in exs) + 1
        ids = torch.full((len(exs), T), pad_id, dtype=torch.long)
        att = torch.zeros((len(exs), T), dtype=torch.long)
        msk = torch.zeros((len(exs), T), dtype=torch.long)
        for i, e in enumerate(exs):
            n = len(e["input_ids"])
            ids[i, T - n:] = torch.tensor(e["input_ids"]); att[i, T - n:] = 1; msk[i, T - n:] = torch.tensor(e["mask"])
        pos = (att.cumsum(-1) - 1).clamp(min=0)
        ids, att, msk, pos = ids.to(device), att.to(device), msk.to(device), pos.to(device)
        kw = {} if len(exs) == 1 else {"attention_mask": att}
        out = model(input_ids=ids, position_ids=pos, logits_to_keep=k, **kw)
        return out.logits, ids[:, -k:], msk[:, -k:]
    ids, att, msk = _collate(exs, pad_id, device)
    return model(input_ids=ids, attention_mask=att).logits, ids, msk


def seq_logps(model, exs, pad_id, device):
    lg, ids, m = forward_logits(model, exs, pad_id, device)
    return O.seq_logp(lg, ids, m)


# ---------------- trainer ----------------
class Trainer:
    def __init__(self, model, ref_ctx, pad_id, cfg, primary, anchors=None, probe=None, val=None, log_path=None, device="cpu"):
        """ref_ctx: zero-arg callable returning a context manager under which `model` computes the REFERENCE policy
        (peft: model.disable_adapter; tests: a context that swaps in a frozen copy)."""
        self.m, self.ref_ctx, self.pad, self.cfg, self.dev = model, ref_ctx, pad_id, cfg, device
        self.P, self.A, self.probe, self.val = primary, anchors or [], probe or [], val or {}
        self.params = [p for p in model.parameters() if p.requires_grad]
        self.opt = torch.optim.AdamW(self.params, lr=cfg["lr"], weight_decay=0.0)
        self.log_path = log_path
        self.rng = random.Random(cfg.get("seed", 0))
        self.steps = cfg["steps"]

    # ----- batches -----
    def _sampler(self, data, n):
        order = []
        while True:
            if len(order) < n:
                extra = list(range(len(data))); self.rng.shuffle(extra); order += extra
            yield [data[i] for i in order[:n]]
            order = order[n:]

    # ----- losses (one micro-batch) -----
    def primary_loss(self, exs):
        obj = self.cfg["objective"]
        if obj == "sft_traj":
            lg, ids, m = forward_logits(self.m, exs, self.pad, self.dev)
            return O.sft_loss(lg, ids, m), {}
        if obj == "sft_action":
            lg, ids, m = forward_logits(self.m, exs, self.pad, self.dev)
            return O.action_nll(lg, ids, m), {}
        if obj == "dpo":
            pc = seq_logps(self.m, [e["c"] for e in exs], self.pad, self.dev)
            pr = seq_logps(self.m, [e["r"] for e in exs], self.pad, self.dev)
            rc = torch.tensor([e["ref_c"] for e in exs], device=pc.device)
            rr = torch.tensor([e["ref_r"] for e in exs], device=pc.device)
            loss, margin = O.dpo_loss(pc, pr, rc, rr, self.cfg.get("beta", 0.1))
            if self.cfg.get("chosen_nll", 0) > 0:
                n = torch.tensor([e["c"]["n_target"] for e in exs], device=pc.device, dtype=pc.dtype)
                loss = loss + self.cfg["chosen_nll"] * (-(pc / n).mean())
            return loss, {"dpo_margin": float(margin.mean()), "dpo_acc": float((margin > 0).float().mean())}
        raise ValueError(obj)

    def anchor_loss(self, exs):
        lg, ids, m = forward_logits(self.m, exs, self.pad, self.dev)
        if self.cfg.get("anchor") == "kl":
            with torch.no_grad(), self.ref_ctx():
                rl, _, _ = forward_logits(self.m, exs, self.pad, self.dev)
            return O.anchor_kl(lg, rl, m), {}
        return O.action_nll(lg, ids, m), {}

    def _grad(self, loss_fn, batch, mb):
        self.opt.zero_grad(set_to_none=True)
        tot, info = 0.0, {}
        chunks = [batch[i:i + mb] for i in range(0, len(batch), mb)]
        for ch in chunks:
            l, inf = loss_fn(ch)
            (l / len(chunks)).backward()
            tot += float(l.detach()) / len(chunks)
            for k, v in inf.items():
                info[k] = info.get(k, 0.0) + v / len(chunks)
        g = O.flat([p.grad if p.grad is not None else torch.zeros_like(p) for p in self.params])
        return g, tot, info

    # ----- reference log-probs for DPO (computed once, under the reference policy) -----
    @torch.no_grad()
    def precompute_ref(self):
        if self.cfg["objective"] != "dpo":
            return
        bs = self.cfg.get("ref_batch", 8)
        with self.ref_ctx():
            for i in range(0, len(self.P), bs):
                ch = self.P[i:i + bs]
                rc = seq_logps(self.m, [e["c"] for e in ch], self.pad, self.dev)
                rr = seq_logps(self.m, [e["r"] for e in ch], self.pad, self.dev)
                for e, a, b in zip(ch, rc.tolist(), rr.tolist()):
                    e["ref_c"], e["ref_r"] = a, b

    # ----- decision accuracy on dev -----
    @torch.no_grad()
    def decision_eval(self, encoder):
        """Share of dev states where log pi(correct) exceeds log pi of every alternative (batched, no grad)."""
        res = {}
        for name, recs in self.val.items():
            items = []
            for r in recs:
                base = {k: r[k] for k in ("task_id", "msgs")}
                c = encoder.encode({**base, "type": "action", "action": r["correct"]})
                alts = [x for x in (encoder.encode({**base, "type": "action", "action": a}) for a in r["alternatives"]) if x is not None]
                if c is not None:
                    items.append([c] + alts)
            flat_ex = [x for it in items for x in it]
            lps = []
            for i in range(0, len(flat_ex), 8):
                lps += seq_logps(self.m, flat_ex[i:i + 8], self.pad, self.dev).tolist()
            ok, j = 0, 0
            for it in items:
                v = lps[j:j + len(it)]; j += len(it)
                ok += all(v[0] > x for x in v[1:])
            res[name] = ok / max(len(items), 1)
        return res

    def train(self, encoder=None):
        cfg = self.cfg
        mb = cfg.get("micro_batch", 1)
        warm = max(1, int(cfg.get("warmup_frac", 0.1) * self.steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            self.opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / self.steps))))
        P = self._sampler(self.P, cfg["batch_size"])
        A = self._sampler(self.A, cfg.get("anchor_batch_size", cfg["batch_size"])) if self.A else None
        Q = self._sampler(self.probe, cfg.get("anchor_batch_size", cfg["batch_size"])) if self.probe else None
        lam, mode = cfg.get("lambda", 1.0), cfg.get("grad_mode", "sum")
        log = open(self.log_path, "a") if self.log_path else None
        self.m.train()
        t0 = time.time()
        tokens = 0
        history = []
        if encoder is not None and self.val:
            self.m.eval(); rec = {"step": 0, "decision": self.decision_eval(encoder)}; self.m.train()
            history.append(rec)
            if log:
                log.write(json.dumps(rec) + "\n"); log.flush()
        for step in range(1, self.steps + 1):
            pb = next(P)
            tokens += sum(len(e["c"]["input_ids"]) + len(e["r"]["input_ids"]) if "c" in e else len(e["input_ids"]) for e in pb)
            gP, lP, info = self._grad(self.primary_loss, pb, mb)
            rec = {"step": step, "loss_primary": lP, **info}
            gA = None
            if A is not None:
                ab = next(A)
                tokens += sum(len(e["input_ids"]) for e in ab)
                gA, lA, _ = self._grad(self.anchor_loss, ab, mb)
                rec["loss_anchor"] = lA
            elif Q is not None and step % cfg.get("probe_every", 10) == 0:
                gQ, lQ, _ = self._grad(self.anchor_loss, next(Q), mb)
                rec["probe_cos"] = O.cosine(gP, gQ); rec["probe_anchor_loss"] = lQ
            if gA is not None:
                rec["cos_PA"] = O.cosine(gP, gA)
                if mode == "sum":
                    g = gP + lam * gA
                elif mode == "pcgrad":
                    g = O.pcgrad(gP, lam * gA)
                elif mode == "agem":
                    g = O.agem(gP, gA)
                else:
                    raise ValueError(mode)
            else:
                g = gP
            for p, gg in zip(self.params, O.unflat(g, self.params)):
                p.grad = gg.clone()
            if cfg.get("max_grad_norm"):
                torch.nn.utils.clip_grad_norm_(self.params, cfg["max_grad_norm"])
            self.opt.step(); sched.step()
            rec["lr"] = sched.get_last_lr()[0]; rec["tokens_seen"] = tokens; rec["elapsed_s"] = round(time.time() - t0, 2)
            if encoder is not None and self.val and step % cfg.get("eval_every", 50) == 0:
                self.m.eval(); rec["decision"] = self.decision_eval(encoder); self.m.train()
            history.append(rec)
            if log:
                log.write(json.dumps(rec) + "\n"); log.flush()
        if log:
            log.close()
        return history
