"""Training objectives (pure functions of logits / log-probs) and gradient-combination operators.

Notation: s = decision state (rendered prefix), a+ / a- = verified corrective / failed action, (s~, a~) = anchor state
and the action the policy must keep. log pi(a|s) is summed over the action's tokens only.

  SFT (whole trajectory)      L = mean_tokens  -log pi(y_t | y_<t)  over assistant tokens with loss=True
  Action-only SFT (B5)        L = mean_pairs   -log pi(a+|s) / |a+|
  DPO (B6 / CSO-style)        L = -log sigma( beta [ (log pi(a+|s) - log pi_ref(a+|s)) - (log pi(a-|s) - log pi_ref(a-|s)) ] )
  Anchor, NLL                 R = mean_anchors -log pi(a~|s~) / |a~|
  Anchor, KL                  R = mean_anchor_tokens  KL( pi_ref(.|s~, a~_<t) || pi(.|s~, a~_<t) )
  CACRT                       L = DPO + lambda * R_twin          (twin anchors: see src/counterfactuals/construct.py)

Gradient operators (B8 and the adaptive ablation): PCGrad (Yu et al. 2020) and A-GEM (Chaudhry et al. 2019).
They are standard methods, implemented here as baselines, not claimed as contributions.
"""
from __future__ import annotations

import torch
import torch.nn.functional as F


def token_logps(logits, input_ids):
    """log p(x_t | x_<t) for t >= 1; returns [B, T-1]."""
    lp = torch.log_softmax(logits[:, :-1].float(), dim=-1)
    return lp.gather(-1, input_ids[:, 1:].unsqueeze(-1)).squeeze(-1)


def masked_sum(x, mask):
    return (x * mask).sum(-1)


def seq_logp(logits, input_ids, mask):
    """Sum of target-token log-probs per sequence. mask: [B, T] over input positions (1 = target token)."""
    return masked_sum(token_logps(logits, input_ids), mask[:, 1:].float())


def sft_loss(logits, input_ids, mask):
    m = mask[:, 1:].float()
    return -(token_logps(logits, input_ids) * m).sum() / m.sum().clamp(min=1)


def action_nll(logits, input_ids, mask):
    """Per-example length-normalised NLL, averaged over examples (each decision counts equally)."""
    m = mask[:, 1:].float()
    per = -(token_logps(logits, input_ids) * m).sum(-1) / m.sum(-1).clamp(min=1)
    return per.mean()


def dpo_loss(pi_c, pi_r, ref_c, ref_r, beta):
    margin = beta * ((pi_c - ref_c) - (pi_r - ref_r))
    return -F.logsigmoid(margin).mean(), margin.detach()


def anchor_kl(logits, ref_logits, mask):
    """Forward KL(ref || pi) at the anchor action's target positions, averaged over those tokens."""
    m = mask[:, 1:].float()
    lp = torch.log_softmax(logits[:, :-1].float(), -1)
    lq = torch.log_softmax(ref_logits[:, :-1].float(), -1)
    kl = (lq.exp() * (lq - lp)).sum(-1)
    return (kl * m).sum() / m.sum().clamp(min=1)


# ---------------- gradient operators ----------------
def flat(grads):
    return torch.cat([g.reshape(-1) for g in grads])


def unflat(vec, like):
    out, i = [], 0
    for p in like:
        n = p.numel(); out.append(vec[i:i + n].view_as(p)); i += n
    return out


def cosine(a, b, eps=1e-12):
    return float((a @ b) / (a.norm() * b.norm() + eps))


def pcgrad(g1, g2):
    """Symmetric PCGrad for two tasks: project each gradient off the other when they conflict, then sum."""
    d = g1 @ g2
    p1, p2 = g1.clone(), g2.clone()
    if d < 0:
        p1 = g1 - d / (g2 @ g2).clamp(min=1e-20) * g2
        p2 = g2 - d / (g1 @ g1).clamp(min=1e-20) * g1
    return p1 + p2


def agem(g, g_ref):
    """A-GEM: if the update g would increase the reference (anchor) loss to first order, project it so that it does
    not: g <- g - (g.g_ref / |g_ref|^2) g_ref. (Gradients of LOSSES; descent direction is -g.)"""
    d = g @ g_ref
    if d < 0:
        return g - d / (g_ref @ g_ref).clamp(min=1e-20) * g_ref
    return g
