"""Objectives on toy tensors and a tiny random Qwen3: values, masks, gradient directions, gradient operators."""
import math

import torch

from src.training import objectives as O
from src.training.render import ToyRenderer, encode_action_example, encode_trajectory


def _logits(T=6, V=5, seed=0, grad=False):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(1, T, V, generator=g)
    return x.requires_grad_(grad)


def test_seq_logp_matches_manual():
    lg = _logits(); ids = torch.tensor([[0, 1, 2, 3, 4, 1]]); m = torch.tensor([[0, 0, 0, 1, 1, 1]])
    lp = torch.log_softmax(lg, -1)
    manual = lp[0, 2, 3] + lp[0, 3, 4] + lp[0, 4, 1]
    assert torch.allclose(O.seq_logp(lg, ids, m)[0], manual)


def test_masked_tokens_get_no_gradient():
    lg = _logits(grad=True); ids = torch.tensor([[0, 1, 2, 3, 4, 1]]); m = torch.tensor([[0, 0, 0, 1, 1, 1]])
    O.sft_loss(lg, ids, m).backward()
    assert lg.grad[0, :2].abs().sum() == 0 and lg.grad[0, 2:5].abs().sum() > 0 and lg.grad[0, 5].abs().sum() == 0


def test_dpo_value_and_direction():
    pc = torch.tensor([-3.0], requires_grad=True); pr = torch.tensor([-3.0], requires_grad=True)
    loss, _ = O.dpo_loss(pc, pr, torch.tensor([-3.0]), torch.tensor([-3.0]), beta=0.1)
    assert math.isclose(float(loss), math.log(2), rel_tol=1e-6)            # policy == reference -> log 2
    loss.backward()
    assert pc.grad < 0 and pr.grad > 0                                         # descent raises chosen, lowers rejected


def test_anchor_kl_zero_and_flat_at_reference():
    lg = _logits(grad=True); ids = torch.tensor([[0, 1, 2, 3, 4, 1]]); m = torch.tensor([[0, 0, 0, 1, 1, 1]])
    kl = O.anchor_kl(lg, lg.detach().clone(), m)
    kl.backward()
    assert abs(float(kl)) < 1e-6 and lg.grad.abs().max() < 1e-6


def test_pcgrad_and_agem_remove_conflict():
    g1 = torch.tensor([1.0, 0.0]); g2 = torch.tensor([-1.0, 1.0])
    p = O.pcgrad(g1, g2)
    assert torch.allclose(p, torch.tensor([0.5, 1.5]))
    a = O.agem(g1, g2)
    assert float(a @ g2) >= -1e-6
    assert torch.allclose(O.agem(g1, torch.tensor([1.0, 1.0])), g1)            # no conflict -> unchanged
    assert torch.allclose(O.pcgrad(g1, torch.tensor([1.0, 1.0])), torch.tensor([2.0, 1.0]))


def test_flat_unflat_roundtrip():
    ps = [torch.randn(2, 3), torch.randn(4)]
    v = O.flat(ps)
    for a, b in zip(O.unflat(v, ps), ps):
        assert torch.equal(a, b)


def _tools():
    return [{"type": "function", "function": {"name": "cd", "description": "", "parameters": {}}}]


def test_render_target_span_and_trajectory_mask():
    r = ToyRenderer()
    msgs = [{"role": "system", "content": "s"}, {"role": "user", "content": "u"}]
    e = encode_action_example(r, msgs, {"name": "cd", "arguments": {"folder": "a"}}, _tools(), 10_000)
    tail = r.tok.decode(e["input_ids"][-e["n_target"]:])
    assert "cd" in tail and "folder" in tail and sum(e["mask"]) == e["n_target"]
    traj = msgs + [{"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "cd", "arguments": {"folder": "x"}}}], "loss": False},
                   {"role": "tool", "content": "err"},
                   {"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": "cd", "arguments": {"folder": "a"}}}]}]
    t = encode_trajectory(r, traj, _tools(), 10_000)
    sup = r.tok.decode([i for i, m in zip(t["input_ids"], t["mask"]) if m])
    assert '"a"' in sup and '"x"' not in sup                                    # loss=False action is context only


def _tiny():
    from transformers import Qwen3Config, Qwen3ForCausalLM
    torch.manual_seed(0)
    return Qwen3ForCausalLM(Qwen3Config(vocab_size=258, hidden_size=32, intermediate_size=64, num_hidden_layers=2,
                                        num_attention_heads=2, num_key_value_heads=1, head_dim=16, tie_word_embeddings=True)).eval()


def test_logits_to_keep_path_equals_full_forward():
    from src.training.trainer import forward_logits
    m = _tiny()
    ex = {"input_ids": list(range(20, 60)), "mask": [0] * 30 + [1] * 10, "n_target": 10, "suffix": True}
    with torch.no_grad():
        a = O.seq_logp(*forward_logits(m, [ex], 256, "cpu"))
        b = O.seq_logp(*forward_logits(m, [dict(ex, suffix=False)], 256, "cpu"))
    assert torch.allclose(a, b, atol=1e-5)


def test_padding_does_not_change_logps():
    from src.training.trainer import forward_logits
    m = _tiny()
    e1 = {"input_ids": list(range(20, 40)), "mask": [0] * 15 + [1] * 5, "n_target": 5}
    e2 = {"input_ids": list(range(30, 60)), "mask": [0] * 25 + [1] * 5, "n_target": 5}
    with torch.no_grad():
        joint = O.seq_logp(*forward_logits(m, [e1, e2], 256, "cpu"))
        solo = O.seq_logp(*forward_logits(m, [e1], 256, "cpu"))
    assert torch.allclose(joint[0], solo[0], atol=1e-4)


def test_left_padded_suffix_batch_matches_single_and_grads_finite():
    from src.training.trainer import forward_logits
    m = _tiny().train()
    e1 = {"input_ids": list(range(20, 40)), "mask": [0] * 15 + [1] * 5, "n_target": 5, "suffix": True}
    e2 = {"input_ids": list(range(30, 70)), "mask": [0] * 33 + [1] * 7, "n_target": 7, "suffix": True}
    m.eval()
    with torch.no_grad():
        joint = O.seq_logp(*forward_logits(m, [e1, e2], 256, "cpu"))
        s1 = O.seq_logp(*forward_logits(m, [e1], 256, "cpu"))
        s2 = O.seq_logp(*forward_logits(m, [e2], 256, "cpu"))
    assert torch.allclose(joint, torch.cat([s1, s2]), atol=1e-4)
    m.train()
    loss = -O.seq_logp(*forward_logits(m, [e1, e2], 256, "cpu")).sum()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in m.parameters() if p.grad is not None)
