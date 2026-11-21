"""Build paired training trajectories from the executable reference policy (train split only).

For every train task and seed we roll out the reference policy in S0, S1, S2, S3 (same task, same seed => same error
string and same k). Each trajectory is converted to chat messages (assistant tool calls + tool observations).
Decision points = assistant messages. For a pair (Sx, S3), x in {1,2}, the DIVERGENCE point is the first assistant index
at which the two trajectories' assistant actions differ. By construction the histories before that point are identical
in actions, and the last observation before it differs (retry/backup succeeded vs failed); that observation is the
evidence that distinguishes the worlds. Hidden labels are used only to build targets, never in model inputs.
"""
import json

from environments.bfcl_faults import Episode, reference_policy
from agents.rollout import initial_messages, openai_tools, SYSTEM_RECOVERY

ESC_FINAL = None  # episode ends at the escalate call (matches rollout termination)


def trajectory_messages(task, scenario, seed, system=SYSTEM_RECOVERY):
    ep = Episode(task, scenario, seed)
    reference_policy(ep)
    msgs = initial_messages(task, system)
    n0 = len(msgs)
    for name, args, obs in ep.calls:
        msgs.append({"role": "assistant", "content": "", "tool_calls": [{"type": "function", "function": {"name": name, "arguments": args}}]})
        if name == "escalate_to_user":
            break
        msgs.append({"role": "tool", "content": obs[:2000]})
    if not ep.escalated:
        msgs.append({"role": "assistant", "content": "Done."})
    g = ep.grade()
    assert g["correct"], (task["id"], scenario, seed)
    return {"msgs": msgs, "first_new": n0, "scenario": scenario, "task_id": task["id"], "seed": seed}


def assistant_indices(tr):
    return [i for i in range(tr["first_new"], len(tr["msgs"])) if tr["msgs"][i]["role"] == "assistant"]


def action_key(m):
    if m.get("tool_calls"):
        f = m["tool_calls"][0]["function"]
        return json.dumps([f["name"], f["arguments"]], sort_keys=True)
    return "TEXT:" + m["content"]


def divergence(tr_a, tr_b):
    """Return (i_a, i_b): indices of the first differing assistant action in a and b, or None."""
    ia, ib = assistant_indices(tr_a), assistant_indices(tr_b)
    for x, y in zip(ia, ib):
        if action_key(tr_a["msgs"][x]) != action_key(tr_b["msgs"][y]):
            return x, y
    return None


def build(tasks, docs, seeds=(0,), scenarios=("S0", "S1", "S2", "S3")):
    trajs, pairs = [], []
    for t in tasks:
        tools = openai_tools(t, docs)
        for s in seeds:
            by = {}
            for sc in scenarios:
                tr = trajectory_messages(t, sc, s)
                tr["tools"] = tools
                by[sc] = tr
                trajs.append(tr)
            if "S3" in by:
                for x in ("S1", "S2"):
                    if x in by:
                        d = divergence(by[x], by["S3"])
                        if d:
                            pairs.append({"solvable": by[x], "impossible": by["S3"], "i_solvable": d[0], "i_impossible": d[1], "type": f"{x}-S3"})
    return trajs, pairs
