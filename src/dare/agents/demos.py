"""In-context demonstrations from ALFWorld's handcoded expert, built from TRAIN-split games only.

ReAct-style ALFWorld agents use task-type-specific few-shot examples; zero-shot
likelihood scoring with small instruct models degenerates into repeated
'examine X' loops (observed, see docs/experiment_log.md). One successful expert
transcript per task family is generated deterministically and cached to JSON.
Using the task family to pick the demo mirrors ReAct's per-type prompts.
"""
from __future__ import annotations

import json
import os

from dare.environments.alfworld_env import TASK_FAMILIES, list_games, select_games


def _expert_transcript(game_file: str, max_steps: int = 40) -> tuple[bool, str]:
    import textworld
    from alfworld.agents.environment.alfred_tw_env import (AlfredDemangler, AlfredExpert, AlfredExpertType,
                                                             AlfredInfos)

    infos = textworld.EnvInfos(won=True, admissible_commands=True, extras=["gamefile", "expert_plan"])
    env = textworld.start(game_file, request_infos=infos,
                          wrappers=[AlfredDemangler(shuffle=False), AlfredInfos, AlfredExpert(AlfredExpertType.HANDCODED)])
    gs = env.reset()
    intro = gs.feedback.strip().replace("-= Welcome to TextWorld, ALFRED! =-", "").strip()
    lines = [intro]
    for _ in range(max_steps):
        plan = gs["extra.expert_plan"]
        if not plan:
            break
        a = plan[0]
        gs, _, done = env.step(a)
        lines.append(f"> {a}\n{gs.feedback.strip()}")
        if gs.won or done:
            break
    env.close()
    return bool(gs.won), "\n".join(lines)


def build_demos(data_root: str, cache_path: str, salt: str = "demo-v1") -> dict[str, str]:
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            return json.load(f)
    demos, meta = {}, {}
    for fam in TASK_FAMILIES:
        for g in select_games(list_games(data_root, "train", [fam]), 10, salt=salt):
            ok, text = _expert_transcript(g.game_file)
            if ok:
                demos[fam] = text
                meta[fam] = g.task_id
                break
    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    with open(cache_path, "w") as f:
        json.dump(demos, f, indent=1)
    with open(cache_path + ".sources.json", "w") as f:
        json.dump(meta, f, indent=1)  # which TRAIN task_ids the demos came from (exclude from datasets)
    return demos
