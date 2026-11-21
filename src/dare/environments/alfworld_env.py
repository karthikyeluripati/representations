"""Thin wrapper around ALFWorld's text (TextWorld PDDL) environment.

We bypass ALFWorld's batched gym wrapper and drive a single TextWorld
``PddlEnv`` directly so that we control resets, can fingerprint the
symbolic state, and can branch by deterministic replay.

State branching: ``PddlEnv.copy()`` raises NotImplementedError (verified with
textworld 1.7.0), so branching is done by *deterministic replay*: reset the
same game file and re-apply the logged action prefix. ``state_fingerprint``
hashes the full set of PDDL facts so equivalence can be checked exactly.
"""
from __future__ import annotations

import glob
import json
import os
import time
from dataclasses import dataclass, field

from dare.utils.common import stable_hash

TASK_FAMILIES = (
    "pick_and_place_simple",
    "look_at_obj_in_light",
    "pick_clean_then_place_in_recep",
    "pick_heat_then_place_in_recep",
    "pick_cool_then_place_in_recep",
    "pick_two_obj_and_place",
)
# PDDL predicates that record interaction *history* rather than world state: `checked(x)` is
# added by look/examine/inventory, `not_atlocation(agent, loc)` accumulates every location
# left. They are monotone, so a full-fact hash never repeats. The world fingerprint drops them.
BOOKKEEPING_PREDICATES = ("checked", "not_atlocation")
SPLIT_DIRS = {"train": "train", "valid_seen": "valid_seen", "valid_unseen": "valid_unseen"}


@dataclass
class GameInfo:
    game_file: str
    split: str
    task_family: str
    task_id: str  # "<task dir>/<trial dir>", unique per task instance


def task_family_of(task_dir_name: str) -> str:
    fam = task_dir_name.split("-")[0]
    if fam not in TASK_FAMILIES:
        raise ValueError(f"Unknown task family in {task_dir_name!r}")
    return fam


def list_games(data_root: str, split: str, task_families=None, check_solvable=True) -> list[GameInfo]:
    """Enumerate ALFWorld games, applying the same filters as ALFWorld's AlfredTWEnv
    (skip 'movable'/'Sliced' tasks and games not marked solvable)."""
    root = os.path.join(os.path.expanduser(data_root), SPLIT_DIRS[split])
    out = []
    for gf in sorted(glob.glob(os.path.join(root, "*", "*", "game.tw-pddl"))):
        trial_dir = os.path.dirname(gf)
        task_dir = os.path.basename(os.path.dirname(trial_dir))
        if "movable" in task_dir or "Sliced" in task_dir:
            continue
        fam = task_family_of(task_dir)
        if task_families and fam not in task_families:
            continue
        if check_solvable:
            with open(gf, "r") as f:
                if not json.load(f).get("solvable", False):
                    continue
        out.append(GameInfo(gf, split, fam, f"{task_dir}/{os.path.basename(trial_dir)}"))
    return out


def select_games(games: list[GameInfo], n: int, salt: str = "select-v1") -> list[GameInfo]:
    """Deterministic family-stratified subset: shuffle within family by stable hash, then round-robin."""
    by_fam: dict[str, list[GameInfo]] = {}
    for g in games:
        by_fam.setdefault(g.task_family, []).append(g)
    for fam in by_fam:
        by_fam[fam].sort(key=lambda g: stable_hash(salt, g.task_id))
    out, i = [], 0
    fams = sorted(by_fam)
    while len(out) < min(n, len(games)):
        for fam in fams:
            if i < len(by_fam[fam]) and len(out) < n:
                out.append(by_fam[fam][i])
        i += 1
    return out


@dataclass
class Step:
    observation: str
    admissible: list[str]
    won: bool
    done: bool
    reward: float
    fingerprint: str          # hash of ALL PDDL facts (strict; used to verify replay/branching)
    world_fp: str = ""        # hash excluding BOOKKEEPING_PREDICATES (used for revisit/waste)
    env_time_s: float = 0.0
    facts: list[str] = field(default_factory=list, repr=False)


def world_fingerprint(facts: list[str]) -> str:
    return stable_hash(*[f for f in facts if f.split("(", 1)[0] not in BOOKKEEPING_PREDICATES])


def extract_objective(intro: str) -> str:
    key = "Your task is to:"
    i = intro.find(key)
    return intro[i + len(key):].strip() if i >= 0 else ""


class AlfWorldEnv:
    """Single-game ALFWorld text environment with fingerprinting and replay."""

    def __init__(self, max_steps: int = 50):
        import textworld  # imported lazily: Linux/WSL only
        from alfworld.agents.environment.alfred_tw_env import AlfredDemangler, AlfredInfos

        self._tw = textworld
        self._wrappers = lambda: [AlfredDemangler(shuffle=False), AlfredInfos]
        self._infos = textworld.EnvInfos(won=True, admissible_commands=True, facts=True, extras=["gamefile"])
        self.max_steps = max_steps
        self._env = None
        self._game_file = None
        self.t = 0
        self.objective = ""

    def _ensure(self, game_file: str):
        if self._env is None or self._game_file != game_file:
            if self._env is not None:
                self._env.close()
            self._env = self._tw.start(game_file, request_infos=self._infos, wrappers=self._wrappers())
            self._game_file = game_file

    @staticmethod
    def _fingerprint(gs) -> tuple[str, list[str]]:
        facts = sorted(str(f) for f in gs["_facts"])
        return stable_hash(*facts), facts

    def _wrap(self, gs, reward: float, done: bool, dt: float) -> Step:
        fp, facts = self._fingerprint(gs)
        won = bool(gs.won)
        truncated = self.t >= self.max_steps
        return Step(gs.feedback.strip(), sorted(gs.admissible_commands), won, bool(done or won or truncated),
                    float(reward), fp, world_fingerprint(facts), dt, facts)

    def reset(self, game_file: str) -> Step:
        t0 = time.perf_counter()
        self._ensure(game_file)
        gs = self._env.reset()
        self.t = 0
        self.objective = extract_objective(gs.feedback)
        return self._wrap(gs, 0.0, False, time.perf_counter() - t0)

    def step(self, action: str) -> Step:
        t0 = time.perf_counter()
        gs, reward, done = self._env.step(action)
        self.t += 1
        return self._wrap(gs, reward, done, time.perf_counter() - t0)

    def replay(self, game_file: str, actions: list[str]) -> Step:
        """Reconstruct the state reached by `actions` from a fresh reset."""
        st = self.reset(game_file)
        for a in actions:
            st = self.step(a)
        return st

    def close(self):
        if self._env is not None:
            self._env.close()
            self._env = None
