"""Append-only, resumable JSONL trajectory logging.

Layout of a run directory:
  transitions.jsonl  one record per environment step
  episodes.jsonl     one record per *completed* episode (written after its transitions)
  run_meta.json      model id, config, config hash, environment versions

An episode counts as complete only if its summary is present in episodes.jsonl.
Transitions of incomplete episodes (e.g. a crash mid-episode) are ignored by
`load_run` and the episode is re-run on resume. Transitions are buffered and
written together with the summary to keep the files consistent.

Histories are not duplicated per transition: a transition references its
episode (`episode_id`) and `step`; the full action history lives once in the
episode summary (`actions`) and the per-step observations in the transitions.
"""
from __future__ import annotations

import json
import os
import uuid
from pathlib import Path


class RunLogger:
    def __init__(self, run_dir: str | Path, meta: dict | None = None):
        self.dir = Path(run_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.t_path = self.dir / "transitions.jsonl"
        self.e_path = self.dir / "episodes.jsonl"
        if meta is not None:
            mp = self.dir / "run_meta.json"
            if mp.exists():
                old = json.loads(mp.read_text())
                if old.get("config_hash") != meta.get("config_hash"):
                    raise RuntimeError(f"Refusing to resume {self.dir}: config hash changed "
                                       f"({old.get('config_hash')} -> {meta.get('config_hash')})")
            else:
                mp.write_text(json.dumps(meta, indent=2, default=str))
        self._buf: list[dict] = []
        # Unique per logger instance: lets load_run drop transitions left by a crashed attempt.
        self.attempt = uuid.uuid4().hex[:12]

    def completed_episode_ids(self) -> set[str]:
        return {e["episode_id"] for e in read_jsonl(self.e_path)}

    def log_transition(self, rec: dict):
        self._buf.append({**rec, "attempt": self.attempt})

    def end_episode(self, summary: dict):
        _terminate_torn_line(self.t_path)
        _terminate_torn_line(self.e_path)
        with open(self.t_path, "a", encoding="utf-8") as f:
            for r in self._buf:
                f.write(json.dumps(r, default=_np_default) + "\n")
            f.flush()
            os.fsync(f.fileno())
        with open(self.e_path, "a", encoding="utf-8") as f:
            f.write(json.dumps({**summary, "attempt": self.attempt}, default=_np_default) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._buf = []

    def discard_episode(self):
        self._buf = []


def _terminate_torn_line(path: Path):
    """If a crash left a partial last line, end it so the next record is not glued onto it."""
    if path.exists() and path.stat().st_size > 0:
        with open(path, "rb+") as f:
            f.seek(-1, os.SEEK_END)
            if f.read(1) != b"\n":
                f.write(b"\n")


def _np_default(o):
    try:
        import numpy as np

        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.generic):
            return o.item()
    except ImportError:
        pass
    raise TypeError(f"not JSON serialisable: {type(o)}")


def read_jsonl(path: str | Path) -> list[dict]:
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                # a torn final line from a crash; ignore it
                continue
    return out


def load_run(run_dir: str | Path) -> tuple[list[dict], list[dict]]:
    """Return (episodes, transitions) restricted to completed episodes."""
    run_dir = Path(run_dir)
    eps = read_jsonl(run_dir / "episodes.jsonl")
    done = {(e["episode_id"], e.get("attempt")) for e in eps}
    trs = [t for t in read_jsonl(run_dir / "transitions.jsonl") if (t["episode_id"], t.get("attempt")) in done]
    return eps, trs
