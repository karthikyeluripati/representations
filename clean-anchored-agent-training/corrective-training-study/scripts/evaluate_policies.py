"""Evaluate a policy (base / pi0 / pi0+adapter) with identical graders and splits, then run the failure taxonomy and
the error-recognition probe.

  python scripts/evaluate_policies.py --model /workspace/models/pi0 [--adapter <dir>] --name <name> --splits dev
  python scripts/evaluate_policies.py ... --splits test,heldout          (final, once per frozen checkpoint)
Outputs: results/eval/<name>.jsonl (+summary), results/failure_types/<name>.json, results/probe/<name>.{probe_items.jsonl,probe_summary.json} (names chosen so the eval-trace glob <name>.json* never matches)
The episode suites are those of the parent project (`src/evaluation/suites.py`), unchanged:
S0; S1/S2/S3 x seeds 0-2 (v2 pool); S1_obs/S2_obs (reserved EVAL outage strings); S2_act (reserved `_replica`
naming); RP/RA (real BFCL errors). Held-out = VehicleControlAPI family; "unseen error types" = normalised real-error
messages absent from all TRAIN corrective states.
"""
import argparse
import json
import os
import shutil
import subprocess
import sys

import _common

ap = argparse.ArgumentParser()
ap.add_argument("--model", required=True)
ap.add_argument("--adapter", default=None)
ap.add_argument("--name", required=True)
ap.add_argument("--splits", default="dev")
ap.add_argument("--skip-probe", action="store_true")
a = ap.parse_args()
P, S = _common.PROJ, _common.STUDY
ev = os.path.join(S, "results", "eval"); os.makedirs(ev, exist_ok=True)
if not os.path.exists(os.path.join(ev, a.name + ".jsonl")):
    cmd = [sys.executable, os.path.join(P, "scripts", "evaluate.py"), "--mode", "run", "--model", a.model, "--name", a.name, "--splits", a.splits]
    if a.adapter:
        cmd += ["--adapter", a.adapter]
    subprocess.run(cmd, check=True)
    for ext in (".jsonl", ".summary.json"):
        shutil.move(os.path.join(P, "results", "eval", a.name + ext), os.path.join(ev, a.name + ext))
subprocess.run([sys.executable, os.path.join(S, "scripts", "analyze_failure_types.py"), "--traces", os.path.join(ev, a.name + ".jsonl"),
                "--name", a.name], check=True, stdout=subprocess.DEVNULL)
if not a.skip_probe:
    from src.evaluation.suites import load_eval_states
    from src.environment.bfcl_env import load_bfcl, openai_tools
    from src.evaluation.agent_loop import ALTS
    from analysis.probe import dump, probe_items, run_probe, summarize_probe
    from vllm import LLM
    from vllm.lora.request import LoRARequest
    tasks = {t["id"]: t for t in json.load(open(os.path.join(P, "data", "tasks_v1.json"), encoding="utf-8"))}
    _, _, docs = load_bfcl()
    tools_fn = lambda tid, alt: openai_tools(tasks[tid], docs, ALTS[alt])  # noqa: E731
    st = load_eval_states([os.path.join(P, "data", "cacrt_pairs_v1", f"eval_states_{s}.jsonl") for s in a.splits.split(",")])
    items = probe_items(st, tasks)
    llm = LLM(model=a.model, enable_lora=bool(a.adapter), max_lora_rank=64, max_model_len=16384, gpu_memory_utilization=0.85, seed=0)
    lora = LoRARequest("arm", 1, a.adapter) if a.adapter else None
    res = run_probe(llm, llm.get_tokenizer(), lora, items, tools_fn)
    pd = os.path.join(S, "results", "probe"); os.makedirs(pd, exist_ok=True)
    dump(res, os.path.join(pd, a.name + ".probe_items.jsonl"))
    json.dump(summarize_probe(res), open(os.path.join(pd, a.name + ".probe_summary.json"), "w"), indent=1)
    print(json.dumps(summarize_probe(res), indent=1))
print(open(os.path.join(ev, a.name + ".summary.json")).read()[:3000])
