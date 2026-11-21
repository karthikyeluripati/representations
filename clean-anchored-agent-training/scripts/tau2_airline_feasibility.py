"""tau2-bench airline (vendored @c06b07dd, MIT) feasibility for uncertain-outcome faults: for every gold write action,
execute it once vs twice (DB hash) and test whether a read the agent can call shows the change.
Usage: TAU2 deps on PYTHONPATH; python scripts/tau2_airline_feasibility.py third_party/tau2_c06b07dd results/tau2/airline_write_events.json"""
import collections, copy, json, os, sys, types
T = sys.argv[1]
os.environ["TAU2_DATA_DIR"] = os.path.join(T, "data")
pkg = types.ModuleType("tau2"); pkg.__path__ = [os.path.join(T, "src", "tau2")]; sys.modules["tau2"] = pkg
from loguru import logger; logger.remove()
from tau2.domains.airline.environment import get_environment, get_tasks
from tau2.domains.airline.data_model import FlightDB
from tau2.domains.airline.utils import AIRLINE_DB_PATH
BASE = FlightDB.load(AIRLINE_DB_PATH)
READS = {"get_user_details": ["user_id"], "get_reservation_details": ["reservation_id"]}
def fresh(t):
    env = get_environment(db=BASE)
    if t.initial_state and t.initial_state.initialization_actions:
        for a in t.initial_state.initialization_actions: env.run_env_function_call(a)
    return env
def run(env, acts):
    out = []
    for a in acts:
        try: out.append(str(env.use_tool(a.name, **a.arguments)))
        except Exception as e: out.append(f"Error: {e}")
    return out
rows, why = [], collections.Counter()
for t in get_tasks(None):
    acts = t.evaluation_criteria.actions or []
    for i, a in enumerate(acts):
        env0 = fresh(t); run(env0, acts[:i]); tool_is_write = env0._is_mutating_tool(a.name)
        if not tool_is_write: continue
        e1 = fresh(t); run(e1, acts[:i + 1]); h1 = e1.get_db_hash()
        e2 = fresh(t); run(e2, acts[:i + 1] + [a]); h2 = e2.get_db_hash()
        non_idem = h1 != h2
        # read-after: does a read the agent can call (ids known from context) show a change between before/after?
        known = {k: v for x in acts[:i + 1] for k, v in x.arguments.items() if isinstance(v, str)}
        dec = []
        for rn, req in READS.items():
            if all(r in known for r in req):
                b = fresh(t); run(b, acts[:i]); ob = str(b.use_tool(rn, **{r: known[r] for r in req}))
                c = fresh(t); run(c, acts[:i + 1]); oc = str(c.use_tool(rn, **{r: known[r] for r in req}))
                if ob != oc: dec.append(rn)
        rows.append({"task": t.id, "i": i, "fn": a.name, "non_idempotent": non_idem, "read_shows_change": dec})
        why[(a.name, "NON-IDEM" if non_idem else "idem", "readable" if dec else "unreadable")] += 1
for k, v in sorted(why.items()): print(k, v)
diag = [r for r in rows if r["non_idempotent"] and r["read_shows_change"]]
print("diagnostic write events:", len(diag), "tasks:", len({r['task'] for r in diag}))
json.dump(rows, open(sys.argv[2], "w"), indent=1)
