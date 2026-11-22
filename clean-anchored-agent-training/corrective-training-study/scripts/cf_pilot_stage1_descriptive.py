"""Stage-1 descriptive extras (not a gate): paired CF-CTRL diffs, CF-CAW2 on T1, raw-row fault-free check, held-out cells.
Usage: python scripts/cf_pilot_stage1_descriptive.py EXTRACTED_ARCHIVE_ROOT  (writes data/cf_pilot_stage1/descriptive.json)"""
import json,sys,collections
sys.path.insert(0,"scripts"); sys.path.insert(0,"src")
import cf_pilot_analysis as A
from verification import semantic_cf as T1, semantic_cf_t2 as T2
X=sys.argv[1]
r2=A.load_runs([X+"/cf_pilot/t2test"],"T2"); r1=A.load_runs([X+"/cf_pilot/t1"],"T1")
out={"paired_CF_minus_CTRL":{}, "CF_minus_CAW2_T1":{}}
for m in ("A_effect_sensitivity","hc_outcome_twin_correct","infeasible_safe_escalation","reconcile_twin_correct","outcome_twin_correct","feasible_end_state","unnecessary_check_rate","excess_calls","fault_free_success","decision_accuracy"):
    out["paired_CF_minus_CTRL"][m]=A.diff(A.pooled(r2,["CF_s0"],"P0",m),A.pooled(r2,["CTRL_s0"],"P0",m))
for m in ("A_effect_sensitivity","infeasible_safe_escalation","twin_joint_history","feasible_end_state","unnecessary_check_rate"):
    out["CF_minus_CAW2_T1"][m]=A.diff(A.pooled(r1,["CF_s0"],"P0",m),A.pooled(r1,["CAW2_s0","CAW2_s1"],"P0",m))
# raw-row independent checks
def rows(p): return [json.loads(l) for l in open(p,encoding="utf-8")]
ind={}
for tpl,d in (("T2","t2test"),("T1","t1")):
    for n in ("CF_s0","CTRL_s0"):
        R=rows(f"{X}/cf_pilot/{d}/{n}@P0.jsonl")
        ff=[x for x in R if x["error"]=="none"]
        ind[f"{tpl}|{n}"]={"n_fault_free":len(ff),"goal_state":sum(bool(x["goal_state"]) for x in ff),
            "no_dup":sum(not x["duplicate"] for x in ff),"no_escalate":sum(not x["escalated"] for x in ff),
            "correct":sum(bool(x["correct"]) for x in ff),"mean_calls":round(sum(x["n_calls"] for x in ff)/len(ff),3),
            "min_calls":round(sum(x["min_calls"] for x in ff)/len(ff),3)}
out["fault_free_raw"]=ind
# per-cell for held-out combos and oracle labels (T2)
cell={}
for n in ("CF_s0","CTRL_s0"):
    R=rows(f"{X}/cf_pilot/t2test/{n}@P0.jsonl")
    c=collections.defaultdict(list)
    for x in R:
        if x["heldout_combo"]: c["HO|"+"|".join((x["effect"],x["verifier"],x["compensation"]))+"|"+("reject" if x["error"]=="reject" else "none" if x["error"]=="none" else "timeout")].append(x["correct"])
        c["oracle|"+x["oracle"]].append(x["correct"])
    cell[n]={k:[round(sum(v)/len(v),3),len(v)] for k,v in sorted(c.items())}
out["cells"]=cell
# what CF does in escalate-oracle episodes
for n in ("CF_s0",):
    R=[x for x in rows(f"{X}/cf_pilot/t2test/{n}@P0.jsonl") if x["oracle"]=="escalate"]
    out["CF_escalate_first_action"]={str(k):v for k,v in collections.Counter((x["first"],x["after_check"]) for x in R).most_common()}
json.dump(out,open("data/cf_pilot_stage1/descriptive.json","w",encoding="utf-8",newline="\n"),indent=1,default=str)
print(json.dumps(out,indent=1,default=str))
