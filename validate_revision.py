"""Cross-check reported experiments against persisted solver evidence."""
from pathlib import Path
import hashlib
import json
import re
import argparse
import numpy as np
import pandas as pd
import revision_model as m

ROOT=Path(__file__).resolve().parent;OUT=ROOT/"revision/results"


def main(results_only=False):
    checks=[];keys=set()
    for name,count in [("historical",722),("sensitivity",240),("boundaries",40),("runtime",30)]:
        df=pd.read_csv(OUT/f"{name}.csv");assert len(df)==count,(name,len(df))
        for prefix in ["qp","miqp"]:keys.update(df[prefix+"_key"])
        cert=df[df.certificate & df.qp_has_solution]
        assert (cert.qp_max_simultaneous_kw<1e-5).all(),name
        assert not df.classification.str.contains("solver_status|error").any() or name=="boundaries"
        checks.append({"dataset":name,"rows":len(df),"classification":df.classification.value_counts().to_dict()})
    for name,count in [("calibration",726),("fleet_reliability",240),("degradation",216)]:
        df=pd.read_csv(OUT/f"{name}.csv");assert len(df)==count,(name,len(df));assert df.optimal.all();keys.update(df.key)
        checks.append({"dataset":name,"rows":len(df),"all_optimal":bool(df.optimal.all())})
    correlation=pd.read_csv(OUT/"correlation.csv");assert len(correlation)==54;keys.update(correlation.key)
    errors=[];max_res=0;max_obj=0
    array_fields={f.name for f in m.Instance.__dataclass_fields__.values()}-{"name","capacity","alpha","dt","capacity_mode","export_limit"}
    for key in sorted(keys):
        folder=ROOT/"revision/solves"/key
        r=json.loads((folder/"result.json").read_text());p=json.loads((folder/"input.json").read_text())
        if not r["has_solution"]:
            assert r["status"]=="infeasible",(key,r["status"]);continue
        d=m.Instance(**{k:np.array(v,float) if k in array_fields else v for k,v in p["instance"].items()})
        x=np.array(r["x"]);y=np.array(r["y"]);e=np.array(r["e"])
        res=m.feasibility_residual(d,x,y,e,r["binary"])
        err=abs(m.objective_components(d,x,y)["objective"]-r["solver_objective"])
        max_res=max(max_res,res);max_obj=max(max_obj,err)
        assert res<2e-5,(key,res)
        assert err<max(1e-6,1e-9*abs(r["solver_objective"])),(key,err)
        log=(folder/"solver.log").read_text(errors="replace")
        assert "22.1.2.0" in log and "CPLEX Error" not in log,key
    split=json.loads((OUT/"calibration_split.json").read_text())
    tr,ca,te=[split[k] for k in ["training_days","calibration_days","test_days"]]
    assert max(tr)<min(ca) and max(ca)<min(te)
    assert not (set(tr)&set(ca)|set(tr)&set(te)|set(ca)&set(te))
    if not results_only:
        tex=(ROOT/"paper.tex").read_text()
        citations=set(c for group in re.findall(r"\\cite\{([^}]+)\}",tex) for c in group.split(","))
        bib=set(re.findall(r"\\bibitem\{([^}]+)\}",tex));assert citations==bib,(citations-bib,bib-citations)
        refs=set(re.findall(r"\\ref\{([^}]+)\}",tex));labels=set(re.findall(r"\\label\{([^}]+)\}",tex));assert refs<=labels,refs-labels
        for path in re.findall(r"\\(?:input|includegraphics)(?:\[[^]]*\])?\{([^}]+)\}",tex):assert (ROOT/path).exists(),path
    summary={"passed":True,"unique_recorded_solves_checked":len(keys),"datasets":checks,
             "max_recomputed_constraint_residual":max_res,"max_objective_recompute_error":max_obj,
             "chronological_split_disjoint":True,"all_references_cited":None if results_only else True,
             "public_repository_status":"pending_author_publication","note":"Solver tolerances, not exact rational arithmetic, determine numerical optimality."}
    (OUT/"validation.json").write_text(json.dumps(summary,indent=2))
    (OUT/"current_solve_keys.json").write_text(json.dumps(sorted(keys)))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("--results-only",action="store_true",help="Validate solver evidence without the private manuscript sources.")
    main(results_only=parser.parse_args().results_only)
