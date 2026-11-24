"""Auditable hourly/subhourly V2G QP/MIQP with the user's CPLEX executable.

All powers are kW, energy kWh, prices EUR/kWh, alpha EUR/(kW^2 h).
Python CPLEX is used only to write models; the executable performs every solve.
No solver exception or time-limited incumbent is silently called an optimum.
"""
from __future__ import annotations
from dataclasses import dataclass, asdict, replace
from pathlib import Path
import hashlib
import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET

import cplex
import numpy as np
import psutil
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
CPLEX_EXE = Path(os.environ.get("CPLEX_EXE", r"C:\Program Files\IBM\ILOG\CPLEX_Studio2212\cplex\bin\x64_win64\cplex.exe"))
MODEL_VERSION = "revision-1.0"


@dataclass
class Instance:
    name: str
    L: np.ndarray
    buy: np.ndarray
    sell: np.ndarray
    availability: np.ndarray
    pcmax: np.ndarray
    pdmax: np.ndarray
    e0: np.ndarray
    emin: np.ndarray
    emax: np.ndarray
    target: np.ndarray
    etac: np.ndarray
    etad: np.ndarray
    rho: np.ndarray
    coeff: np.ndarray
    capacity: float
    alpha: float = 2e-4
    dt: float = 1.0
    capacity_mode: str = "charging"
    export_limit: float = 0.0

    @property
    def N(self): return self.availability.shape[0]
    @property
    def T(self): return len(self.L)

    def validate(self):
        n,t=self.N,self.T
        assert self.capacity_mode in {"charging", "net_import", "two_sided"}
        assert self.dt > 0 and self.alpha >= 0
        assert self.availability.shape == (n,t) and self.coeff.shape == (n,t)
        assert np.all(self.coeff >= 0) and np.all((self.availability >= 0)&(self.availability <= 1))
        for a in [self.buy,self.sell]: assert a.shape == (t,) and np.isfinite(a).all()
        for a in [self.pcmax,self.pdmax,self.e0,self.emin,self.emax,self.target,self.etac,self.etad,self.rho]:
            assert a.shape == (n,) and np.isfinite(a).all()
        assert np.all((self.etac>0)&(self.etac<=1)&(self.etad>0)&(self.etad<=1))
        assert np.all(self.rho>=0) and np.all(self.pcmax>=0) and np.all(self.pdmax>=0)


def serializable(inst):
    return {k:(v.tolist() if isinstance(v,np.ndarray) else v) for k,v in asdict(inst).items()}


def from_legacy(d, *, raw_prices=False):
    """Import old instance parameters, explicitly interpreting scaled load as synthetic kW."""
    n=getattr(d,"N",1);t=d.T
    def vec(x): return np.broadcast_to(np.asarray(x,float),(n,)).copy()
    av=np.asarray(d.availability,float).reshape(n,t)
    coeff=np.broadcast_to(np.asarray(d.u_nominal,float),(n,t)).copy()
    return Instance(d.name,np.array(d.L),np.array(d.price_buy),np.array(d.price_sell),av,
                    vec(d.X),vec(d.Y),vec(d.E0),vec(d.Emin),vec(d.Emax),vec(d.Etarget),
                    vec(d.eta_c),vec(d.eta_d),vec(d.rho),coeff,float(d.M),float(d.alpha))


def objective_components(d,x,y):
    load=d.L+(x-y).sum(axis=0)
    market=float(d.dt*np.sum(d.buy*x-d.sell*y))
    throughput=float(d.dt*np.sum(d.rho[:,None]*(x+y)))
    smoothing=float(d.dt*d.alpha*np.sum(load**2))
    return {"objective":market+throughput+smoothing,"market_cost":market,
            "throughput_cost":throughput,"linear_cost":market+throughput,
            "load_cost":smoothing,"discharge_kwh":float(d.dt*y.sum()),
            "charge_kwh":float(d.dt*x.sum()),"min_net_load":float(load.min()),
            "max_simultaneous_kw":float(np.minimum(x,y).max())}


def feasibility_residual(d,x,y,e,complementarity=False):
    load=d.L+(x-y).sum(axis=0)
    r=[-x.min(),-y.min(),np.max(x-d.pcmax[:,None]*d.availability),
       np.max(y-d.pdmax[:,None]*d.availability),np.max(np.abs(e[:,0]-d.e0)),
       np.max(np.abs(np.diff(e,axis=1)-d.dt*(d.etac[:,None]*x-y/d.etad[:,None]))),
       np.max(d.emin[:,None]-e),np.max(e-d.emax[:,None]),np.max(d.target-e[:,-1])]
    if d.capacity_mode=="charging": r.append(np.max(d.L+(d.coeff*x).sum(axis=0)-d.capacity))
    else: r.append(np.max(load-d.capacity))
    if d.capacity_mode=="two_sided": r.append(np.max(-d.export_limit-load))
    if complementarity: r.append(np.minimum(x,y).max())
    return float(max(0,*r))


def certificate(d):
    """Conservative parameter-only specialization of Liang's directional condition.

    It is sufficient, never claimed necessary. Bounds apply over the full box.
    Export constraints are safe only when their lower bound is redundant over it.
    """
    k=d.etac*d.etad
    lower=d.L-(d.pdmax[:,None]*d.availability).sum(axis=0)
    D=d.buy[None,:]-k[:,None]*d.sell[None,:]+(1+k[:,None])*d.rho[:,None]
    D+=2*d.alpha*(1-k[:,None])*lower[None,:]
    active=(d.availability>0)&(d.pcmax[:,None]>0)&(d.pdmax[:,None]>0)
    margin=float(D[active].min()) if active.any() else float("inf")
    safe=d.capacity_mode!="two_sided" or np.all(lower>=-d.export_limit-1e-12)
    old=bool(np.all(d.buy>=d.sell)&np.all(d.sell>=0)&np.all(k<1)&np.all(d.rho>0)&np.all(lower>=0))
    return {"certificate":bool(margin>1e-10 and safe),"directional_margin":margin,
            "constraint_direction_safe":bool(safe),"old_assumptions":old}


def cycle_repair(d,x,y):
    k=(d.etac*d.etad)[:,None]
    s=np.minimum(x,y/k)
    xx=x-s;yy=y-k*s
    ee=np.concatenate([d.e0[:,None],d.e0[:,None]+np.cumsum(d.dt*(d.etac[:,None]*xx-yy/d.etad[:,None]),axis=1)],axis=1)
    return xx,yy,ee


def solve(d: Instance, binary=False, tag="run", time_limit=60, force=False):
    d.validate()
    settings={"binary":binary,"time_limit":time_limit,"threads":1,"mipgap":1e-8,
              "feasibility_tol":1e-8,"optimality_tol":1e-8,"integrality_tol":1e-8,
              "version":MODEL_VERSION,"cplex_exe":str(CPLEX_EXE)}
    payload={"instance":serializable(d),"settings":settings}
    key=hashlib.sha256(json.dumps(payload,sort_keys=True).encode()).hexdigest()[:20]
    folder=ROOT/"revision"/"solves"/key
    folder.mkdir(parents=True,exist_ok=True)
    resultfile=folder/"result.json"
    if resultfile.exists() and not force:
        r=json.loads(resultfile.read_text());r["tag"]=tag;return r
    # A forced failed re-solve must never be mistaken for a stale successful XML.
    if (folder/"solution.sol").exists():
        (folder/"solution.sol").unlink()
    (folder/"input.json").write_text(json.dumps(payload,indent=2),encoding="utf-8")
    c=cplex.Cplex()
    for setter in [c.set_log_stream,c.set_results_stream,c.set_warning_stream,c.set_error_stream]: setter(None)
    begin=time.perf_counter();n,t=d.N,d.T;nt=n*t
    x=np.arange(nt).reshape(n,t);y=x+nt
    e=np.arange(2*nt,2*nt+n*(t+1)).reshape(n,t+1)
    ell=np.arange(e.max()+1,e.max()+1+t);z=np.arange(ell.max()+1,ell.max()+1+nt).reshape(n,t)
    lb=np.r_[np.zeros(2*nt),np.repeat(d.emin,t+1),np.full(t,-cplex.infinity)]
    ub=np.r_[(d.pcmax[:,None]*d.availability).ravel(),(d.pdmax[:,None]*d.availability).ravel(),np.repeat(d.emax,t+1),np.full(t,cplex.infinity)]
    cost=np.r_[(d.dt*(d.buy[None,:]+d.rho[:,None])+np.zeros((n,t))).ravel(),
               (d.dt*(-d.sell[None,:]+d.rho[:,None])+np.zeros((n,t))).ravel(),np.zeros(n*(t+1)+t)]
    names=[f"x_{i}_{h}" for i in range(n) for h in range(t)]+[f"y_{i}_{h}" for i in range(n) for h in range(t)]+[f"e_{i}_{h}" for i in range(n) for h in range(t+1)]+[f"l_{h}" for h in range(t)]
    c.variables.add(obj=cost.tolist(),lb=lb.tolist(),ub=ub.tolist(),names=names)
    if binary: c.variables.add(lb=[0.]*nt,ub=[1.]*nt,types="B"*nt,names=[f"z_{i}_{h}" for i in range(n) for h in range(t)])
    if d.alpha: c.objective.set_quadratic_coefficients([(int(j),int(j),2*d.dt*d.alpha) for j in ell])
    expr=[];senses=[];rhs=[]
    def row(ids,vals,s,b):
        expr.append(cplex.SparsePair([int(j) for j in ids],[float(v) for v in vals]));senses.append(s);rhs.append(float(b))
    for i in range(n):
        row([e[i,0]],[1],"E",d.e0[i]);row([e[i,-1]],[1],"G",d.target[i])
        for h in range(t):
            row([e[i,h+1],e[i,h],x[i,h],y[i,h]],[1,-1,-d.dt*d.etac[i],d.dt/d.etad[i]],"E",0)
            if binary:
                row([x[i,h],z[i,h]],[1,-d.pcmax[i]*d.availability[i,h]],"L",0)
                row([y[i,h],z[i,h]],[1,d.pdmax[i]*d.availability[i,h]],"L",d.pdmax[i]*d.availability[i,h])
    for h in range(t):
        row([ell[h],*x[:,h],*y[:,h]],[1,*(-np.ones(n)),*np.ones(n)],"E",d.L[h])
        if d.capacity_mode=="charging":row(x[:,h],d.coeff[:,h],"L",d.capacity-d.L[h])
        else:row([ell[h]],[1],"L",d.capacity)
        if d.capacity_mode=="two_sided":row([ell[h]],[1],"G",-d.export_limit)
    c.linear_constraints.add(lin_expr=expr,senses="".join(senses),rhs=rhs)
    c.write(str(folder/"model.sav"));build_seconds=time.perf_counter()-begin
    nv=c.variables.get_num();nr=c.linear_constraints.get_num();c.end()
    commands=["read model.sav","set threads 1","set randomseed 20261112",f"set timelimit {time_limit}",
              "set simplex tolerances feasibility 1e-8","set simplex tolerances optimality 1e-8",
              "set mip tolerances mipgap 1e-8","set mip tolerances integrality 1e-8",
              "optimize","write solution.sol","quit"]
    start=time.perf_counter();peak=0
    with (folder/"solver.log").open("w",encoding="utf-8") as f:
        proc=subprocess.Popen([str(CPLEX_EXE),"-c",*commands],cwd=folder,stdout=f,stderr=subprocess.STDOUT,
                              creationflags=getattr(subprocess,"CREATE_NO_WINDOW",0))
        ps=psutil.Process(proc.pid)
        while proc.poll() is None:
            try: peak=max(peak,ps.memory_info().rss)
            except psutil.Error: pass
            if time.perf_counter()-start>time_limit+30:
                proc.kill();proc.wait();break
            time.sleep(.005)
    elapsed=time.perf_counter()-start
    log=(folder/"solver.log").read_text(encoding="utf-8",errors="replace")
    r={"tag":tag,"case":d.name,"key":key,"binary":binary,"N":n,"T":t,"dt":d.dt,
       "variables":nv,"constraints":nr,"binaries":nt if binary else 0,
       "build_seconds":build_seconds,"process_seconds":elapsed,"sampled_peak_rss_mb":peak/2**20,
       "status":"no_solution","optimal":False,"has_solution":False,"objective":None,**certificate(d)}
    timings=re.findall(r"(?:Solution time|Total \(root\+branch&cut\))\s*=\s*([0-9.]+)\s*sec",log)
    r["solver_reported_seconds"]=float(timings[-1]) if timings else None
    sol=folder/"solution.sol"
    if sol.exists():
        root=ET.parse(sol).getroot();head=root.find("header").attrib
        r["status"]=head.get("solutionStatusString","");r["status_code"]=head.get("solutionStatusValue")
        r["optimal"]=r["status_code"] in {"1","101","102"}
        r["has_solution"]=True
        vals=np.zeros(nv)
        for v in root.find("variables"):vals[int(v.attrib["index"])]=float(v.attrib["value"])
        xx=vals[x];yy=vals[y];ee=vals[e]
        r.update(objective_components(d,xx,yy))
        r["solver_objective"]=float(head["objectiveValue"])
        r["objective_recompute_error"]=abs(r["objective"]-r["solver_objective"])
        r["feasibility_residual"]=feasibility_residual(d,xx,yy,ee,binary)
        gap_match=re.findall(r"gap\s*=\s*[^,\n]+,\s*([0-9.]+)%",log)
        r["mip_relative_gap"]=(float(gap_match[-1])/100 if gap_match else (0.0 if r["status_code"]=="101" else None)) if binary else None
        rr=cycle_repair(d,xx,yy)
        r["repair_feasibility_residual"]=feasibility_residual(d,*rr,True)
        r["repair_objective"]=objective_components(d,rr[0],rr[1])["objective"]
        r["x"]=xx.tolist();r["y"]=yy.tolist();r["e"]=ee.tolist()
        np.savez_compressed(folder/"schedule.npz",x=xx,y=yy,e=ee)
    elif "infeasible" in log.lower(): r["status"]="infeasible"
    elif "1016" in log: r["status"]="license_size_error"
    else:r["status"]="solver_error"
    resultfile.write_text(json.dumps(r,indent=2,allow_nan=False),encoding="utf-8")
    return r


def flat(r): return {k:v for k,v in r.items() if k not in {"x","y","e"}}


def pair(d,tag,time_limit=60):
    q=solve(d,False,tag+"_qp",time_limit);m=solve(d,True,tag+"_miqp",time_limit)
    r={"tag":tag,"case":d.name,"N":d.N,"T":d.T,"alpha":d.alpha,"rho":float(d.rho[0]),
       "capacity_mode":d.capacity_mode,**certificate(d)}
    for prefix,res in [("qp",q),("miqp",m)]:
        r.update({prefix+"_"+k:v for k,v in flat(res).items() if k not in {"tag","case","N","T"}})
    r["objective_gap"]=m["objective"]-q["objective"] if q["optimal"] and m["optimal"] else None
    if not q["has_solution"]:r["classification"]="relaxation_infeasible_or_error"
    elif m["status"]=="infeasible":r["classification"]="infeasibility_masked"
    elif not (q["optimal"] and m["optimal"]):r["classification"]="uncertified_solver_status"
    elif r["objective_gap"]>max(1e-5,1e-7*abs(m["objective"])):r["classification"]="positive_value_gap"
    elif q["max_simultaneous_kw"]>1e-5:r["classification"]="equal_value_simultaneous_solution"
    elif r["certificate"]:r["classification"]="certified_exact"
    else:r["classification"]="observed_exact_uncertified"
    return r


def fit_ecdf(training):
    a=np.asarray(training,float)
    if a.ndim!=2 or len(a)==0 or not np.isfinite(a).all():raise ValueError("finite nonempty training paths required")
    return np.sort(a,axis=0)


def apply_ecdf(fitted,paths):
    a=np.asarray(paths,float)
    return np.column_stack([np.searchsorted(fitted[:,j],a[:,j],side="right")/len(fitted) for j in range(a.shape[1])])


def conformal_quantile(scores,epsilon):
    a=np.sort(np.asarray(scores,float));n=len(a)
    if n==0 or not 0<epsilon<1:raise ValueError("nonempty scores and 0<epsilon<1 required")
    k=int(np.ceil((n+1)*(1-epsilon)))
    return float("inf") if k>n else float(a[k-1])


def uniform_paths(samples,T,correlation,seed):
    """Gaussian AR(1) copula: uniform marginals, latent lag-one correlation r."""
    rng=np.random.default_rng(seed);z=rng.normal(size=(samples,T))
    for t in range(1,T):z[:,t]=correlation*z[:,t-1]+np.sqrt(1-correlation**2)*z[:,t]
    return norm.cdf(z)


def wilson(successes,total):
    if total==0:return (float("nan"),float("nan"))
    p=successes/total;z=1.95996398454;den=1+z*z/total
    mid=(p+z*z/(2*total))/den
    half=z*np.sqrt(p*(1-p)/total+z*z/(4*total**2))/den
    return float(mid-half),float(mid+half)
