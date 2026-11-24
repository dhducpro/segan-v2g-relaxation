"""Reproduce the revised manuscript. Each solve is content-addressed and resumable."""
from __future__ import annotations
import argparse
from dataclasses import replace
from pathlib import Path
import hashlib
import json
import platform
import sys
import time

import numpy as np
import pandas as pd
import psutil
from scipy.optimize import linprog
from scipy import sparse
import ev_robust_charging as legacy
import revision_model as model

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"revision/results"
OUT.mkdir(parents=True,exist_ok=True)
SEED=20261112
POLICIES={"Deterministic":1.,"Slotwise Q75":1.25,"Robust":1.5}


def save(rows,name):
    df=pd.DataFrame(rows);df.to_csv(OUT/f"{name}.csv",index=False);return df


def days():
    frame=pd.read_csv(ROOT/"data/opsd_de_lu_2019.csv")
    frame["date"]=frame.utc_timestamp.str[:10]
    groups={d:g for d,g in frame.groupby("date") if len(g)==24 and not g[legacy.OPSD_COLUMNS].isna().any().any()}
    return frame,groups


def day_instance(date,raw=True,spread=.98):
    d=model.from_legacy(legacy.build_case_from_profile(date))
    if raw:
        prices=legacy.load_public_profile(date).raw_price_eur_per_mwh.to_numpy()/1000.
        d=replace(d,buy=prices,sell=spread*prices)
    return d


def price_audit():
    df,groups=days();p=df.DE_LU_price_day_ahead
    eligible=np.concatenate([g.DE_LU_price_day_ahead.to_numpy()[10:21] for g in groups.values()])
    info={"all_rows":len(df),"nonmissing_prices":int(p.notna().sum()),"negative_hours":int((p<0).sum()),
          "below_1_eur_mwh_hours":int((p<1).sum()),"complete_days":len(groups),
          "negative_price_complete_days":int(sum((g.DE_LU_price_day_ahead<0).any() for g in groups.values())),
          "connected_negative_hours":int((eligible<0).sum()),"connected_hours":len(eligible),
          "negative_price_days_connected":int(sum((g.DE_LU_price_day_ahead.iloc[10:21]<0).any() for g in groups.values())),
          "min_price_eur_mwh":float(p.min()),"max_price_eur_mwh":float(p.max())}
    (OUT/"price_audit.json").write_text(json.dumps(info,indent=2));print(info,flush=True)


def historical():
    _,groups=days();rows=[]
    for j,date in enumerate(groups):
        base=day_instance(date)
        for alpha in [0.,2e-4]:
            d=replace(base,alpha=alpha)
            row=model.pair(d,f"historical_{date}_{alpha}")
            row.update(date=date,negative_connected=int((d.buy[d.availability[0]>0]<0).sum()))
            rows.append(row)
        if (j+1)%20==0:save(rows,"historical");print(f"historical {j+1}/{len(groups)}",flush=True)
    save(rows,"historical")


def sensitivity():
    _,groups=days()
    # Predetermined stress sample: eight days with lowest connected-window prices.
    ranked=sorted(groups,key=lambda d:groups[d].DE_LU_price_day_ahead.iloc[10:21].min())
    selected=ranked[:8]
    rows=[]
    for date in selected:
        base=day_instance(date)
        for spread in [.90,.98,1.,1.02,1.10]:
            for rho in [0.,.001,.0023,.0045,.01,.05]:
                d=replace(base,sell=spread*base.buy,rho=np.array([rho]),alpha=0.)
                row=model.pair(d,f"sensitivity_{date}_{spread}_{rho}")
                row.update(date=date,spread=spread);rows.append(row)
        print(f"sensitivity {date}",flush=True);save(rows,"sensitivity")


def boundaries():
    rows=[]
    base=day_instance("2019-01-24")
    h=np.arange(24)
    for pv in [0.,5.,15.,30.]:
        L=8.+3*np.cos(2*np.pi*(h-19)/24)-pv*np.maximum(0,np.sin(np.pi*(h-6)/12))
        for mode in ["charging","net_import","two_sided"]:
            for alpha in [0.,2e-4,.01]:
                d=replace(base,name=f"pv_{pv}_{mode}_{alpha}",L=L,capacity=45.,capacity_mode=mode,
                          export_limit=5.,alpha=alpha,pcmax=np.array([15.]),pdmax=np.array([15.]),
                          availability=np.ones((1,24)))
                row=model.pair(d,d.name);row["pv_peak_kw"]=pv;rows.append(row)
    # One-slot analytical controls isolate economics and hard export feasibility.
    common=dict(name="one_slot",L=np.array([0.]),buy=np.array([1.]),sell=np.array([1.]),
                availability=np.ones((1,1)),pcmax=np.array([10.]),pdmax=np.array([10.]),e0=np.array([5.]),
                emin=np.array([0.]),emax=np.array([5.]),target=np.array([5.]),etac=np.array([.95]),
                etad=np.array([.9]),rho=np.array([0.]),coeff=np.ones((1,1)),capacity=100.,alpha=0.)
    d=model.Instance(**common)
    cases=[replace(d,name="positive_zero_penalty"),replace(d,name="negative_price",buy=np.array([-1.]),sell=np.array([-1.])),
           replace(d,name="export_infeasibility",L=np.array([-1.]),capacity_mode="two_sided",rho=np.array([.01])),
           replace(d,name="surplus_smoothing",L=np.array([-1.]),buy=np.array([0.]),sell=np.array([0.]),alpha=1.,rho=np.array([.001]))]
    for d in cases:
        row=model.pair(d,d.name);row["pv_peak_kw"]=-float(d.L.min());rows.append(row)
    save(rows,"boundaries")


def calibration():
    _,groups=days();dates=list(groups)
    raw=np.vstack([g.DE_LU_load_actual_entsoe_transparency.to_numpy() for g in groups.values()])
    train_n=96;cal_end=240
    fitted=model.fit_ecdf(raw[:train_n]);ranks=model.apply_ecdf(fitted,raw)
    paths=1.5+.75*ranks
    cal=paths[train_n:cal_end];test=paths[cal_end:];scores=((cal-1.5)/.75).max(axis=1)
    fixedq=model.conformal_quantile(scores,.05)
    split={"training_days":dates[:train_n],"calibration_days":dates[train_n:cal_end],"test_days":dates[cal_end:],
           "training_n":train_n,"calibration_n":len(cal),"test_n":len(test),"hat_u":1.5,"scale_d":.75,
           "conformal_epsilon":.05,"order_statistic":int(np.ceil((len(cal)+1)*.95)),"fixed_q":fixedq,
           "interpretation":"Synthetic coefficient proxy; temporal test coverage is empirical, not exchangeability-guaranteed."}
    (OUT/"calibration_split.json").write_text(json.dumps(split,indent=2))
    np.savez_compressed(OUT/"calibration_paths.npz",training_raw=raw[:train_n],fitted_ecdf=fitted,
                        calibration=cal,test=test,dates=np.array(dates))
    rows=[]
    for j,date in enumerate(dates[cal_end:]):
        # Controlled positive-price station arm isolates reliability from exactness failure.
        base=day_instance(date,raw=False)
        previous=paths[train_n:cal_end+j][-144:]
        rollingq=model.conformal_quantile(((previous-1.5)/.75).max(axis=1),.05)
        coeffs={"Deterministic":np.full(24,1.5),"Slotwise Q75":np.full(24,1.875),
                "Empirical 95%":np.quantile(cal,.95,axis=0),"Conformal 95%":np.full(24,1.5+.75*fixedq),
                "Rolling conformal 95%":np.full(24,1.5+.75*rollingq),"Robust":np.full(24,2.25)}
        for label,coeff in coeffs.items():
            d=replace(base,coeff=coeff[None,:]);sol=model.solve(d,tag=f"cal_{date}_{label}")
            row={**model.flat(sol),"date":date,"policy":label,"month":date[:7],
                 "path_covered":bool(np.all(test[j]<=coeff+1e-12)),"protection_mean":float(coeff.mean())}
            if sol["has_solution"]:
                excess=base.L+test[j]*np.array(sol["x"])[0]-base.capacity
                row.update(violation=int(excess.max()>1e-6),max_excess=max(0,float(excess.max())))
            rows.append(row)
        if (j+1)%20==0:save(rows,"calibration");print(f"calibration {j+1}/{len(test)}",flush=True)
    df=save(rows,"calibration")
    # Moving-block bootstrap across test days; descriptive uncertainty, no coverage theorem.
    summary=[];rng=np.random.default_rng(SEED);n=len(test);block=7
    indices=np.concatenate([(rng.integers(0,n,size=(2000,1))+np.arange(block))%n for _ in range(int(np.ceil(n/block)))],axis=1)[:,:n]
    for label,g in df.groupby("policy",sort=False):
        g=g.sort_values("date");v=g.violation.to_numpy(float)
        ci=np.quantile(v[indices].mean(axis=1),[.025,.975])
        summary.append({"policy":label,"n":len(g),"violations":int(v.sum()),"violation_rate":v.mean(),
                        "block_ci_low":ci[0],"block_ci_high":ci[1],"path_coverage":g.path_covered.mean(),
                        **{col:g[col].mean() for col in ["objective","market_cost","throughput_cost","linear_cost","load_cost","discharge_kwh","protection_mean"]},
                        "max_excess":g.max_excess.max()})
    save(summary,"calibration_summary")
    df.groupby(["month","policy"]).agg(n=("violation","size"),violation_rate=("violation","mean"),path_coverage=("path_covered","mean")).reset_index().to_csv(OUT/"calibration_monthly.csv",index=False)


def correlation():
    base=day_instance("2019-01-24",raw=False);rows=[]
    for demand in [10.,40.]:
      base=replace(base,e0=np.array([60.-demand]))
      for delta in [.1,.3,.5]:
        for label,q in [("Deterministic",.5),("Slotwise Q75",.75),("Robust",1.)]:
            coeff=base.coeff*(1-delta+2*delta*q)
            d=replace(base,coeff=coeff);sol=model.solve(d,tag=f"corr_{delta}_{label}")
            if not sol["optimal"]:raise RuntimeError(sol["status"])
            x=np.array(sol["x"])[0]
            limit=np.full(24,np.inf);np.divide(base.capacity-base.L,x,out=limit,where=x>1e-8)
            phi=np.clip((limit-1.5*(1-delta))/(3*delta),0,1)
            independent_exact=1-np.prod(phi)
            for corr in [0.,.5,.9]:
                u=1.5*(1-delta+2*delta*model.uniform_paths(20000,24,corr,SEED))
                violation=(base.L+u*x-base.capacity).max(axis=1)>1e-6
                lo,hi=model.wilson(int(violation.sum()),len(violation))
                rows.append({"key":sol["key"],"policy":label,"demand_kwh":demand,"delta":delta,"latent_ar1":corr,"samples":len(u),"violation_rate":violation.mean(),
                             "ci_low":lo,"ci_high":hi,"independent_formula":independent_exact,**model.objective_components(d,np.array(sol["x"]),np.array(sol["y"]))})
    save(rows,"correlation")


def degradation():
    _,groups=days();selected=[]
    for month in range(1,13):selected.append(next(d for d in groups if int(d[5:7])==month))
    rows=[]
    for date in selected:
        base=day_instance(date,raw=False)
        for rho in [0.,.001,.0023,.0045,.01,.05]:
            for label,mult in POLICIES.items():
                d=replace(base,rho=np.array([rho]),coeff=base.coeff*mult)
                sol=model.solve(d,tag=f"degradation_{date}_{rho}_{label}")
                row={**model.flat(sol),"date":date,"rho":rho,"policy":label}
                if sol["has_solution"]:
                    u=1.5*(.5+model.uniform_paths(10000,24,.7,SEED))
                    row["violation_rate"]=float(((d.L+u*np.array(sol["x"])[0]-d.capacity).max(axis=1)>1e-6).mean())
                rows.append(row)
        save(rows,"degradation")


def exchangeability():
    hits=[]
    # Independent daily paths; within each path coefficients remain temporally correlated.
    for j in range(2000):
        u=model.uniform_paths(145,24,.7,SEED+j)
        q=model.conformal_quantile(u[:144].max(axis=1),.05)
        hits.append(u[-1].max()<=q)
    lo,hi=model.wilson(sum(hits),len(hits))
    (OUT/"exchangeable_control.json").write_text(json.dumps({"replications":len(hits),"calibration_n":144,
        "path_length":24,"latent_ar1":.7,"coverage":float(np.mean(hits)),"ci_low":lo,"ci_high":hi,
        "rank":138,"finite_sample_lower_bound":138/145},indent=2))


def acn_pool():
    """Fully contained sessions, exact fractional hourly availability; observed energy is AC-side."""
    s=legacy.load_acn_sessions("caltech","2019-01-01","2019-03-31",2.,ROOT/"caltech.csv")
    st=legacy.parse_acn_timestamp(s.connectionTime,"America/Los_Angeles")
    en=legacy.parse_acn_timestamp(s.disconnectTime,"America/Los_Angeles")
    same=(st.dt.date==en.dt.date)&(en>st)&st.notna()&en.notna()
    pool=[]
    for i in np.flatnonzero(same.to_numpy()):
        a=(st.iloc[i]-st.iloc[i].normalize()).total_seconds()/3600
        b=(en.iloc[i]-en.iloc[i].normalize()).total_seconds()/3600
        av=np.maximum(0,np.minimum(np.arange(24)+1,b)-np.maximum(np.arange(24),a))
        req=float(s.kWhDelivered.iloc[i])
        if req<=7.2*av.sum()+1e-9:
            pool.append({"session_id":str(s.session_id.iloc[i]),"date":str(st.iloc[i].date()),"arrival":a,"departure":b,"ac_energy":req,"availability":av})
    return pool,{"filtered_source_sessions":len(s),"same_day_feasible_pool":len(pool),"energy_conversion":"target=e0+0.95*kWhDelivered (AC-side delivered energy)"}


def fleet(N,seed,T=24,*,within_day=False):
    pool,info=acn_pool();rng=np.random.default_rng(seed)
    if within_day:
        counts=pd.Series([p["date"] for p in pool]).value_counts();valid=sorted(counts[counts>=N].index)
        date=valid[(seed-SEED)%len(valid)];pool=[p for p in pool if p["date"]==date]
        chosen=rng.choice(len(pool),N,replace=False)
    else:
        date="resampled_Q1";chosen=rng.choice(len(pool),N,replace=N>len(pool))
    picked=[pool[i] for i in chosen]
    base=day_instance("2019-01-24",raw=False)
    req=.95*np.array([p["ac_energy"] for p in picked]);emax=np.maximum(55,req/.5)
    e0=.4*emax;target=e0+req
    factor=T//24
    if T%24:raise ValueError("T must be a multiple of 24")
    av=np.zeros((N,T))
    for i,p in enumerate(picked):
        h=np.arange(T)/factor
        av[i]=np.maximum(0,np.minimum(h+1/factor,p["departure"])-np.maximum(h,p["arrival"]))*factor
    coeff=np.repeat(base.coeff,T//24,axis=1)*(0.95+.1*((np.arange(N)%7)/6))[:,None]*np.linspace(.97,1.03,T)[None,:]
    d=model.Instance(f"fleet_{N}_{T}_{seed}_{date}",np.repeat(base.L,factor),np.repeat(base.buy,factor),np.repeat(base.sell,factor),av,
                     np.full(N,7.2),np.full(N,4.),e0,.1*emax,emax,target,np.full(N,.95),np.full(N,.90),np.full(N,.001),coeff,
                     float(max(base.L)+4*N),alpha=2e-4,dt=24/T)
    metadata={"case":d.name,"seed":seed,"N":N,"T":T,"date":date,"source":"same_day_sessions" if within_day else "resampled_session_templates",**info,
              "sessions":[{k:v for k,v in p.items() if k!="availability"} for p in picked]}
    (ROOT/"revision/instances"/f"{d.name}.json").write_text(json.dumps({"metadata":metadata,"instance":model.serializable(d)},indent=2))
    return d


def minimum_capacity(d):
    n,t=d.N,d.T;nt=n*t
    # Charging-only feasibility LP: for charging-side protection discharging cannot reduce required capacity.
    ids=np.arange(nt).reshape(n,t);rr=[];cc=[];vv=[];b=[]
    for i in range(n):
        rr.extend([i]*t);cc.extend(ids[i]);vv.extend([-d.dt*d.etac[i]]*t);b.append(-(d.target[i]-d.e0[i]))
    for h in range(t):
        rr.extend([n+h]*n+[n+h]);cc.extend([*ids[:,h],nt]);vv.extend([*d.coeff[:,h],-1]);b.append(-d.L[h])
    A=sparse.coo_matrix((vv,(rr,cc)),shape=(n+t,nt+1)).tocsr()
    bounds=list(zip(np.zeros(nt),(d.pcmax[:,None]*d.availability).ravel()))+[(float(d.L.max()),None)]
    res=linprog(np.r_[np.zeros(nt),1],A_ub=A,b_ub=b,bounds=bounds,method="highs")
    if not res.success:raise RuntimeError(res.message)
    return float(res.x[-1])


def fleet_reliability():
    rows=[]
    for j in range(20):
        seed=SEED+j;base=fleet(20,seed,within_day=True)
        mmin=minimum_capacity(replace(base,coeff=1.5*base.coeff))
        # Shared station disturbance across EVs, temporally correlated; common random numbers.
        noise=model.uniform_paths(10000,24,.7,seed)
        for gamma in [0.,.05,.10,.15]:
            for label,mult in POLICIES.items():
                d=replace(base,capacity=mmin*(1+gamma)+1e-5,coeff=base.coeff*mult)
                sol=model.solve(d,tag=f"fleet_rel_{seed}_{gamma}_{label}")
                row={**model.flat(sol),"seed":seed,"gamma_percent":100*gamma,"policy":label,"robust_min_capacity":mmin,"capacity":d.capacity}
                if sol["has_solution"]:
                    x=np.array(sol["x"]);impact=(base.coeff*x).sum(axis=0)
                    exceed=(base.L+(0.5+noise)*impact-d.capacity).max(axis=1)
                    row.update(violation_rate=float((exceed>1e-6).mean()),max_excess=float(max(0,exceed.max())))
                rows.append(row)
        save(rows,"fleet_reliability");print(f"fleet reliability {j+1}/20",flush=True)


def runtime():
    rows=[]
    for N in [5,20,50,100,200]:
        for T in [24,96]:
            for rep in range(3):
                d=fleet(N,SEED+rep,T)
                # Scale background load with fleet size so all sizes satisfy the same sufficient regime.
                scale=max(1,N/20)
                d=replace(d,L=d.L*scale,capacity=max(d.L)*scale+4*N)
                row=model.pair(d,f"runtime_{N}_{T}_{rep}",time_limit=60)
                row["replicate"]=rep;rows.append(row)
                save(rows,"runtime");print(f"runtime N={N},T={T},rep={rep}: {row['miqp_status']}",flush=True)


def manifest():
    files=["revision_model.py","run_revision_experiments.py","ev_robust_charging.py","caltech.csv","data/opsd_de_lu_2019.csv"]
    info={"created_utc":pd.Timestamp.now(tz="UTC").isoformat(),"python":sys.version,"platform":platform.platform(),
          "processor":platform.processor(),"logical_cpus":psutil.cpu_count(),"ram_gb":psutil.virtual_memory().total/2**30,
          "cplex_exe":str(model.CPLEX_EXE),"cplex_version":"22.1.2.0","threads":1,"seed":SEED,
          "sha256":{p:hashlib.sha256((ROOT/p).read_bytes()).hexdigest() for p in files}}
    (OUT/"run_manifest.json").write_text(json.dumps(info,indent=2))


STAGES={"audit":price_audit,"historical":historical,"sensitivity":sensitivity,"boundaries":boundaries,
        "calibration":calibration,"correlation":correlation,"fleet":fleet_reliability,"runtime":runtime,
        "degradation":degradation,"exchangeability":exchangeability}
if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stages",nargs="+",default=list(STAGES),choices=list(STAGES));args=parser.parse_args()
    manifest()
    for name in args.stages:
        start=time.perf_counter();print(f"START {name}",flush=True);STAGES[name]();print(f"DONE {name} {time.perf_counter()-start:.1f}s",flush=True)
