"""Paired native IBM CPLEX QP and MIQP using the archived instance functions."""
import argparse,sys,json,hashlib,platform
from pathlib import Path
import cplex,numpy as np,pandas as pd
parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('--project-root', type=Path, required=True, help='SEGAN folder with revision/original, data and outputs')
parser.add_argument('--output-dir', type=Path, default=Path(__file__).resolve().parent, help='Destination for CSV and metadata')
args=parser.parse_args()
scratch=args.output_dir.resolve()
scratch.mkdir(parents=True,exist_ok=True)
project=args.project_root.resolve()
orig=project/'revision/original/ev_robust_charging.py'
profile=project/'data/opsd_de_lu_2019_01_24_profile.csv'
sys.path.insert(0,str(orig.parent))
import ev_robust_charging as ev
def load_profile(date):
    assert date=='2019-01-24'
    return pd.read_csv(profile)
ev.load_public_profile=load_profile

def solve(d,coef,binary):
    n=d.T
    with cplex.Cplex() as c:
        for setter in (c.set_log_stream,c.set_results_stream,c.set_warning_stream): setter(None)
        p=c.parameters
        p.threads.set(1);p.timelimit.set(60.0)
        p.simplex.tolerances.feasibility.set(1e-9);p.simplex.tolerances.optimality.set(1e-9)
        p.barrier.convergetol.set(1e-11)
        p.mip.tolerances.integrality.set(1e-9);p.mip.tolerances.mipgap.set(1e-10);p.mip.tolerances.absmipgap.set(1e-9)
        x=list(range(n));y=list(range(n,2*n));e=list(range(2*n,3*n+1));load=list(range(3*n+1,4*n+1))
        a=d.availability
        c.variables.add(obj=[*(d.price_buy+d.rho),*(-d.price_sell+d.rho),*np.zeros(2*n+1)],
            lb=[*np.zeros(2*n),d.E0,*np.full(n,d.Emin),*np.full(n,-cplex.infinity)],
            ub=[*(a*d.X),*(a*d.Y),d.E0,*np.full(n,d.Emax),*np.full(n,cplex.infinity)])
        if binary:
            z=list(range(4*n+1,5*n+1))
            c.variables.add(types='B'*n)
        c.objective.set_quadratic_coefficients([(i,i,float(2*d.alpha)) for i in load])
        def row(ids,coefficients,sense,rhs):
            c.linear_constraints.add(lin_expr=[cplex.SparsePair(ind=ids,val=list(map(float,coefficients)))],senses=sense,rhs=[float(rhs)])
        row([e[-1]],[1],'G',d.Etarget)
        for t in range(n):
            row([e[t+1],e[t],x[t],y[t]],[1,-1,-d.eta_c,1/d.eta_d],'E',0)
            row([load[t],x[t],y[t]],[1,-1,1],'E',d.L[t])
            row([x[t]],[coef[t]],'L',d.M-d.L[t])
            if binary:
                row([x[t],z[t]],[1,-d.X*a[t]],'L',0)
                row([y[t],z[t]],[1,d.Y*a[t]],'L',d.Y*a[t])
        c.solve(); s=c.solution
        assert s.get_status() in (1,101,102),s.get_status_string()
        xx=np.array(s.get_values(x)); yy=np.array(s.get_values(y)); E=np.array(s.get_values(e))
        F=float(np.sum(d.price_buy*xx-d.price_sell*yy+d.rho*(xx+yy)+d.alpha*(d.L+xx-yy)**2))
        residual=max(0.0,np.max(np.abs(np.diff(E)-d.eta_c*xx+yy/d.eta_d)),np.max(d.L+coef*xx-d.M),np.max(xx-d.X*a),np.max(yy-d.Y*a),np.max(d.Emin-E[1:]),np.max(E[1:]-d.Emax),d.Etarget-E[-1],-xx.min(),-yy.min())
        return dict(status=s.get_status_string(),code=s.get_status(),F=F,solverF=s.get_objective_value(),simultaneous=max(0,float(np.minimum(xx,yy).max())),residual=float(residual),x=xx,y=yy)
cases=ev.build_miqp_validation_cases(ev.build_base_case())
checks=pd.read_csv(project/'outputs/q1_binary_complementarity_summary.csv').set_index(['case','validation_policy'])
rows=[]
for d in cases:
    for pol in [p for p in ev.POLICIES if p.name in ('deterministic','quantile_75','robust')]:
        coef=ev.policy_u(d,0.5,pol)
        q=solve(d,coef,False); m=solve(d,coef,True)
        ref=checks.loc[(d.name,pol.name)]
        r=dict(case=d.name,T=d.T,policy=pol.name,M=d.M,target=d.Etarget,gap=abs(m['F']-q['F']),charge_l1=float(np.abs(q['x']-m['x']).sum()),discharge_l1=float(np.abs(q['y']-m['y']).sum()),old_qp_F=ref.relaxed_objective,vs_old_qp=abs(q['F']-ref.relaxed_objective))
        for prefix,sol in [('qp',q),('miqp',m)]:
            r.update({prefix+'_'+k:v for k,v in sol.items() if k not in ('x','y')})
        rows.append(r)
df=pd.DataFrame(rows)
df.to_csv(scratch/'original_table6_cplex_pairs.csv',index=False)
assert len(df)==12 and df.vs_old_qp.max()<1e-4
with cplex.Cplex() as c: version=c.get_version()
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
meta=dict(version=version,python=sys.version,archive_hash=sha(orig),profile_hash=sha(profile),pairs=len(df),qp_status=list(df.qp_status.unique()),miqp_status=list(df.miqp_status.unique()),maxima=df[['gap','charge_l1','discharge_l1','qp_simultaneous','miqp_simultaneous','qp_residual','miqp_residual','vs_old_qp']].max().to_dict())
meta.update(cplex_threads=1,time_limit_seconds=60,feasibility_tolerance=1e-9,optimality_tolerance=1e-9,
    integrality_tolerance=1e-9,barrier_convergence_tolerance=1e-11,relative_mip_gap=1e-10,absolute_mip_gap=1e-9,
    objective='Sum of market payment, grid throughput penalty and squared net station load per hourly slot',
    paired_gap='Absolute difference of full objectives recomputed from raw charging and discharging values',
    overlap='Max over slots of min(charge,discharge), clipped below at zero, in kW',
    source_paths={'archive':str(orig.relative_to(project)),'profile':str(profile.relative_to(project)),'archive_comparison':'outputs/q1_binary_complementarity_summary.csv'},
    solver_type='Native CPLEX Python API; convex QP and MIQP with physical binary bounds',delta=0.5,archive_comparison_sha256=sha(project/'outputs/q1_binary_complementarity_summary.csv'),
    script_sha256=sha(Path(__file__)),run_utc=__import__('datetime').datetime.now(__import__('datetime').timezone.utc).isoformat())
print(json.dumps(meta))
(scratch/'original_table6_cplex_metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf8')

