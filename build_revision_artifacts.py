"""Generate publication figures and LaTeX tables exclusively from recorded results."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
OUT=ROOT/"revision/results";TAB=ROOT/"revision/tables";FIG=ROOT/"revision/figures"
TAB.mkdir(exist_ok=True);FIG.mkdir(exist_ok=True)
COLORS={"Deterministic":"#0072B2","Slotwise Q75":"#E69F00","Robust":"#009E73",
        "Empirical 95%":"#CC79A7","Conformal 95%":"#D55E00","Rolling conformal 95%":"#6A51A3"}
plt.rcParams.update({"font.family":"DejaVu Sans","font.size":9,"axes.spines.top":False,
                     "axes.spines.right":False,"savefig.bbox":"tight","pdf.fonttype":42})


def table(name,headers,rows,align=None):
    align=align or "l"+"r"*(len(headers)-1)
    text="\\begin{tabular}{"+align+"}\n\\toprule\n"+" & ".join(headers)+" \\\\\n\\midrule\n"
    text+="\n".join(" & ".join(map(str,r))+" \\\\" for r in rows)
    text+="\n\\bottomrule\n\\end{tabular}\n"
    (TAB/f"{name}.tex").write_text(text,encoding="utf-8")


def figure(fig,name):
    fig.savefig(FIG/f"{name}.pdf");fig.savefig(FIG/f"{name}.png",dpi=170);plt.close(fig)


def main():
    hist=pd.read_csv(OUT/"historical.csv");sens=pd.read_csv(OUT/"sensitivity.csv")
    bounds=pd.read_csv(OUT/"boundaries.csv");cal=pd.read_csv(OUT/"calibration_summary.csv")
    corr=pd.read_csv(OUT/"correlation.csv");fleet=pd.read_csv(OUT/"fleet_reliability.csv");runtime=pd.read_csv(OUT/"runtime.csv")
    rows=[]
    for alpha,g in hist.groupby("alpha"):
        counts=g.classification.value_counts();rows.append([f"{alpha:g}",len(g),counts.get("certified_exact",0),
            counts.get("observed_exact_uncertified",0),counts.get("positive_value_gap",0),f"{max(0.0,g.objective_gap.max()):.6f}"])
    table("historical",[r"$\alpha$","Pairs","Certified","Observed only","Gap cases","Max gap (EUR)"],rows)
    bad=hist[hist.classification=="positive_value_gap"]
    table("historical_failures",["Date","QP (EUR)","MIQP (EUR)","Gap (EUR)","SCD (kW)"],
          [[r.date,f"{r.qp_objective:.4f}",f"{r.miqp_objective:.4f}",f"{r.objective_gap:.4f}",f"{r.qp_max_simultaneous_kw:.3f}"] for r in bad.itertuples()])
    cats=["certified_exact","observed_exact_uncertified","positive_value_gap","infeasibility_masked","relaxation_infeasible_or_error"]
    labels=["Certified","Observed only","Positive gap","Masked infeasibility","Both infeasible"]
    rows=[]
    for mode,g in bounds.iloc[:36].groupby("capacity_mode",sort=False):
        counts=g.classification.value_counts();rows.append([mode.replace("_"," "),*[counts.get(c,0) for c in cats]])
    table("boundaries",["Protection",*labels],rows)
    rows=[]
    for r in cal.itertuples():
        rows.append([r.policy.replace("%", r"\%"),int(r.violations),f"{100*r.path_coverage:.1f}",f"{r.objective:.6f}",f"{r.linear_cost:.6f}",f"{r.load_cost:.6f}"])
    table("calibration",["Policy","Events / 121","Envelope (\\%)","Full (EUR)","Linear (EUR)","Load (EUR)"],rows)
    c=corr[(corr.delta==.5)&(corr.demand_kwh==40)]
    table("correlation",["Policy",r"Latent $r$","Joint violation","95\\% MC interval","Independent formula"],
          [[r.policy,f"{r.latent_ar1:.1f}",f"{r.violation_rate:.4f}",f"[{r.ci_low:.4f}, {r.ci_high:.4f}]",f"{r.independent_formula:.4f}"] for r in c.itertuples()],"lrrrr")
    fs=fleet.groupby(["gamma_percent","policy"],sort=False).agg(n=("objective","size"),violation=("violation_rate","mean"),
        violation_sd=("violation_rate","std"),objective=("objective","mean"),objective_sd=("objective","std"),linear=("linear_cost","mean"),market=("market_cost","mean"),throughput=("throughput_cost","mean"),load=("load_cost","mean")).reset_index()
    fs.to_csv(OUT/"fleet_summary.csv",index=False)
    table("fleet",[r"$\gamma$ (\%)","Policy","Violation mean (SD)","Full (EUR)","Linear (EUR)"],
          [[f"{r.gamma_percent:.0f}",r.policy,f"{r.violation:.3f} ({r.violation_sd:.3f})",f"{r.objective:.3f}",f"{r.linear:.3f}"] for r in fs.itertuples()],"rlrrr")
    rt=runtime.groupby(["N","T"]).agg(samples=("N","size"),qp=("qp_process_seconds","mean"),miqp=("miqp_process_seconds","mean"),
        qp_sd=("qp_process_seconds","std"),miqp_sd=("miqp_process_seconds","std"),qp_mb=("qp_sampled_peak_rss_mb","mean"),miqp_mb=("miqp_sampled_peak_rss_mb","mean")).reset_index()
    rt["ratio"]=rt.miqp/rt.qp;rt.to_csv(OUT/"runtime_summary.csv",index=False)
    small=rt[(rt.N==20)&(rt["T"]==24)].iloc[0];large=rt[(rt.N==200)&(rt["T"]==96)].iloc[0]
    macros={"SmallQP":f"{small.qp:.3f}","SmallMIQP":f"{small.miqp:.3f}","LargeQP":f"{large.qp:.3f}",
            "LargeMIQP":f"{large.miqp:.3f}","LargeQPMemory":f"{large.qp_mb:.1f}","LargeMIQPMemory":f"{large.miqp_mb:.1f}",
            "LargestGap":f"{runtime.objective_gap.abs().max():.3g}"}
    (TAB/"result_macros.tex").write_text("\n".join("\\newcommand{\\"+k+"}{"+v+"}" for k,v in macros.items()))
    table("runtime",["EVs","Slots","Binaries","QP (s)","MIQP (s)","Ratio","QP RSS","MIQP RSS"],
          [[r.N,r.T,r.N*r.T,f"{r.qp:.3f}",f"{r.miqp:.3f}",f"{r.ratio:.2f}",f"{r.qp_mb:.1f}",f"{r.miqp_mb:.1f}"] for r in rt.itertuples()])
    dg=pd.read_csv(OUT/"degradation.csv").groupby(["rho","policy"]).agg(objective=("objective","mean"),linear=("linear_cost","mean"),discharge=("discharge_kwh","mean"),violation=("violation_rate","mean")).reset_index()
    dg.to_csv(OUT/"degradation_summary.csv",index=False)
    table("degradation",[r"$\rho$ (EUR/kWh)","Policy","Full (EUR)","Linear (EUR)","V2G (kWh)","Violation"],
          [[f"{r.rho:g}",r.policy,f"{r.objective:.3f}",f"{r.linear:.3f}",f"{r.discharge:.3f}",f"{r.violation:.3f}"] for r in dg.itertuples()],"rlrrrr")
    # Figure 1: historical failure and price/spread sensitivity, including zero-penalty boundary.
    fig,axs=plt.subplots(1,3,figsize=(12,3.3),layout="constrained")
    h=hist[(hist.date=="2019-06-08")&(hist.alpha==0)].iloc[0]
    q=json.loads((ROOT/"revision/solves"/h.qp_key/"result.json").read_text());m=json.loads((ROOT/"revision/solves"/h.miqp_key/"result.json").read_text())
    for ax,res,title in zip(axs[:2],[q,m],["(a) Relaxed QP","(b) Physical MIQP"]):
        ax.step(np.arange(24),np.array(res["x"])[0],where="mid",label="Charging",color="#0072B2")
        ax.step(np.arange(24),np.array(res["y"])[0],where="mid",label="Discharging",color="#D55E00")
        ax.set(xlabel="UTC hour",ylabel="Power (kW)",title=title,xlim=(8,22),ylim=(-1,31));ax.legend(fontsize=8)
    grid=sens.assign(failed=sens.classification=="positive_value_gap").pivot_table(index="rho",columns="spread",values="failed",aggfunc="mean")
    im=axs[2].imshow(grid,origin="lower",aspect="auto",vmin=0,vmax=1,cmap="YlOrRd")
    axs[2].set(xticks=range(len(grid.columns)),xticklabels=[f"{x:g}" for x in grid.columns],yticks=range(len(grid.index)),yticklabels=[f"{x:g}" for x in grid.index],xlabel="Selling / buying price multiplier",ylabel=r"$\rho$ (EUR/kWh)",title="(c) Failure fraction over 8 stress days")
    for i in range(len(grid)):
        for j in range(len(grid.columns)):axs[2].text(j,i,f"{grid.iloc[i,j]*100:.0f}%",ha="center",va="center",fontsize=8)
    figure(fig,"figure1_exactness")
    fig,axs=plt.subplots(1,2,figsize=(9,3.2),layout="constrained")
    for ax,demand in zip(axs,[10,40]):
        for label in COLORS:
            g=corr[(corr.delta==.5)&(corr.demand_kwh==demand)&(corr.policy==label)]
            if g.empty:continue
            ax.errorbar(g.latent_ar1,g.violation_rate,yerr=np.array([g.violation_rate-g.ci_low,g.ci_high-g.violation_rate]),marker="o",label=label,color=COLORS[label],capsize=3)
        ax.set(xlabel="Latent lag-one correlation",ylabel="Joint violation probability",title=f"Required battery increment: {demand} kWh",ylim=(-.02,.8));ax.legend(fontsize=8)
    figure(fig,"figure2_correlation")
    fig,axs=plt.subplots(1,2,figsize=(10,3.5),layout="constrained")
    labels=cal.policy.str.replace("Rolling conformal 95%","Rolling\nconformal").str.replace("Conformal 95%","Conformal").str.replace("Empirical 95%","Empirical").str.replace("Slotwise Q75","Q75")
    axs[0].bar(np.arange(len(cal)),100*cal.path_coverage,color=[COLORS[x] for x in cal.policy]);axs[0].axhline(95,color="black",linestyle="--",linewidth=1,label="Nominal 95%")
    axs[0].set(xticks=np.arange(len(cal)),xticklabels=labels,ylabel="Held-out envelope coverage (%)",ylim=(0,104));axs[0].tick_params(axis="x",labelrotation=35);axs[0].legend(fontsize=8)
    monthly=pd.read_csv(OUT/"calibration_monthly.csv")
    for label in ["Conformal 95%","Rolling conformal 95%","Robust"]:
        g=monthly[monthly.policy==label];axs[1].plot(g.month,100*g.path_coverage,marker="o",color=COLORS[label],label=label)
    axs[1].axhline(95,color="black",linestyle="--",linewidth=1);axs[1].set(ylabel="Monthly envelope coverage (%)",ylim=(35,103),xlabel="Test month");axs[1].legend(fontsize=8)
    figure(fig,"figure3_calibration")
    fig,axs=plt.subplots(1,2,figsize=(9,3.3),layout="constrained")
    pivot=fleet.pivot(index=["seed","gamma_percent"],columns="policy",values="objective")
    for label in ["Deterministic","Slotwise Q75","Robust"]:
        g=fs[fs.policy==label].sort_values("gamma_percent")
        axs[0].plot(g.gamma_percent,g.violation,marker="o",color=COLORS[label],label=label)
        axs[0].fill_between(g.gamma_percent,np.maximum(0,g.violation-g.violation_sd),np.minimum(1,g.violation+g.violation_sd),color=COLORS[label],alpha=.15)
        premium=(pivot[label]-pivot["Deterministic"]).groupby("gamma_percent").agg(["mean","std"])
        axs[1].plot(premium.index,premium["mean"],marker="o",color=COLORS[label],label=label)
        axs[1].fill_between(premium.index,premium["mean"]-premium["std"],premium["mean"]+premium["std"],color=COLORS[label],alpha=.15)
    axs[0].set(ylabel="Joint violation probability",ylim=(-.02,1));axs[1].set(ylabel="Full-objective premium (EUR/day)")
    for ax in axs:ax.set(xlabel=r"Capacity margin $\gamma$ (%)",xticks=[0,5,10,15]);ax.legend(fontsize=8)
    figure(fig,"figure4_fleet")
    fig,axs=plt.subplots(1,2,figsize=(9,3.3),layout="constrained")
    for T,style in [(24,"--"),(96,"-")]:
        g=rt[rt["T"]==T]
        for kind,color in [("qp","#0072B2"),("miqp","#D55E00")]:
            axs[0].plot(g.N,g[kind],marker="o",ls=style,color=color,label=f"{kind.upper()}, T={T}")
            axs[1].plot(g.N,g[kind+"_mb"],marker="o",ls=style,color=color,label=f"{kind.upper()}, T={T}")
    axs[0].set(ylabel="Mean solver-process wall time (s)",yscale="log");axs[1].set(ylabel="Mean sampled peak solver RSS (MiB)")
    for ax in axs:ax.set(xlabel="Number of EVs",xticks=[5,20,50,100,200]);ax.legend(fontsize=8)
    figure(fig,"figure5_runtime")
    summary={"historical_pairs":len(hist),"historical_gap_cases":int((hist.classification=="positive_value_gap").sum()),
             "historical_max_gap":float(hist.objective_gap.max()),"sensitivity_pairs":len(sens),"sensitivity_gap_cases":int((sens.classification=="positive_value_gap").sum()),
             "runtime_pairs":len(runtime),"runtime_max_abs_gap":float(runtime.objective_gap.abs().max()),
             "runtime_all_optimal":bool(runtime.qp_optimal.all() and runtime.miqp_optimal.all()),
             "max_feasibility_residual":float(max(runtime.qp_feasibility_residual.max(),runtime.miqp_feasibility_residual.max())),
             "calibration_days":121,"fleet_runs":len(fleet)}
    (OUT/"paper_summary.json").write_text(json.dumps(summary,indent=2))
    print(json.dumps(summary,indent=2))


if __name__=="__main__":main()
