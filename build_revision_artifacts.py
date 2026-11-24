"""Build the paper's tables and figures from a completed experiment run."""

import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
TAB = OUT / "tables"
FIG = OUT / "figures"
SOLVES = ROOT / "runs" / "solves"
FLEET_POLICIES = ("Deterministic", "Slotwise Q75", "Robust")
COLORS = {
    "Deterministic": "#0072B2", "Slotwise Q75": "#E69F00", "Robust": "#009E73",
    "Empirical 95%": "#CC79A7", "Conformal 95%": "#D55E00", "Rolling conformal 95%": "#6A51A3",
}
PLOT_STYLE = {
    "font.family": "DejaVu Sans", "font.size": 9, "axes.spines.top": False,
    "axes.spines.right": False, "savefig.bbox": "tight", "pdf.fonttype": 42,
}


def table(name, headers, rows, align=None):
    align = align or "l" + "r" * (len(headers) - 1)
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule", " & ".join(headers) + r" \\", r"\midrule"]
    lines.extend(" & ".join(map(str, row)) + r" \\" for row in rows)
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    TAB.mkdir(parents=True, exist_ok=True)
    (TAB / f"{name}.tex").write_text("\n".join(lines) + "\n", encoding="utf-8")


def figure(fig, name):
    FIG.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=170)
    plt.close(fig)


def historical_tables(history, boundaries):
    rows = []
    for alpha, group in history.groupby("alpha"):
        counts = group["classification"].value_counts()
        rows.append([
            f"{alpha:g}", len(group), counts.get("certified_exact", 0),
            counts.get("observed_exact_uncertified", 0), counts.get("positive_value_gap", 0),
            f"{max(0.0, group.objective_gap.max()):.6f}",
        ])
    table("historical", [r"$\alpha$", "Pairs", "Certified", "Observed only", "Gap cases", "Max gap (EUR)"], rows)

    failures = history.loc[history["classification"] == "positive_value_gap"]
    failure_rows = [
        [r.date, f"{r.qp_objective:.4f}", f"{r.miqp_objective:.4f}", f"{r.objective_gap:.4f}",
         f"{r.qp_max_simultaneous_kw:.3f}"]
        for r in failures.itertuples()
    ]
    table("historical_failures", ["Date", "QP (EUR)", "MIQP (EUR)", "Gap (EUR)", "SCD (kW)"], failure_rows)

    categories = {
        "certified_exact": "Certified", "observed_exact_uncertified": "Observed only",
        "positive_value_gap": "Positive gap", "infeasibility_masked": "Masked infeasibility",
        "relaxation_infeasible_or_error": "Both infeasible",
    }
    pv_cases = boundaries.loc[boundaries["tag"].str.startswith("pv_")]
    rows = []
    for mode, group in pv_cases.groupby("capacity_mode", sort=False):
        counts = group["classification"].value_counts()
        rows.append([mode.replace("_", " "), *(counts.get(key, 0) for key in categories)])
    table("boundaries", ["Protection", *categories.values()], rows)


def reliability_tables(calibration, correlation):
    rows = [
        [r.policy.replace("%", r"\%"), int(r.violations), f"{100*r.path_coverage:.1f}",
         f"{r.objective:.6f}", f"{r.linear_cost:.6f}", f"{r.load_cost:.6f}"]
        for r in calibration.itertuples()
    ]
    table("calibration", ["Policy", "Events / 121", r"Envelope (\%)", "Full (EUR)", "Linear (EUR)", "Load (EUR)"], rows)

    cases = correlation.loc[(correlation["delta"] == 0.5) & (correlation["demand_kwh"] == 40)]
    rows = [
        [r.policy, f"{r.latent_ar1:.1f}", f"{r.violation_rate:.4f}", f"[{r.ci_low:.4f}, {r.ci_high:.4f}]",
         f"{r.independent_formula:.4f}"]
        for r in cases.itertuples()
    ]
    table("correlation", ["Policy", r"Latent $r$", "Joint violation", r"95\% MC interval", "Independent formula"], rows)


def fleet_table(fleet):
    summary = fleet.groupby(["gamma_percent", "policy"], sort=False).agg(
        n=("objective", "size"), violation=("violation_rate", "mean"), violation_sd=("violation_rate", "std"),
        objective=("objective", "mean"), objective_sd=("objective", "std"), linear=("linear_cost", "mean"),
        market=("market_cost", "mean"), throughput=("throughput_cost", "mean"), load=("load_cost", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "fleet_summary.csv", index=False)
    rows = [
        [f"{r.gamma_percent:.0f}", r.policy, f"{r.violation:.3f} ({r.violation_sd:.3f})",
         f"{r.objective:.3f}", f"{r.linear:.3f}"]
        for r in summary.itertuples()
    ]
    table("fleet", [r"$\gamma$ (\%)", "Policy", "Violation mean (SD)", "Full (EUR)", "Linear (EUR)"], rows, "rlrrr")
    return summary


def runtime_table(runtime):
    summary = runtime.groupby(["N", "T"]).agg(
        samples=("N", "size"), qp=("qp_process_seconds", "mean"), miqp=("miqp_process_seconds", "mean"),
        qp_sd=("qp_process_seconds", "std"), miqp_sd=("miqp_process_seconds", "std"),
        qp_mb=("qp_sampled_peak_rss_mb", "mean"), miqp_mb=("miqp_sampled_peak_rss_mb", "mean"),
    ).reset_index()
    summary["ratio"] = summary["miqp"] / summary["qp"]
    summary.to_csv(OUT / "runtime_summary.csv", index=False)
    rows = [
        [r.N, r.T, r.N*r.T, f"{r.qp:.3f}", f"{r.miqp:.3f}", f"{r.ratio:.2f}",
         f"{r.qp_mb:.1f}", f"{r.miqp_mb:.1f}"]
        for r in summary.itertuples()
    ]
    table("runtime", ["EVs", "Slots", "Binaries", "QP (s)", "MIQP (s)", "Ratio", "QP RSS", "MIQP RSS"], rows)

    by_size = summary.set_index(["N", "T"])
    small, large = by_size.loc[(20, 24)], by_size.loc[(200, 96)]
    macros = {
        "SmallQP": f"{small.qp:.3f}", "SmallMIQP": f"{small.miqp:.3f}",
        "LargeQP": f"{large.qp:.3f}", "LargeMIQP": f"{large.miqp:.3f}",
        "LargeQPMemory": f"{large.qp_mb:.1f}", "LargeMIQPMemory": f"{large.miqp_mb:.1f}",
        "LargestGap": f"{runtime.objective_gap.abs().max():.3g}",
    }
    commands = [rf"\newcommand{{\{name}}}{{{value}}}" for name, value in macros.items()]
    (TAB / "result_macros.tex").write_text("\n".join(commands), encoding="utf-8")
    return summary


def degradation_table(degradation):
    summary = degradation.groupby(["rho", "policy"]).agg(
        objective=("objective", "mean"), linear=("linear_cost", "mean"),
        discharge=("discharge_kwh", "mean"), violation=("violation_rate", "mean"),
    ).reset_index()
    summary.to_csv(OUT / "degradation_summary.csv", index=False)
    rows = [
        [f"{r.rho:g}", r.policy, f"{r.objective:.3f}", f"{r.linear:.3f}",
         f"{r.discharge:.3f}", f"{r.violation:.3f}"]
        for r in summary.itertuples()
    ]
    table("degradation", [r"$\rho$ (EUR/kWh)", "Policy", "Full (EUR)", "Linear (EUR)", "V2G (kWh)", "Violation"], rows, "rlrrrr")

def plot_exactness(history, sensitivity):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.3), layout="constrained")
    day = history.loc[(history["date"] == "2019-06-08") & (history["alpha"] == 0)].iloc[0]
    for ax, key, title in zip(axes, [day.qp_key, day.miqp_key], ["(a) Relaxed QP", "(b) Physical MIQP"]):
        record = json.loads((SOLVES / key / "result.json").read_text(encoding="utf-8"))
        for field, label, color in [("x", "Charging", "#0072B2"), ("y", "Discharging", "#D55E00")]:
            power = record[field][0]
            ax.step(np.arange(len(power)), power, where="mid", label=label, color=color)
        ax.set(xlabel="UTC hour", ylabel="Power (kW)", title=title, xlim=(8, 22), ylim=(-1, 31))
        ax.legend(fontsize=8)

    grid = sensitivity.assign(failed=sensitivity["classification"] == "positive_value_gap").pivot_table(
        index="rho", columns="spread", values="failed", aggfunc="mean"
    )
    ax = axes[2]
    ax.imshow(grid, origin="lower", aspect="auto", vmin=0, vmax=1, cmap="YlOrRd")
    ax.set(
        xticks=range(len(grid.columns)), xticklabels=[f"{v:g}" for v in grid.columns],
        yticks=range(len(grid.index)), yticklabels=[f"{v:g}" for v in grid.index],
        xlabel="Selling / buying price multiplier", ylabel=r"$\rho$ (EUR/kWh)",
        title="(c) Failure fraction over 8 stress days",
    )
    for (row, col), value in np.ndenumerate(grid.to_numpy()):
        ax.text(col, row, f"{value*100:.0f}%", ha="center", va="center", fontsize=8)
    figure(fig, "figure1_exactness")


def plot_correlation(correlation):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    for ax, demand in zip(axes, [10, 40]):
        cases = correlation.loc[(correlation["delta"] == 0.5) & (correlation["demand_kwh"] == demand)]
        for label in FLEET_POLICIES:
            group = cases.loc[cases["policy"] == label]
            errors = np.array([group.violation_rate - group.ci_low, group.ci_high - group.violation_rate])
            ax.errorbar(group.latent_ar1, group.violation_rate, yerr=errors, marker="o",
                        label=label, color=COLORS[label], capsize=3)
        ax.set(xlabel="Latent lag-one correlation", ylabel="Joint violation probability",
               title=f"Required battery increment: {demand} kWh", ylim=(-0.02, 0.8))
        ax.legend(fontsize=8)
    figure(fig, "figure2_correlation")


def plot_calibration(calibration, monthly):
    fig, (daily_ax, monthly_ax) = plt.subplots(1, 2, figsize=(10, 3.5), layout="constrained")
    short_labels = {
        "Rolling conformal 95%": "Rolling\nconformal", "Conformal 95%": "Conformal",
        "Empirical 95%": "Empirical", "Slotwise Q75": "Q75",
    }
    positions = np.arange(len(calibration))
    daily_ax.bar(positions, 100 * calibration.path_coverage, color=calibration.policy.map(COLORS))
    daily_ax.axhline(95, color="black", linestyle="--", linewidth=1, label="Nominal 95%")
    daily_ax.set(xticks=positions, xticklabels=calibration.policy.replace(short_labels),
                 ylabel="Held-out envelope coverage (%)", ylim=(0, 104))
    daily_ax.tick_params(axis="x", labelrotation=35)
    daily_ax.legend(fontsize=8)

    for label in ("Conformal 95%", "Rolling conformal 95%", "Robust"):
        group = monthly.loc[monthly["policy"] == label]
        monthly_ax.plot(group.month, 100 * group.path_coverage, marker="o", color=COLORS[label], label=label)
    monthly_ax.axhline(95, color="black", linestyle="--", linewidth=1)
    monthly_ax.set(ylabel="Monthly envelope coverage (%)", ylim=(35, 103), xlabel="Test month")
    monthly_ax.legend(fontsize=8)
    figure(fig, "figure3_calibration")


def plot_fleets(fleet, summary):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
    risk_ax, cost_ax = axes
    objectives = fleet.pivot(index=["seed", "gamma_percent"], columns="policy", values="objective")
    for label in FLEET_POLICIES:
        color = COLORS[label]
        group = summary.loc[summary["policy"] == label].sort_values("gamma_percent")
        risk_ax.plot(group.gamma_percent, group.violation, marker="o", color=color, label=label)
        risk_ax.fill_between(group.gamma_percent, (group.violation - group.violation_sd).clip(lower=0),
                             (group.violation + group.violation_sd).clip(upper=1), color=color, alpha=0.15)
        paired = objectives[label] - objectives["Deterministic"]
        premium = paired.groupby("gamma_percent").agg(["mean", "std"])
        cost_ax.plot(premium.index, premium["mean"], marker="o", color=color, label=label)
        cost_ax.fill_between(premium.index, premium["mean"] - premium["std"],
                             premium["mean"] + premium["std"], color=color, alpha=0.15)
    risk_ax.set(ylabel="Joint violation probability", ylim=(-0.02, 1))
    cost_ax.set(ylabel="Full-objective premium (EUR/day)")
    for ax in axes:
        ax.set(xlabel=r"Capacity margin $\gamma$ (%)", xticks=[0, 5, 10, 15])
        ax.legend(fontsize=8)
    figure(fig, "figure4_fleet")


def plot_runtime(summary):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.3), layout="constrained")
    for slots, style in [(24, "--"), (96, "-")]:
        group = summary.loc[summary["T"] == slots]
        for formulation, color in [("qp", "#0072B2"), ("miqp", "#D55E00")]:
            for ax, suffix in zip(axes, ["", "_mb"]):
                ax.plot(group.N, group[formulation + suffix], marker="o", ls=style, color=color,
                        label=f"{formulation.upper()}, T={slots}")
    axes[0].set(ylabel="Mean solver-process wall time (s)", yscale="log")
    axes[1].set(ylabel="Mean sampled peak solver RSS (MiB)")
    for ax in axes:
        ax.set(xlabel="Number of EVs", xticks=[5, 20, 50, 100, 200])
        ax.legend(fontsize=8)
    figure(fig, "figure5_runtime")


def main():
    history = pd.read_csv(OUT / "historical.csv")
    sensitivity = pd.read_csv(OUT / "sensitivity.csv")
    calibration = pd.read_csv(OUT / "calibration_summary.csv")
    correlation = pd.read_csv(OUT / "correlation.csv")
    fleet = pd.read_csv(OUT / "fleet_reliability.csv")
    runtime = pd.read_csv(OUT / "runtime.csv")

    historical_tables(history, pd.read_csv(OUT / "boundaries.csv"))
    reliability_tables(calibration, correlation)
    fleet_summary = fleet_table(fleet)
    runtime_summary = runtime_table(runtime)
    degradation_table(pd.read_csv(OUT / "degradation.csv"))
    with plt.rc_context(PLOT_STYLE):
        plot_exactness(history, sensitivity)
        plot_correlation(correlation)
        plot_calibration(calibration, pd.read_csv(OUT / "calibration_monthly.csv"))
        plot_fleets(fleet, fleet_summary)
        plot_runtime(runtime_summary)

    summary = {
        "historical_pairs": len(history),
        "historical_gap_cases": int(history["classification"].eq("positive_value_gap").sum()),
        "historical_max_gap": float(history.objective_gap.max()),
        "sensitivity_pairs": len(sensitivity),
        "sensitivity_gap_cases": int(sensitivity["classification"].eq("positive_value_gap").sum()),
        "runtime_pairs": len(runtime),
        "runtime_max_abs_gap": float(runtime.objective_gap.abs().max()),
        "runtime_all_optimal": bool(runtime.qp_optimal.all() and runtime.miqp_optimal.all()),
        "max_feasibility_residual": float(max(runtime.qp_feasibility_residual.max(), runtime.miqp_feasibility_residual.max())),
        "calibration_days": 121,
        "fleet_runs": len(fleet),
    }
    rendered = json.dumps(summary, indent=2)
    (OUT / "paper_summary.json").write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
