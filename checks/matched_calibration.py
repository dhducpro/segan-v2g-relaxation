"""Reproduce the fixed January 24 schedule experiment and its train-only correction.

From an environment with the original project's dependencies installed:
    python checks/matched_calibration.py

The project supplies notebooks/ev_robust_charging.py and the 361 cached,
complete 2019 OPSD daily profiles in data/. Each CPLEX schedule is evaluated
against the same 121 held-out proxy paths. Results are written to results/
unless --output-dir is supplied.
"""

import argparse
import hashlib
import json
from dataclasses import replace
from pathlib import Path
import sys

import cplex
import cvxpy as cp
import numpy as np
import pandas as pd


def read_profiles(project, model):
    files = sorted((project / "data").glob("opsd_de_lu_2019_*_profile.csv"))
    frames = [pd.read_csv(path) for path in files]
    if len(files) != 361 or any(len(frame) != 24 for frame in frames):
        raise ValueError("Expected 361 complete, cached, 24-hour daily profiles.")
    dates = [str(frame["utc_timestamp"].iloc[0])[:10] for frame in frames]
    if dates != sorted(set(dates)):
        raise ValueError("Daily profiles must be unique and sorted chronologically.")
    scaled = np.stack([frame["station_baseload"].to_numpy() for frame in frames])
    rebuilt = np.stack([model.affine_scale(frame["raw_load_mw"].to_numpy(), 390, 475)
                        for frame in frames])
    np.testing.assert_allclose(scaled, rebuilt)
    np.testing.assert_array_equal(scaled.max(axis=1), np.full(len(dates), 475.0))
    return dates, scaled, dict(zip(dates, files))


def training_ecdf(loads):
    """Per clock hour, the fraction of the first 96 loads <= the query load."""
    reference = np.sort(loads[:96], axis=0)
    return np.column_stack([
        np.searchsorted(reference[:, hour], loads[:, hour], side="right") / len(reference)
        for hour in range(loads.shape[1])
    ])


def evaluate(model, base, protocol, ranks, first_calibration):
    nominal = base.u_nominal
    scale = 0.5 * nominal
    paths = nominal * (1.0 + 0.5 * ranks)
    calibration = paths[first_calibration:240]
    test = paths[240:]
    scores = np.max((calibration - nominal) / scale, axis=1)
    k = int(np.ceil((len(calibration) + 1) * 0.95))
    q = float(np.sort(scores)[k - 1])
    policies = {
        "Deterministic": nominal,
        "Empirical 90%": np.quantile(calibration, 0.90, axis=0),
        "Empirical 95%": np.quantile(calibration, 0.95, axis=0),
        "Conformal 95%": nominal + q * scale,
        "Robust": nominal + scale,
    }
    rows = []
    for label, coeff in policies.items():
        data = replace(base, u_nominal=coeff)
        sol = model.solve_model(data, delta=0.0, policy=model.Policy("deterministic", label))
        if sol["status"] != "optimal" or sol["solver"] != "CPLEX":
            raise RuntimeError(f"{protocol}, {label}: {sol['solver']} returned {sol['status']}")
        charge = sol["x"]
        discharge = sol["y"]
        market = np.sum(base.price_buy * charge - base.price_sell * discharge)
        throughput = base.rho * np.sum(charge + discharge)
        smoothing = base.alpha * np.sum((base.L + charge - discharge) ** 2)
        recomputed_objective = float(market + throughput + smoothing)
        daily_excess = (base.L + test * charge - base.M).max(axis=1)
        violations = int((daily_excess > 1e-6).sum())
        covered = int(np.all(test <= coeff + 1e-12, axis=1).sum())
        rows.append({
            "protocol": protocol, "policy": label, "n_test_days": len(test),
            "capacity_violation_days": violations, "capacity_violation_rate": violations / len(test),
            "envelope_covered_days": covered, "envelope_coverage_rate": covered / len(test),
            "max_capacity_excess": float(np.maximum(daily_excess, 0).max()),
            "energy_plus_throughput_cost_eur": market + throughput,
            "v2g_energy_kwh": sol["total_discharge"], "full_objective": recomputed_objective,
            "market_cost_eur": market, "throughput_cost_eur": throughput, "load_cost_eur": smoothing,
            "solver_objective": sol["objective"],
            "objective_recompute_error": abs(recomputed_objective - sol["objective"]),
            "solver": sol["solver"], "solver_status": sol["status"],
        })
    info = {
        "calibration_days": len(calibration), "rank_one_based": k, "q": q,
        "conformal_coefficient": float(policies["Conformal 95%"][0]),
        "calibration_scores_equal_to_one": int((scores == 1).sum()),
        "test_scores_equal_to_one": int((((test - nominal) / scale).max(axis=1) == 1).sum()),
    }
    return rows, info


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("project", type=Path, nargs="?", default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, help="Default: PROJECT/results/")
    args = parser.parse_args()
    project = args.project.resolve()
    sys.path.insert(0, str(project / "notebooks"))
    import ev_robust_charging as model

    dates, loads, inputs = read_profiles(project, model)
    base = model.build_case_from_profile("2019-01-24", name="base", profile=pd.read_csv(inputs["2019-01-24"]))
    if base.profile_date != "2019-01-24" or base.M != 475 or not np.all(base.u_nominal == 1.5):
        raise ValueError("Unexpected original base case or station/nominal parameters.")
    original_ranks = pd.DataFrame(loads).rank(method="average", pct=True).to_numpy()
    rows, original_info = evaluate(model, base, "original_full_year_ranks", original_ranks, 0)
    matched_rows, matched_info = evaluate(model, base, "matched_train_only_ecdf",
                                          training_ecdf(loads), 96)
    with cplex.Cplex() as solver:
        solver_version = solver.get_version()
    metadata = {
        "base_day": base.profile_date, "base_schedule": "one fixed schedule per policy and protocol",
        "profiles": "All inputs are cached 24-hour daily station baseloads scaled independently to [390,475].",
        "original_protocol": "Per-hour percentile ranks over all 361 days, with average ranks for ties. The first 240 days calibrate the envelope.",
        "corrected_protocol": "Per-hour frozen training ECDF on the first 96 profiles, using # training loads <= query / 96. The following 144 days calibrate the envelope.",
        "corrected_training_dates": [dates[0], dates[95]],
        "corrected_calibration_dates": [dates[96], dates[239]],
        "shared_test_dates": [dates[240], dates[-1]],
        "coefficient_construction": "1.5 + 0.75 * percentile; scale = 0.75 at each hour.",
        "score": "Maximum over 24 hours of (coefficient - 1.5) / 0.75.",
        "measurement": "A day violates capacity if maximum L + u*charge - M exceeds 1e-6. A day has envelope coverage if every u <= envelope + 1e-12.",
        "interpretation": "The train-only correction also changes calibration sample size and tie convention. It does not change the physical base day or scaled-load definition. Every daily load profile reaches 475, hence its maximum ECDF rank is 1; the corrected conformal and robust bands coincide by construction.",
        "original_full_year_ranks": original_info, "matched_train_only_ecdf": matched_info,
        "versions": {"numpy": np.__version__, "pandas": pd.__version__, "cvxpy": cp.__version__,
                     "cplex_engine": solver_version},
        "solver_names_observed": sorted({row["solver"] for row in rows + matched_rows}),
        "max_objective_recompute_error": max(row["objective_recompute_error"] for row in rows + matched_rows),
        "objective": "Market, grid-side throughput and smoothing components are computed from the same returned power schedule. The solver objective and absolute recomputation difference are also recorded.",
        "original_model_sha256": hashlib.sha256(Path(model.__file__).read_bytes()).hexdigest(),
    }
    output = (args.output_dir or project / "results").resolve()
    output.mkdir(exist_ok=True, parents=True)
    pd.DataFrame(rows + matched_rows).to_csv(output / "matched_calibration_results.csv", index=False)
    (output / "matched_calibration_metadata.json").write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for name, results in [("Original", rows), ("Train-only", matched_rows)]:
        print(f"{name} capacity violation days: {[r['capacity_violation_days'] for r in results]}")


if __name__ == "__main__":
    main()
