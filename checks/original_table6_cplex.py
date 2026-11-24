"""Compare the continuous and binary CPLEX models on four single-EV windows.

Each window uses the original January 24, 2019 station profile and three
charging-capacity policies. The output records both individual solutions and
the difference in their full objectives recomputed from the returned schedules.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import cplex
import numpy as np
import pandas as pd


THREADS = 1
TIME_LIMIT = 60.0
FEASIBILITY_TOL = 1e-9
OPTIMALITY_TOL = 1e-9
INTEGRALITY_TOL = 1e-9
BARRIER_TOL = 1e-11
RELATIVE_MIP_GAP = 1e-10
ABSOLUTE_MIP_GAP = 1e-9
UNCERTAINTY_RADIUS = 0.5
POLICIES = ("deterministic", "quantile_75", "robust")


def solve_station(data, coefficients, *, binary):
    """Solve the single-EV QP or its physical binary counterpart in CPLEX."""
    n = data.T
    charge = list(range(n))
    discharge = list(range(n, 2 * n))
    energy = list(range(2 * n, 3 * n + 1))
    net_load = list(range(3 * n + 1, 4 * n + 1))
    charge_limit = data.X * data.availability
    discharge_limit = data.Y * data.availability

    with cplex.Cplex() as model:
        model.set_log_stream(None)
        model.set_results_stream(None)
        model.set_warning_stream(None)
        params = model.parameters
        params.threads.set(THREADS)
        params.timelimit.set(TIME_LIMIT)
        params.simplex.tolerances.feasibility.set(FEASIBILITY_TOL)
        params.simplex.tolerances.optimality.set(OPTIMALITY_TOL)
        params.barrier.convergetol.set(BARRIER_TOL)
        params.mip.tolerances.integrality.set(INTEGRALITY_TOL)
        params.mip.tolerances.mipgap.set(RELATIVE_MIP_GAP)
        params.mip.tolerances.absmipgap.set(ABSOLUTE_MIP_GAP)

        model.variables.add(
            obj=[*(data.price_buy + data.rho), *(-data.price_sell + data.rho), *np.zeros(2 * n + 1)],
            lb=[*np.zeros(2 * n), data.E0, *np.full(n, data.Emin), *np.full(n, -cplex.infinity)],
            ub=[*charge_limit, *discharge_limit, data.E0, *np.full(n, data.Emax), *np.full(n, cplex.infinity)],
        )
        if binary:
            mode = list(range(4 * n + 1, 5 * n + 1))
            model.variables.add(types="B" * n)
        # CPLEX uses one half times the quadratic form.
        model.objective.set_quadratic_coefficients([(i, i, float(2 * data.alpha)) for i in net_load])

        def add_constraint(indices, values, sense, rhs):
            model.linear_constraints.add(
                lin_expr=[cplex.SparsePair(ind=indices, val=list(map(float, values)))],
                senses=sense,
                rhs=[float(rhs)],
            )

        add_constraint([energy[-1]], [1], "G", data.Etarget)
        for t in range(n):
            add_constraint(
                [energy[t + 1], energy[t], charge[t], discharge[t]],
                [1, -1, -data.eta_c, 1 / data.eta_d], "E", 0,
            )
            add_constraint([net_load[t], charge[t], discharge[t]], [1, -1, 1], "E", data.L[t])
            add_constraint([charge[t]], [coefficients[t]], "L", data.M - data.L[t])
            if binary:
                add_constraint([charge[t], mode[t]], [1, -charge_limit[t]], "L", 0)
                add_constraint([discharge[t], mode[t]], [1, discharge_limit[t]], "L", discharge_limit[t])

        model.solve()
        solution = model.solution
        code = solution.get_status()
        status = solution.get_status_string()
        if code not in (1, 101, 102):
            raise RuntimeError(f"CPLEX did not certify an optimum for {data.name} (binary={binary}): {status}")
        x = np.asarray(solution.get_values(charge))
        y = np.asarray(solution.get_values(discharge))
        soc = np.asarray(solution.get_values(energy))
        objective = np.sum(data.price_buy * x - data.price_sell * y + data.rho * (x + y)
                           + data.alpha * (data.L + x - y) ** 2)
        residual = max(
            0, abs(soc[0] - data.E0), np.abs(np.diff(soc) - data.eta_c * x + y / data.eta_d).max(),
            (data.L + coefficients * x - data.M).max(), (x - charge_limit).max(),
            (y - discharge_limit).max(), (data.Emin - soc[1:]).max(), (soc[1:] - data.Emax).max(),
            data.Etarget - soc[-1], -x.min(), -y.min(),
        )
        return x, y, {
            "status": status, "code": code, "F": float(objective), "solverF": solution.get_objective_value(),
            "simultaneous": max(0.0, float(np.minimum(x, y).max())), "residual": float(residual),
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output-dir", type=Path, help="Default: PROJECT/results/")
    args = parser.parse_args()
    project = args.project_root.resolve()
    sys.path.insert(0, str(project / "notebooks"))
    import ev_robust_charging as archived

    profile = project / "data" / "opsd_de_lu_2019_01_24_profile.csv"
    reference_path = project / "checks" / "q1_binary_complementarity_summary.csv"
    reference = pd.read_csv(reference_path).set_index(["case", "validation_policy"], verify_integrity=True)
    if not profile.is_file():
        raise FileNotFoundError(profile)
    base = archived.build_case_from_profile("2019-01-24", name="base", profile=pd.read_csv(profile))
    cases = archived.build_miqp_validation_cases(base)
    policies = {policy.name: policy for policy in archived.POLICIES}
    rows = []
    for case in cases:
        for name in POLICIES:
            coefficients = archived.policy_u(case, UNCERTAINTY_RADIUS, policies[name])
            qp_x, qp_y, qp = solve_station(case, coefficients, binary=False)
            miqp_x, miqp_y, miqp = solve_station(case, coefficients, binary=True)
            historical = reference.at[(case.name, name), "relaxed_objective"]
            rows.append({
                "case": case.name, "T": case.T, "policy": name, "M": case.M, "target": case.Etarget,
                "gap": abs(qp["F"] - miqp["F"]), "charge_l1": np.abs(qp_x - miqp_x).sum(),
                "discharge_l1": np.abs(qp_y - miqp_y).sum(), "old_qp_F": historical,
                "vs_old_qp": abs(qp["F"] - historical),
                **{f"qp_{key}": value for key, value in qp.items()},
                **{f"miqp_{key}": value for key, value in miqp.items()},
            })

    results = pd.DataFrame(rows)
    if len(results) != 12 or results.vs_old_qp.max() >= 1e-4:
        raise RuntimeError("The four-window comparison disagrees with the archived reference.")
    diagnostics = ["gap", "charge_l1", "discharge_l1", "qp_simultaneous", "miqp_simultaneous",
                   "qp_residual", "miqp_residual", "vs_old_qp"]
    archive = Path(archived.__file__).resolve()
    def sha256(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()
    with cplex.Cplex() as solver:
        solver_version = solver.get_version()
    metadata = {
        "version": solver_version, "python": sys.version.split()[0],
        "archive_hash": sha256(archive), "profile_hash": sha256(profile), "pairs": len(results),
        "qp_status": results.qp_status.unique().tolist(), "miqp_status": results.miqp_status.unique().tolist(),
        "maxima": results[diagnostics].max().to_dict(), "cplex_threads": THREADS,
        "time_limit_seconds": TIME_LIMIT, "feasibility_tolerance": FEASIBILITY_TOL,
        "optimality_tolerance": OPTIMALITY_TOL, "integrality_tolerance": INTEGRALITY_TOL,
        "barrier_convergence_tolerance": BARRIER_TOL, "relative_mip_gap": RELATIVE_MIP_GAP,
        "absolute_mip_gap": ABSOLUTE_MIP_GAP, "delta": UNCERTAINTY_RADIUS,
        "objective": "Market payment, grid-side throughput charge and squared net station load at one-hour intervals.",
        "paired_gap": "Absolute difference in full objectives recomputed from the returned power schedules.",
        "overlap": "Maximum across slots of min(charge, discharge) in kW, clipped below at zero.",
        "source_paths": {"archive": archive.relative_to(project).as_posix(), "profile": profile.relative_to(project).as_posix(),
                         "archive_comparison": reference_path.relative_to(project).as_posix()},
        "solver_type": "IBM ILOG CPLEX Python API, continuous QP and binary MIQP",
        "archive_comparison_sha256": sha256(reference_path), "script_sha256": sha256(Path(__file__)),
    }
    output = (args.output_dir or project / "results").resolve()
    output.mkdir(parents=True, exist_ok=True)
    results.to_csv(output / "original_table6_cplex_pairs.csv", index=False)
    (output / "original_table6_cplex_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"{len(results)} optimal QP--MIQP pairs; maximum absolute objective difference {max(results.gap):.3g}.")


if __name__ == "__main__":
    main()
