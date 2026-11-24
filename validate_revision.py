"""Check the experiment tables, stored schedules and chronological calibration split."""

import argparse
from dataclasses import fields
import json
from pathlib import Path
import re

import numpy as np
import pandas as pd

import revision_model as model


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
SOLVES = ROOT / "runs" / "solves"
PAIR_COUNTS = {"historical": 722, "sensitivity": 240, "boundaries": 40, "runtime": 30}
SCHEDULE_COUNTS = {"calibration": 726, "fleet_reliability": 240, "degradation": 216}


def read_table(name, expected_rows):
    table = pd.read_csv(OUT / f"{name}.csv")
    if len(table) != expected_rows:
        raise ValueError(f"{name} has {len(table)} rows, expected {expected_rows}")
    return table


def check_experiment_tables():
    summaries = []
    solve_keys = set()
    for name, count in PAIR_COUNTS.items():
        table = read_table(name, count)
        solve_keys.update(table["qp_key"], table["miqp_key"])
        certified = table.loc[table["certificate"] & table["qp_has_solution"]]
        if not certified["qp_max_simultaneous_kw"].lt(1e-5).all():
            raise ValueError(f"{name}: simultaneous power exceeds tolerance in a certified solution")
        classes = table["classification"]
        failures = classes.str.contains("solver_status|error", na=True)
        if name == "boundaries":
            # Some export cases are infeasible in both models. Other solver errors are failures.
            both_infeasible = table["qp_status"].eq("infeasible") & table["miqp_status"].eq("infeasible")
            failures &= ~(classes.eq("relaxation_infeasible_or_error") & both_infeasible)
        if failures.any():
            raise ValueError(f"{name}: unexpected solver status in {list(table.index[failures])}")
        summaries.append({"dataset": name, "rows": count, "classification": classes.value_counts().to_dict()})

    for name, count in SCHEDULE_COUNTS.items():
        table = read_table(name, count)
        if not table["optimal"].eq(True).all():
            raise ValueError(f"{name} contains a schedule without an optimal status")
        solve_keys.update(table["key"])
        summaries.append({"dataset": name, "rows": count, "all_optimal": True})

    solve_keys.update(read_table("correlation", 54)["key"])
    return summaries, solve_keys


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check_solver_records(keys):
    largest_residual = largest_cost_error = 0.0
    scalar_fields = {"name", "capacity", "alpha", "dt", "capacity_mode", "export_limit"}
    array_fields = {field.name for field in fields(model.Instance)} - scalar_fields

    for key in sorted(keys):
        folder = SOLVES / key
        result = read_json(folder / "result.json")
        log = (folder / "solver.log").read_text(encoding="utf-8", errors="replace")
        if "CPLEX" not in log:
            raise ValueError(f"{key}: missing CPLEX log")
        if not result["has_solution"]:
            if result["status"] != "infeasible" or "infeasible" not in log.lower():
                raise ValueError(f"{key}: unexpected solve status {result['status']}")
            continue
        if "CPLEX Error" in log:
            raise ValueError(f"{key}: solver error in the log for a saved solution")

        parameters = read_json(folder / "input.json")["instance"]
        instance = model.Instance(**{
            name: np.asarray(value, dtype=float) if name in array_fields else value
            for name, value in parameters.items()
        })
        charge, discharge, energy = (np.asarray(result[name]) for name in ("x", "y", "e"))
        residual = model.feasibility_residual(instance, charge, discharge, energy, result["binary"])
        recomputed = model.objective_components(instance, charge, discharge)["objective"]
        solver_cost = result["solver_objective"]
        cost_error = abs(recomputed - solver_cost)
        if not residual < 2e-5:
            raise ValueError(f"{key}: constraint residual {residual:g}")
        if not cost_error < max(1e-6, 1e-9 * abs(solver_cost)):
            raise ValueError(f"{key}: objective discrepancy {cost_error:g}")
        largest_residual = max(largest_residual, residual)
        largest_cost_error = max(largest_cost_error, cost_error)
    return largest_residual, largest_cost_error


def check_calibration_split():
    split = read_json(OUT / "calibration_split.json")
    train, calibration, test = (split[k] for k in ("training_days", "calibration_days", "test_days"))
    if not (max(train) < min(calibration) and max(calibration) < min(test)):
        raise ValueError("Training, calibration and test days must be disjoint and chronological")


def check_manuscript(path):
    tex = path.read_text(encoding="utf-8")
    citations = {key for group in re.findall(r"\\cite\{([^}]+)\}", tex) for key in group.split(",")}
    bibliography = set(re.findall(r"\\bibitem\{([^}]+)\}", tex))
    if citations != bibliography:
        raise ValueError(f"Citation mismatch: {sorted(citations ^ bibliography)}")
    labels = set(re.findall(r"\\label\{([^}]+)\}", tex))
    missing_labels = set(re.findall(r"\\(?:eq)?ref\{([^}]+)\}", tex)) - labels
    if missing_labels:
        raise ValueError(f"Missing LaTeX labels: {sorted(missing_labels)}")
    for filename in re.findall(r"\\(?:input|includegraphics)(?:\[[^]]*\])?\{([^}]+)\}", tex):
        if not (path.parent / filename).is_file():
            raise FileNotFoundError(filename)


def main(results_only=False):
    summaries, keys = check_experiment_tables()
    max_residual, max_cost_error = check_solver_records(keys)
    check_calibration_split()
    manuscript = ROOT / "paper.tex"
    check_references = not results_only and manuscript.is_file()
    if check_references:
        check_manuscript(manuscript)

    summary = {
        "passed": True,
        "unique_recorded_solves_checked": len(keys),
        "datasets": summaries,
        "max_recomputed_constraint_residual": max_residual,
        "max_objective_recompute_error": max_cost_error,
        "chronological_split_disjoint": True,
        "all_references_cited": True if check_references else None,
        "note": "Numerical optimality is assessed using solver tolerances.",
    }
    rendered = json.dumps(summary, indent=2)
    (OUT / "validation.json").write_text(rendered, encoding="utf-8")
    (OUT / "current_solve_keys.json").write_text(json.dumps(sorted(keys)), encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-only", action="store_true", help="Check numerical results only.")
    main(results_only=parser.parse_args().results_only)
