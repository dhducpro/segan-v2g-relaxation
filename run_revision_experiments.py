"""Run the station scheduling experiments and save their measurements by stage."""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
import time
from dataclasses import replace
from pathlib import Path

import cplex
import numpy as np
import pandas as pd
import psutil

import ev_robust_charging as inputs
import revision_model as model


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "results"
OUT.mkdir(parents=True, exist_ok=True)
SEED = 20261112
POLICIES = {"Deterministic": 1.0, "Slotwise Q75": 1.25, "Robust": 1.5}
THROUGHPUT_PENALTIES = (0.0, 0.001, 0.0023, 0.0045, 0.01, 0.05)


def save(rows, name):
    frame = pd.DataFrame(rows)
    frame.to_csv(OUT / f"{name}.csv", index=False)
    return frame


def days():
    """Return the hourly archive and complete UTC days in chronological order."""
    hourly = inputs.load_opsd_year()
    hourly["date"] = hourly.utc_timestamp.str[:10]
    complete_days = {
        date: group
        for date, group in hourly.groupby("date")
        if len(group) == 24 and group[inputs.OPSD_COLUMNS].notna().all().all()
    }
    return hourly, complete_days


def day_instance(date, raw=True, spread=0.98):
    station = model.from_legacy(inputs.build_case_from_profile(date))
    if raw:
        profile = inputs.load_public_profile(date)
        prices = profile.raw_price_eur_per_mwh.to_numpy() / 1000
        station = replace(station, buy=prices, sell=spread * prices)
    return station


def price_audit():
    hourly, complete_days = days()
    prices = hourly.DE_LU_price_day_ahead
    daily_connected = [group.DE_LU_price_day_ahead.to_numpy()[10:21] for group in complete_days.values()]
    connected = np.concatenate(daily_connected)
    audit = {
        "all_rows": len(hourly),
        "nonmissing_prices": int(prices.notna().sum()),
        "negative_hours": int((prices < 0).sum()),
        "below_1_eur_mwh_hours": int((prices < 1).sum()),
        "complete_days": len(complete_days),
        "negative_price_complete_days": sum(bool((g.DE_LU_price_day_ahead < 0).any()) for g in complete_days.values()),
        "connected_negative_hours": int((connected < 0).sum()),
        "connected_hours": len(connected),
        "negative_price_days_connected": sum(bool((p < 0).any()) for p in daily_connected),
        "min_price_eur_mwh": float(prices.min()),
        "max_price_eur_mwh": float(prices.max()),
    }
    (OUT / "price_audit.json").write_text(json.dumps(audit, indent=2))
    print(audit, flush=True)


def historical():
    _, complete_days = days()
    rows = []
    for number, date in enumerate(complete_days, start=1):
        base = day_instance(date)
        for alpha in (0.0, 2e-4):
            station = replace(base, alpha=alpha)
            row = model.pair(station, f"historical_{date}_{alpha}")
            connected_prices = station.buy[station.availability[0] > 0]
            row.update(date=date, negative_connected=int((connected_prices < 0).sum()))
            rows.append(row)
        if number % 20 == 0:
            save(rows, "historical")
            print(f"historical {number}/{len(complete_days)}", flush=True)
    save(rows, "historical")


def sensitivity():
    _, complete_days = days()
    # Select the eight lowest connected-window prices before any optimization.
    ranked = sorted(complete_days, key=lambda date: complete_days[date].DE_LU_price_day_ahead.iloc[10:21].min())
    rows = []
    for date in ranked[:8]:
        base = day_instance(date)
        for spread in (0.90, 0.98, 1.0, 1.02, 1.10):
            for rho in THROUGHPUT_PENALTIES:
                station = replace(base, sell=spread * base.buy, rho=np.array([rho]), alpha=0.0)
                row = model.pair(station, f"sensitivity_{date}_{spread}_{rho}")
                row.update(date=date, spread=spread)
                rows.append(row)
        print(f"sensitivity {date}", flush=True)
        save(rows, "sensitivity")


def boundaries():
    base = day_instance("2019-01-24")
    hours = np.arange(24)
    demand = 8.0 + 3 * np.cos(2 * np.pi * (hours - 19) / 24)
    solar = np.maximum(0, np.sin(np.pi * (hours - 6) / 12))
    rows = []
    for peak_pv in (0.0, 5.0, 15.0, 30.0):
        for mode in ("charging", "net_import", "two_sided"):
            for alpha in (0.0, 2e-4, 0.01):
                station = replace(
                    base, name=f"pv_{peak_pv}_{mode}_{alpha}", L=demand - peak_pv * solar,
                    capacity=45.0, capacity_mode=mode, export_limit=5.0, alpha=alpha,
                    pcmax=np.array([15.0]), pdmax=np.array([15.0]), availability=np.ones((1, 24)),
                )
                row = model.pair(station, station.name)
                row["pv_peak_kw"] = peak_pv
                rows.append(row)

    # One-slot controls separate price incentives from a binding export limit.
    one_slot = model.Instance(
        name="one_slot", L=np.array([0.0]), buy=np.array([1.0]), sell=np.array([1.0]),
        availability=np.ones((1, 1)), pcmax=np.array([10.0]), pdmax=np.array([10.0]),
        e0=np.array([5.0]), emin=np.array([0.0]), emax=np.array([5.0]), target=np.array([5.0]),
        etac=np.array([0.95]), etad=np.array([0.9]), rho=np.array([0.0]),
        coeff=np.ones((1, 1)), capacity=100.0, alpha=0.0,
    )
    cases = [
        replace(one_slot, name="positive_zero_penalty"),
        replace(one_slot, name="negative_price", buy=np.array([-1.0]), sell=np.array([-1.0])),
        replace(one_slot, name="export_infeasibility", L=np.array([-1.0]),
                capacity_mode="two_sided", rho=np.array([0.01])),
        replace(one_slot, name="surplus_smoothing", L=np.array([-1.0]), buy=np.array([0.0]),
                sell=np.array([0.0]), alpha=1.0, rho=np.array([0.001])),
    ]
    for case in cases:
        row = model.pair(case, case.name)
        row["pv_peak_kw"] = -float(case.L.min())
        rows.append(row)
    save(rows, "boundaries")


def calibration():
    """Fit hourly load distributions on training days and evaluate later days."""
    _, complete_days = days()
    dates = list(complete_days)
    raw_loads = np.vstack([group.DE_LU_load_actual_entsoe_transparency.to_numpy() for group in complete_days.values()])
    training_days, test_start = 96, 240
    fitted = model.fit_ecdf(raw_loads[:training_days])
    ranks = model.apply_ecdf(fitted, raw_loads)
    nominal, scale, epsilon = 1.5, 0.75, 0.05
    paths = nominal + scale * ranks
    calibration_paths = paths[training_days:test_start]
    test_paths = paths[test_start:]
    scores = ((calibration_paths - nominal) / scale).max(axis=1)
    fixed_quantile = model.conformal_quantile(scores, epsilon)
    split = {
        "training_days": dates[:training_days], "calibration_days": dates[training_days:test_start],
        "test_days": dates[test_start:], "training_n": training_days, "calibration_n": len(calibration_paths),
        "test_n": len(test_paths), "hat_u": nominal, "scale_d": scale, "conformal_epsilon": epsilon,
        "order_statistic": int(np.ceil((len(calibration_paths) + 1) * (1 - epsilon))),
        "fixed_q": fixed_quantile,
        "interpretation": "Synthetic coefficient proxy; temporal test coverage is empirical, not exchangeability-guaranteed.",
    }
    (OUT / "calibration_split.json").write_text(json.dumps(split, indent=2))
    np.savez_compressed(
        OUT / "calibration_paths.npz", training_raw=raw_loads[:training_days], fitted_ecdf=fitted,
        calibration=calibration_paths, test=test_paths, dates=np.array(dates),
    )
    empirical = np.quantile(calibration_paths, 0.95, axis=0)
    rows = []
    for offset, date in enumerate(dates[test_start:]):
        # Reliability controls use the floored tariff and reschedule each day.
        base = day_instance(date, raw=False)
        available_paths = paths[training_days:test_start + offset][-len(calibration_paths):]
        past_scores = ((available_paths - nominal) / scale).max(axis=1)
        rolling_quantile = model.conformal_quantile(past_scores, epsilon)
        envelopes = {
            "Deterministic": np.full(24, nominal),
            "Slotwise Q75": np.full(24, 1.875),
            "Empirical 95%": empirical,
            "Conformal 95%": np.full(24, nominal + scale * fixed_quantile),
            "Rolling conformal 95%": np.full(24, nominal + scale * rolling_quantile),
            "Robust": np.full(24, 2.25),
        }
        observed_path = test_paths[offset]
        for label, envelope in envelopes.items():
            station = replace(base, coeff=envelope[None, :])
            solution = model.solve(station, tag=f"cal_{date}_{label}")
            row = {
                **model.flat(solution), "date": date, "policy": label, "month": date[:7],
                "path_covered": bool(np.all(observed_path <= envelope + 1e-12)),
                "protection_mean": float(envelope.mean()),
            }
            if solution["has_solution"]:
                charging = np.array(solution["x"])[0]
                excess = float((base.L + observed_path * charging - base.capacity).max())
                row.update(violation=int(excess > 1e-6), max_excess=max(0.0, excess))
            rows.append(row)
        if (offset + 1) % 20 == 0:
            save(rows, "calibration")
            print(f"calibration {offset + 1}/{len(test_paths)}", flush=True)
    daily_results = save(rows, "calibration")
    summarize_calibration(daily_results, len(test_paths))


def summarize_calibration(results, test_days):
    # Sample seven-day blocks to describe uncertainty across the time-ordered test days.
    block = 7
    rng = np.random.default_rng(SEED)
    samples = [
        (rng.integers(0, test_days, size=(2000, 1)) + np.arange(block)) % test_days
        for _ in range(int(np.ceil(test_days / block)))
    ]
    indices = np.concatenate(samples, axis=1)[:, :test_days]
    means = ("objective", "market_cost", "throughput_cost", "linear_cost", "load_cost",
             "discharge_kwh", "protection_mean")
    summary = []
    for label, group in results.groupby("policy", sort=False):
        group = group.sort_values("date")
        violations = group.violation.to_numpy(float)
        low, high = np.quantile(violations[indices].mean(axis=1), [0.025, 0.975])
        summary.append({
            "policy": label, "n": len(group), "violations": int(violations.sum()),
            "violation_rate": violations.mean(), "block_ci_low": low, "block_ci_high": high,
            "path_coverage": group.path_covered.mean(),
            **{column: group[column].mean() for column in means}, "max_excess": group.max_excess.max(),
        })
    save(summary, "calibration_summary")
    monthly = results.groupby(["month", "policy"]).agg(
        n=("violation", "size"), violation_rate=("violation", "mean"), path_coverage=("path_covered", "mean"),
    )
    monthly.to_csv(OUT / "calibration_monthly.csv")


def correlation():
    reference = day_instance("2019-01-24", raw=False)
    rows = []
    for increment in (10.0, 40.0):
        base = replace(reference, e0=np.array([60.0 - increment]))
        for delta in (0.1, 0.3, 0.5):
            for label, quantile in (("Deterministic", 0.5), ("Slotwise Q75", 0.75), ("Robust", 1.0)):
                coefficients = base.coeff * (1 - delta + 2 * delta * quantile)
                station = replace(base, coeff=coefficients)
                solution = model.solve(station, tag=f"corr_{delta}_{label}")
                if not solution["optimal"]:
                    raise RuntimeError(solution["status"])
                charge = np.array(solution["x"])
                discharge = np.array(solution["y"])
                # The product formula uses the independent, uniform hourly marginals.
                hourly_limits = np.full(24, np.inf)
                np.divide(base.capacity - base.L, charge[0], out=hourly_limits, where=charge[0] > 1e-8)
                hourly_coverage = np.clip((hourly_limits - 1.5 * (1 - delta)) / (3 * delta), 0, 1)
                independent_risk = 1 - np.prod(hourly_coverage)
                costs = model.objective_components(station, charge, discharge)
                for lag_correlation in (0.0, 0.5, 0.9):
                    uniform = model.uniform_paths(20000, 24, lag_correlation, SEED)
                    paths = 1.5 * (1 - delta + 2 * delta * uniform)
                    violations = (base.L + paths * charge[0] - base.capacity).max(axis=1) > 1e-6
                    low, high = model.wilson(int(violations.sum()), len(violations))
                    rows.append({
                        "key": solution["key"], "policy": label, "demand_kwh": increment, "delta": delta,
                        "latent_ar1": lag_correlation, "samples": len(paths), "violation_rate": violations.mean(),
                        "ci_low": low, "ci_high": high, "independent_formula": independent_risk, **costs,
                    })
    save(rows, "correlation")


def degradation():
    _, complete_days = days()
    dates = [next(date for date in complete_days if int(date[5:7]) == month) for month in range(1, 13)]
    rows = []
    for date in dates:
        base = day_instance(date, raw=False)
        for rho in THROUGHPUT_PENALTIES:
            for label, multiplier in POLICIES.items():
                station = replace(base, rho=np.array([rho]), coeff=base.coeff * multiplier)
                solution = model.solve(station, tag=f"degradation_{date}_{rho}_{label}")
                row = {**model.flat(solution), "date": date, "rho": rho, "policy": label}
                if solution["has_solution"]:
                    paths = 1.5 * (0.5 + model.uniform_paths(10000, 24, 0.7, SEED))
                    charging = np.array(solution["x"])[0]
                    violations = (base.L + paths * charging - base.capacity).max(axis=1) > 1e-6
                    row["violation_rate"] = float(violations.mean())
                rows.append(row)
        save(rows, "degradation")


def exchangeability():
    # Generate independent daily paths with within-day temporal dependence.
    covered = []
    for replication in range(2000):
        paths = model.uniform_paths(145, 24, 0.7, SEED + replication)
        quantile = model.conformal_quantile(paths[:144].max(axis=1), 0.05)
        covered.append(paths[-1].max() <= quantile)
    low, high = model.wilson(sum(covered), len(covered))
    result = {
        "replications": len(covered), "calibration_n": 144, "path_length": 24, "latent_ar1": 0.7,
        "coverage": float(np.mean(covered)), "ci_low": low, "ci_high": high,
        "rank": 138, "finite_sample_lower_bound": 138 / 145,
    }
    (OUT / "exchangeable_control.json").write_text(json.dumps(result, indent=2))


def acn_pool():
    """Collect eligible same-day Caltech sessions and their connected fractions."""
    cache = ROOT / "caltech.csv"
    if not cache.is_file():
        raise FileNotFoundError(
            f"The fleet and runtime stages require an ACN-Data Caltech session export at {cache}. "
            "See data/README.md for the source and expected columns."
        )
    sessions = inputs.load_acn_sessions("caltech", "2019-01-01", "2019-03-31", 2.0, cache)
    start = inputs.parse_acn_timestamp(sessions.connectionTime, "America/Los_Angeles")
    end = inputs.parse_acn_timestamp(sessions.disconnectTime, "America/Los_Angeles")
    same_day = start.notna() & end.notna() & (start.dt.date == end.dt.date) & (end > start)
    pool = []
    hours = np.arange(24)
    for index in np.flatnonzero(same_day.to_numpy()):
        midnight = start.iloc[index].normalize()
        arrival = (start.iloc[index] - midnight).total_seconds() / 3600
        departure = (end.iloc[index] - midnight).total_seconds() / 3600
        connected = np.maximum(0, np.minimum(hours + 1, departure) - np.maximum(hours, arrival))
        energy = float(sessions.kWhDelivered.iloc[index])
        if energy <= 7.2 * connected.sum() + 1e-9:
            pool.append({
                "session_id": str(sessions.session_id.iloc[index]), "date": str(start.iloc[index].date()),
                "arrival": arrival, "departure": departure, "ac_energy": energy, "availability": connected,
            })
    return pool, {
        "filtered_source_sessions": len(sessions), "same_day_feasible_pool": len(pool),
        "energy_conversion": "target=e0+0.95*kWhDelivered (AC-side delivered energy)",
    }


def fleet(N, seed, T=24, *, within_day=False):
    """Build a fleet scenario from same-day sessions or resampled session templates."""
    pool, source_info = acn_pool()
    if T <= 0 or T % 24:
        raise ValueError("T must be a positive multiple of 24")
    if N <= 0 or not pool:
        raise ValueError("A positive fleet size and at least one eligible session are required")

    rng = np.random.default_rng(seed)
    if within_day:
        counts = pd.Series([session["date"] for session in pool]).value_counts()
        eligible_dates = sorted(counts[counts >= N].index)
        if not eligible_dates:
            raise ValueError(f"No eligible Caltech day contains {N} feasible sessions")
        date = eligible_dates[(seed - SEED) % len(eligible_dates)]
        pool = [session for session in pool if session["date"] == date]
    else:
        date = "resampled_Q1"
    chosen = rng.choice(len(pool), N, replace=N > len(pool))
    sessions = [pool[index] for index in chosen]

    base = day_instance("2019-01-24", raw=False)
    target_increments = 0.95 * np.array([session["ac_energy"] for session in sessions])
    battery_sizes = np.maximum(55, target_increments / 0.5)
    initial_energy = 0.4 * battery_sizes
    slots_per_hour = T // 24
    slot_start = np.arange(T) / slots_per_hour
    availability = np.zeros((N, T))
    for index, session in enumerate(sessions):
        end = np.minimum(slot_start + 1 / slots_per_hour, session["departure"])
        start = np.maximum(slot_start, session["arrival"])
        availability[index] = np.maximum(0, end - start) * slots_per_hour
    vehicle_factors = 0.95 + 0.1 * ((np.arange(N) % 7) / 6)
    coefficients = np.repeat(base.coeff, slots_per_hour, axis=1) * vehicle_factors[:, None] * np.linspace(0.97, 1.03, T)[None, :]
    station = model.Instance(
        name=f"fleet_{N}_{T}_{seed}_{date}", L=np.repeat(base.L, slots_per_hour),
        buy=np.repeat(base.buy, slots_per_hour), sell=np.repeat(base.sell, slots_per_hour),
        availability=availability, pcmax=np.full(N, 7.2), pdmax=np.full(N, 4.0),
        e0=initial_energy, emin=0.1 * battery_sizes, emax=battery_sizes, target=initial_energy + target_increments,
        etac=np.full(N, 0.95), etad=np.full(N, 0.90), rho=np.full(N, 0.001), coeff=coefficients,
        capacity=float(max(base.L) + 4 * N), alpha=2e-4, dt=24 / T,
    )
    metadata = {
        "case": station.name, "seed": seed, "N": N, "T": T, "date": date,
        "source": "same_day_sessions" if within_day else "resampled_session_templates", **source_info,
        "sessions": [{key: value for key, value in session.items() if key != "availability"} for session in sessions],
    }
    instances = ROOT / "runs" / "instances"
    instances.mkdir(parents=True, exist_ok=True)
    record = {"metadata": metadata, "instance": model.serializable(station)}
    (instances / f"{station.name}.json").write_text(json.dumps(record, indent=2))
    return station


def minimum_capacity(d):
    """Minimize the charging-side limit that lets every EV reach its target.

    One nonnegative grid-side charging variable is used per EV and slot.
    The last variable is the common station limit in kW. Discharging cannot
    create headroom under this charging-only protection rule.
    """
    n_charge = d.N * d.T
    charge_index = np.arange(n_charge).reshape(d.N, d.T)
    capacity_index = n_charge
    with cplex.Cplex() as problem:
        problem.set_log_stream(None)
        problem.set_results_stream(None)
        problem.set_warning_stream(None)
        problem.parameters.threads.set(1)
        problem.objective.set_sense(problem.objective.sense.minimize)
        charge_bounds = d.pcmax[:, None] * d.availability
        problem.variables.add(
            obj=[0.0] * n_charge + [1.0],
            lb=[0.0] * n_charge + [float(d.L.max())],
            ub=charge_bounds.ravel().tolist() + [cplex.infinity],
        )
        for vehicle in range(d.N):
            energy_per_kw = float(d.dt * d.etac[vehicle])
            row = cplex.SparsePair(charge_index[vehicle].tolist(), [energy_per_kw] * d.T)
            problem.linear_constraints.add(
                lin_expr=[row], senses="G", rhs=[float(d.target[vehicle] - d.e0[vehicle])]
            )
        for slot in range(d.T):
            row = cplex.SparsePair(
                charge_index[:, slot].tolist() + [capacity_index],
                d.coeff[:, slot].tolist() + [-1.0],
            )
            problem.linear_constraints.add(
                lin_expr=[row], senses="L", rhs=[-float(d.L[slot])]
            )
        return model.solve_lp_objective(problem)


def fleet_reliability():
    rows = []
    for replicate in range(20):
        seed = SEED + replicate
        base = fleet(20, seed, within_day=True)
        robust_limit = minimum_capacity(replace(base, coeff=1.5 * base.coeff))
        # Each scenario shares a station disturbance across EVs. Paths are paired across policies.
        disturbance = model.uniform_paths(10000, 24, 0.7, seed)
        for margin in (0.0, 0.05, 0.10, 0.15):
            capacity = robust_limit * (1 + margin) + 1e-5
            for label, multiplier in POLICIES.items():
                station = replace(base, capacity=capacity, coeff=base.coeff * multiplier)
                solution = model.solve(station, tag=f"fleet_rel_{seed}_{margin}_{label}")
                row = {
                    **model.flat(solution), "seed": seed, "gamma_percent": 100 * margin, "policy": label,
                    "robust_min_capacity": robust_limit, "capacity": capacity,
                }
                if solution["has_solution"]:
                    charging = np.array(solution["x"])
                    nominal_impact = (base.coeff * charging).sum(axis=0)
                    excess = (base.L + (0.5 + disturbance) * nominal_impact - capacity).max(axis=1)
                    row.update(violation_rate=float((excess > 1e-6).mean()), max_excess=float(max(0, excess.max())))
                rows.append(row)
        save(rows, "fleet_reliability")
        print(f"fleet reliability {replicate + 1}/20", flush=True)


def runtime():
    rows = []
    for size in (5, 20, 50, 100, 200):
        for slots in (24, 96):
            for replicate in range(3):
                station = fleet(size, SEED + replicate, slots)
                # The background load scales only in this benchmark, keeping all sizes in the certified regime.
                scale = max(1, size / 20)
                station = replace(station, L=station.L * scale, capacity=max(station.L) * scale + 4 * size)
                row = model.pair(station, f"runtime_{size}_{slots}_{replicate}", time_limit=60)
                row["replicate"] = replicate
                rows.append(row)
                save(rows, "runtime")
                print(f"runtime N={size},T={slots},rep={replicate}: {row['miqp_status']}", flush=True)


def manifest():
    files = ["revision_model.py", "run_revision_experiments.py", "ev_robust_charging.py", "data/opsd_de_lu_2019.csv"]
    if (ROOT / "caltech.csv").is_file():
        files.append("caltech.csv")
    with cplex.Cplex() as cplex_api:
        cplex_version = cplex_api.get_version()
    info = {
        "created_utc": pd.Timestamp.now(tz="UTC").isoformat(), "python": sys.version, "platform": platform.platform(),
        "processor": platform.processor(), "logical_cpus": psutil.cpu_count(),
        "ram_gb": psutil.virtual_memory().total / 2**30, "cplex_exe": str(model.CPLEX_EXE),
        "cplex_version": cplex_version, "threads": 1, "seed": SEED,
        "sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in files},
    }
    (OUT / "run_manifest.json").write_text(json.dumps(info, indent=2))


STAGES = {
    "audit": price_audit, "historical": historical, "sensitivity": sensitivity,
    "boundaries": boundaries, "calibration": calibration, "correlation": correlation,
    "fleet": fleet_reliability, "runtime": runtime, "degradation": degradation,
    "exchangeability": exchangeability,
}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stages", nargs="+", default=list(STAGES), choices=list(STAGES))
    args = parser.parse_args()
    manifest()
    for stage in args.stages:
        start = time.perf_counter()
        print(f"START {stage}", flush=True)
        STAGES[stage]()
        print(f"DONE {stage} {time.perf_counter() - start:.1f}s", flush=True)
