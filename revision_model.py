"""Station scheduling with IBM CPLEX. Powers are kW, energies kWh, and costs EUR."""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory

import cplex
import numpy as np
import psutil
from filelock import FileLock
from scipy.stats import norm

ROOT = Path(__file__).resolve().parent
CPLEX_EXE = Path(os.environ.get(
    "CPLEX_EXE",
    r"C:\Program Files\IBM\ILOG\CPLEX_Studio2212\cplex\bin\x64_win64\cplex.exe",
))
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
    def N(self):
        return self.availability.shape[0]

    @property
    def T(self):
        return len(self.L)

    def validate(self):
        if self.capacity_mode not in {"charging", "net_import", "two_sided"}:
            raise ValueError(f"Unknown capacity mode: {self.capacity_mode}")
        if not (self.dt > 0 and self.alpha >= 0 and self.N and self.T):
            raise ValueError("Provide a nonempty fleet and horizon, dt > 0 and alpha >= 0.")
        for values in (self.L, self.buy, self.sell):
            if values.shape != (self.T,) or not np.isfinite(values).all():
                raise ValueError("Load and price vectors need one finite value per slot.")
        vehicle_vectors = (self.pcmax, self.pdmax, self.e0, self.emin, self.emax,
                           self.target, self.etac, self.etad, self.rho)
        for values in vehicle_vectors:
            if values.shape != (self.N,) or not np.isfinite(values).all():
                raise ValueError("Vehicle parameters need one finite value per vehicle.")
        for values in (self.availability, self.coeff):
            if values.shape != (self.N, self.T) or not np.isfinite(values).all():
                raise ValueError("Availability and coefficients need shape (vehicles, slots).")
        if np.any((self.availability < 0) | (self.availability > 1)) or np.any(self.coeff < 0):
            raise ValueError("Availability must be in [0, 1] and coefficients nonnegative.")
        if any(np.any((v <= 0) | (v > 1)) for v in (self.etac, self.etad)):
            raise ValueError("Charging and discharging efficiencies must be in (0, 1].")
        if any(np.any(v < 0) for v in (self.rho, self.pcmax, self.pdmax)):
            raise ValueError("Throughput costs and power limits must be nonnegative.")


def serializable(inst):
    return {key: value.tolist() if isinstance(value, np.ndarray) else value
            for key, value in asdict(inst).items()}


def from_legacy(d, *, raw_prices=False):
    """Import the given price arrays and interpret the scaled station load as kW."""
    n = getattr(d, "N", 1)

    def per_vehicle(value):
        return np.broadcast_to(np.asarray(value, dtype=float), (n,)).copy()

    return Instance(
        name=d.name, L=np.array(d.L), buy=np.array(d.price_buy), sell=np.array(d.price_sell),
        availability=np.asarray(d.availability, dtype=float).reshape(n, d.T),
        pcmax=per_vehicle(d.X), pdmax=per_vehicle(d.Y), e0=per_vehicle(d.E0),
        emin=per_vehicle(d.Emin), emax=per_vehicle(d.Emax), target=per_vehicle(d.Etarget),
        etac=per_vehicle(d.eta_c), etad=per_vehicle(d.eta_d), rho=per_vehicle(d.rho),
        coeff=np.broadcast_to(np.asarray(d.u_nominal, dtype=float), (n, d.T)).copy(),
        capacity=float(d.M), alpha=float(d.alpha),
    )


def objective_components(d, x, y):
    net_load = d.L + (x - y).sum(axis=0)
    market = float(d.dt * np.sum(d.buy * x - d.sell * y))
    throughput = float(d.dt * np.sum(d.rho[:, None] * (x + y)))
    smoothing = float(d.dt * d.alpha * np.sum(net_load**2))
    return {
        "objective": market + throughput + smoothing, "market_cost": market,
        "throughput_cost": throughput, "linear_cost": market + throughput,
        "load_cost": smoothing, "charge_kwh": float(d.dt * x.sum()),
        "discharge_kwh": float(d.dt * y.sum()), "min_net_load": float(net_load.min()),
        "max_simultaneous_kw": float(np.minimum(x, y).max()),
    }


def feasibility_residual(d, x, y, e, complementarity=False):
    """Return the largest positive violation across the original model constraints."""
    net_load = d.L + (x - y).sum(axis=0)
    change = d.dt * (d.etac[:, None] * x - y / d.etad[:, None])
    violations = [
        -x.min(), -y.min(), (x - d.pcmax[:, None] * d.availability).max(),
        (y - d.pdmax[:, None] * d.availability).max(), np.abs(e[:, 0] - d.e0).max(),
        np.abs(np.diff(e, axis=1) - change).max(), (d.emin[:, None] - e).max(),
        (e - d.emax[:, None]).max(), (d.target - e[:, -1]).max(),
    ]
    protected_load = d.L + (d.coeff * x).sum(axis=0) if d.capacity_mode == "charging" else net_load
    violations.append((protected_load - d.capacity).max())
    if d.capacity_mode == "two_sided":
        violations.append((-d.export_limit - net_load).max())
    if complementarity:
        violations.append(np.minimum(x, y).max())
    return float(max(0, *violations))


def certificate(d):
    """Test strict cycle-removal improvement and feasible movement over the power box."""
    k = d.etac * d.etad
    lower_load = d.L - (d.pdmax[:, None] * d.availability).sum(axis=0)
    efficiency = k[:, None]
    direction = (d.buy - efficiency * d.sell + (1 + efficiency) * d.rho[:, None]
                 + 2 * d.alpha * (1 - efficiency) * lower_load)
    active = (d.availability > 0) & (d.pcmax[:, None] > 0) & (d.pdmax[:, None] > 0)
    # With no bidirectional slot, complementarity holds from the power bounds alone.
    margin = float(direction[active].min()) if active.any() else None
    safe = d.capacity_mode != "two_sided" or np.all(lower_load >= -d.export_limit - 1e-12)
    original = (np.all(d.buy >= d.sell) and np.all(d.sell >= 0) and np.all(k < 1)
                and np.all(d.rho > 0) and np.all(lower_load >= 0))
    return {
        "certificate": bool(safe and (margin is None or margin > 1e-10)),
        "directional_margin": margin, "constraint_direction_safe": bool(safe),
        "old_assumptions": bool(original),
    }


def cycle_repair(d, x, y):
    """Remove simultaneous power in each slot without changing its battery increment."""
    efficiency = (d.etac * d.etad)[:, None]
    removable = np.minimum(x, y / efficiency)
    charge = x - removable
    discharge = y - efficiency * removable
    increment = d.dt * (d.etac[:, None] * charge - discharge / d.etad[:, None])
    initial = d.e0[:, None]
    energy = np.column_stack((initial, initial + np.cumsum(increment, axis=1)))
    return charge, discharge, energy


def _write_model(d, binary, path):
    """Write the native model, returning variable indices for reading CPLEX solutions."""
    n, t = d.N, d.T
    slots = n * t
    x = np.arange(slots).reshape(n, t)
    y = x + slots
    e = np.arange(2 * slots, 2 * slots + n * (t + 1)).reshape(n, t + 1)
    net = np.arange(e.max() + 1, e.max() + 1 + t)
    mode = x + net.max() + 1

    charge_limit = d.pcmax[:, None] * d.availability
    discharge_limit = d.pdmax[:, None] * d.availability
    cost = np.r_[(d.dt * (d.buy + d.rho[:, None])).ravel(),
                 (d.dt * (-d.sell + d.rho[:, None])).ravel(), np.zeros(e.size + t)]
    lower = np.r_[np.zeros(2 * slots), np.repeat(d.emin, t + 1), np.full(t, -cplex.infinity)]
    upper = np.r_[charge_limit.ravel(), discharge_limit.ravel(), np.repeat(d.emax, t + 1),
                  np.full(t, cplex.infinity)]
    with cplex.Cplex() as solver:
        for stream in (solver.set_log_stream, solver.set_results_stream,
                       solver.set_warning_stream, solver.set_error_stream):
            stream(None)
        names = [f"{prefix}_{i}_{h}" for prefix, width in (("x", t), ("y", t), ("e", t + 1))
                 for i in range(n) for h in range(width)]
        names.extend(f"l_{h}" for h in range(t))
        solver.variables.add(obj=cost.tolist(), lb=lower.tolist(), ub=upper.tolist(), names=names)
        if binary:
            solver.variables.add(lb=[0.] * slots, ub=[1.] * slots, types="B" * slots,
                                 names=[f"z_{i}_{h}" for i in range(n) for h in range(t)])
        if d.alpha:
            # CPLEX uses (1/2) v'Qv for its quadratic objective.
            solver.objective.set_quadratic_coefficients([(int(j), int(j), 2 * d.dt * d.alpha) for j in net])

        rows, senses, bounds = [], [], []

        def row(indices, values, sense, bound):
            rows.append(cplex.SparsePair(list(map(int, indices)), list(map(float, values))))
            senses.append(sense)
            bounds.append(float(bound))

        for i in range(n):
            row([e[i, 0]], [1], "E", d.e0[i])
            row([e[i, -1]], [1], "G", d.target[i])
            for h in range(t):
                row([e[i, h + 1], e[i, h], x[i, h], y[i, h]],
                    [1, -1, -d.dt * d.etac[i], d.dt / d.etad[i]], "E", 0)
                if binary:
                    row([x[i, h], mode[i, h]], [1, -charge_limit[i, h]], "L", 0)
                    row([y[i, h], mode[i, h]], [1, discharge_limit[i, h]], "L", discharge_limit[i, h])
        for h in range(t):
            row([net[h], *x[:, h], *y[:, h]], [1, *([-1] * n), *([1] * n)], "E", d.L[h])
            if d.capacity_mode == "charging":
                row(x[:, h], d.coeff[:, h], "L", d.capacity - d.L[h])
            else:
                row([net[h]], [1], "L", d.capacity)
            if d.capacity_mode == "two_sided":
                row([net[h]], [1], "G", -d.export_limit)
        solver.linear_constraints.add(lin_expr=rows, senses="".join(senses), rhs=bounds)
        solver.write(str(path))
        return x, y, e, solver.variables.get_num(), solver.linear_constraints.get_num()


def _run_cplex(folder, time_limit):
    """Measure the native CPLEX process, including reading the model and writing its solution."""
    commands = [
        "read model.sav", "set threads 1", "set randomseed 20261112", f"set timelimit {time_limit}",
        "set simplex tolerances feasibility 1e-8", "set simplex tolerances optimality 1e-8",
        "set mip tolerances mipgap 1e-8", "set mip tolerances integrality 1e-8",
        "optimize", "write solution.sol", "quit",
    ]
    started = time.perf_counter()
    peak_rss = 0
    timed_out = False
    with (folder / "solver.log").open("w", encoding="utf-8") as log:
        with subprocess.Popen([str(CPLEX_EXE), "-c", *commands], cwd=folder, stdout=log,
                              stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)) as process:
            try:
                monitor = psutil.Process(process.pid)
                while process.poll() is None:
                    try:
                        peak_rss = max(peak_rss, monitor.memory_info().rss)
                    except psutil.Error:
                        pass
                    if time.perf_counter() - started > time_limit + 30:
                        process.kill()
                        timed_out = True
                        break
                    time.sleep(.005)
            except psutil.NoSuchProcess:
                pass
    return time.perf_counter() - started, peak_rss / 2**20, timed_out, process.returncode


def solve_lp_objective(problem):
    """Return a plain LP's optimal objective using the same licensed executable as the schedules."""
    with TemporaryDirectory(prefix="segan_capacity_") as directory:
        folder = Path(directory)
        problem.write(str(folder / "model.sav"))
        _, _, timed_out, returncode = _run_cplex(folder, 60)
        solution = folder / "solution.sol"
        if returncode or timed_out or not solution.exists():
            log = (folder / "solver.log").read_text(encoding="utf-8", errors="replace")
            raise RuntimeError(f"CPLEX could not solve the capacity LP.\n{log[-1500:]}")
        header = ET.parse(solution).getroot().find("header").attrib
        if header["solutionStatusValue"] != "1":
            raise RuntimeError(f"Fleet capacity LP: {header['solutionStatusString']}")
        return float(header["objectiveValue"])


def _read_solution(d, result, folder, indices, log):
    solution = ET.parse(folder / "solution.sol").getroot()
    header = solution.find("header").attrib
    status = header["solutionStatusValue"]
    result.update(status=header["solutionStatusString"], status_code=status)
    if status in {"3", "103"}:
        result["status"] = "infeasible"
        return
    if header.get("primalFeasible") == "0":
        return
    values = np.zeros(result["variables"])
    for item in solution.find("variables"):
        values[int(item.attrib["index"])] = float(item.attrib["value"])
    charge, discharge, energy = (values[block] for block in indices)
    result.update(objective_components(d, charge, discharge))
    result.update(has_solution=True, optimal=status in {"1", "101", "102"},
                  solver_objective=float(header["objectiveValue"]))
    result["objective_recompute_error"] = abs(result["objective"] - result["solver_objective"])
    result["feasibility_residual"] = feasibility_residual(d, charge, discharge, energy, result["binary"])
    result["mip_relative_gap"] = None
    if result["binary"]:
        gaps = re.findall(r"gap\s*=\s*[^,\n]+,\s*([0-9.]+)%", log)
        if gaps:
            result["mip_relative_gap"] = float(gaps[-1]) / 100
        elif status == "101":
            result["mip_relative_gap"] = 0.
    repaired = cycle_repair(d, charge, discharge)
    result["repair_feasibility_residual"] = feasibility_residual(d, *repaired, complementarity=True)
    result["repair_objective"] = objective_components(d, *repaired[:2])["objective"]
    result.update(x=charge.tolist(), y=discharge.tolist(), e=energy.tolist())
    np.savez_compressed(folder / "schedule.npz", x=charge, y=discharge, e=energy)


def solve(d: Instance, binary=False, tag="run", time_limit=60, force=False):
    d.validate()
    settings = {"binary": binary, "time_limit": time_limit, "threads": 1, "mipgap": 1e-8,
                "feasibility_tol": 1e-8, "optimality_tol": 1e-8, "integrality_tol": 1e-8,
                "version": MODEL_VERSION, "cplex_exe": str(CPLEX_EXE)}
    payload = {"instance": serializable(d), "settings": settings}
    key = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()[:20]
    cache = ROOT / "runs" / "solves"
    folder = cache / key
    # Before the per-solve lock can exist, serialize creation of the shared cache directories.
    if not folder.is_dir():
        with FileLock(str(ROOT / ".runs-init.lock")):
            folder.mkdir(parents=True, exist_ok=True)
    result_file = folder / "result.json"
    with FileLock(str(folder / ".lock")):
        if result_file.exists() and not force:
            cached = json.loads(result_file.read_text(encoding="utf-8"))
            if cached["optimal"] or cached["status"] == "infeasible":
                return {**cached, "tag": tag}

        # An interrupted or failed retry must never expose a previous run's solution.
        for filename in ("result.json", "solution.sol", "schedule.npz"):
            (folder / filename).unlink(missing_ok=True)
        (folder / "input.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        start = time.perf_counter()
        x, y, e, variables, constraints = _write_model(d, binary, folder / "model.sav")
        build_seconds = time.perf_counter() - start
        elapsed, peak, timed_out, returncode = _run_cplex(folder, time_limit)
        log = (folder / "solver.log").read_text(encoding="utf-8", errors="replace")
        timings = re.findall(r"(?:Solution time|Total \(root\+branch&cut\))\s*=\s*([0-9.]+)\s*sec", log)
        result = {
            "tag": tag, "case": d.name, "key": key, "binary": binary, "N": d.N, "T": d.T, "dt": d.dt,
            "variables": variables, "constraints": constraints, "binaries": d.N * d.T if binary else 0,
            "build_seconds": build_seconds, "process_seconds": elapsed, "sampled_peak_rss_mb": peak,
            "solver_reported_seconds": float(timings[-1]) if timings else None,
            "status": "solver_error", "optimal": False, "has_solution": False, "objective": None,
            **certificate(d),
        }
        if timed_out:
            result["status"] = "time_limit"
        elif re.search(r"CPLEX Error\s+1016\b", log):
            result["status"] = "license_size_error"
        elif returncode == 0 and (folder / "solution.sol").exists():
            _read_solution(d, result, folder, (x, y, e), log)
        elif re.search(r"^\s*(?:Presolve|MIP|QP|LP|Barrier|Primal simplex|Dual simplex)\s*-\s*(?:integer\s+)?infeasible(?:\s+problem)?[.\s]*$",
                       log, re.I | re.M):
            result["status"] = "infeasible"
        elif re.search(r"\btime limit exceeded\b", log, re.I):
            result["status"] = "time_limit"

        # Only a complete JSON result commits the cache entry.
        pending = folder / "result.json.tmp"
        pending.write_text(json.dumps(result, indent=2, allow_nan=False), encoding="utf-8")
        pending.replace(result_file)
        return result


def flat(r):
    return {key: value for key, value in r.items() if key not in {"x", "y", "e"}}


def pair(d, tag, time_limit=60):
    qp = solve(d, False, tag + "_qp", time_limit)
    miqp = solve(d, True, tag + "_miqp", time_limit)
    result = {"tag": tag, "case": d.name, "N": d.N, "T": d.T, "alpha": d.alpha,
              "rho": float(d.rho[0]), "capacity_mode": d.capacity_mode, **certificate(d)}
    for prefix, solution in (("qp", qp), ("miqp", miqp)):
        result.update({f"{prefix}_{key}": value for key, value in flat(solution).items()
                       if key not in {"tag", "case", "N", "T"}})
    both_optimal = qp["optimal"] and miqp["optimal"]
    gap = miqp["objective"] - qp["objective"] if both_optimal else None
    result["objective_gap"] = gap
    if not qp["has_solution"]:
        classification = "relaxation_infeasible_or_error"
    elif miqp["status"] == "infeasible":
        classification = "infeasibility_masked"
    elif not both_optimal:
        classification = "uncertified_solver_status"
    elif gap > max(1e-5, 1e-7 * abs(miqp["objective"])):
        classification = "positive_value_gap"
    elif qp["max_simultaneous_kw"] > 1e-5:
        classification = "equal_value_simultaneous_solution"
    elif result["certificate"]:
        classification = "certified_exact"
    else:
        classification = "observed_exact_uncertified"
    result["classification"] = classification
    return result


def fit_ecdf(training):
    values = np.asarray(training, dtype=float)
    if values.ndim != 2 or not len(values) or not np.isfinite(values).all():
        raise ValueError("Finite nonempty training paths are required.")
    return np.sort(values, axis=0)


def apply_ecdf(fitted, paths):
    values = np.asarray(paths, dtype=float)
    return np.column_stack([
        np.searchsorted(fitted[:, hour], values[:, hour], side="right") / len(fitted)
        for hour in range(values.shape[1])
    ])


def conformal_quantile(scores, epsilon):
    ordered = np.sort(np.asarray(scores, dtype=float))
    n = len(ordered)
    if n == 0 or not 0 < epsilon < 1:
        raise ValueError("Nonempty scores and 0 < epsilon < 1 required.")
    rank = int(np.ceil((n + 1) * (1 - epsilon)))
    return float("inf") if rank > n else float(ordered[rank - 1])


def uniform_paths(samples, T, correlation, seed):
    """Generate Gaussian AR(1) copula paths with uniform marginal distributions."""
    rng = np.random.default_rng(seed)
    latent = rng.normal(size=(samples, T))
    innovation_weight = np.sqrt(1 - correlation**2)
    for t in range(1, T):
        latent[:, t] = correlation * latent[:, t - 1] + innovation_weight * latent[:, t]
    return norm.cdf(latent)


def wilson(successes, total):
    if total == 0:
        return float("nan"), float("nan")
    p = successes / total
    z = 1.95996398454
    denominator = 1 + z**2 / total
    center = (p + z**2 / (2 * total)) / denominator
    half_width = z * np.sqrt(p * (1 - p) / total + z**2 / (4 * total**2)) / denominator
    return float(center - half_width), float(center + half_width)
