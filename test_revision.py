from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from threading import Barrier

import numpy as np
import pytest

import revision_model as model


@pytest.fixture(autouse=True)
def temporary_solver_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(model, "ROOT", tmp_path)


def tiny(**changes):
    instance = model.Instance(
        name="analytic", L=np.array([0.]), buy=np.array([1.]), sell=np.array([1.]),
        availability=np.ones((1, 1)), pcmax=np.array([10.]), pdmax=np.array([10.]),
        e0=np.array([5.]), emin=np.array([0.]), emax=np.array([5.]), target=np.array([5.]),
        etac=np.array([.95]), etad=np.array([.9]), rho=np.array([0.]),
        coeff=np.ones((1, 1)), capacity=100., alpha=0.,
    )
    return replace(instance, **changes)


def test_holdout_cannot_change_ecdf():
    fitted = model.fit_ecdf([[0., 2.], [1., 4.], [3., 6.]])
    first = model.apply_ecdf(fitted, [[2., 5.]])
    second = model.apply_ecdf(fitted, [[2., 5.], [999., -999.]])
    np.testing.assert_array_equal(first, second[:1])
    np.testing.assert_allclose(first, [[2 / 3, 2 / 3]])


def test_quantile_finite_sample_boundary():
    assert np.isinf(model.conformal_quantile(np.arange(5), .05))
    assert model.conformal_quantile(np.arange(240), .05) == 228


def test_negative_price_analytic_gap():
    instance = tiny(buy=np.array([-1.]), sell=np.array([-1.]))
    pair = model.pair(instance, "test_negative")
    assert pair["qp_optimal"] and pair["miqp_optimal"]
    assert pair["objective_gap"] == pytest.approx(1.45, abs=1e-6)
    assert pair["classification"] == "positive_value_gap"


def test_export_limit_masks_infeasibility():
    instance = tiny(L=np.array([-1.]), capacity_mode="two_sided", rho=np.array([.01]))
    pair = model.pair(instance, "test_export")
    assert pair["classification"] == "infeasibility_masked"
    assert not pair["constraint_direction_safe"]


def test_positive_price_zero_penalty_is_certified():
    pair = model.pair(tiny(), "test_zero_rho")
    assert pair["certificate"] and pair["classification"] == "certified_exact"


def test_timestep_energy_and_objective():
    instance = tiny(emax=np.array([10.]), target=np.array([6.]), dt=.25)
    solution = model.solve(instance, binary=True, tag="test_dt")
    assert solution["optimal"]
    assert solution["charge_kwh"] == pytest.approx(1 / .95, abs=1e-6)
    assert solution["objective"] == pytest.approx(1 / .95, abs=1e-6)


def test_cycle_repair_preserves_soc_and_detects_export_violation():
    instance = tiny(L=np.array([-1.]), capacity_mode="two_sided")
    charge = np.array([[1 / (1 - .855)]])
    discharge = .855 * charge
    repaired = model.cycle_repair(instance, charge, discharge)
    np.testing.assert_allclose(repaired[2], [[5, 5]])
    assert model.feasibility_residual(instance, *repaired, complementarity=True) > .99


def test_copula_uniform_marginals_and_dependence():
    paths = model.uniform_paths(20000, 4, .8, 7)
    assert np.max(abs(paths.mean(axis=0) - .5)) < .01
    assert np.corrcoef(paths.T)[0, 1] > .7


def test_infeasible_relaxation_and_inactive_power_bounds():
    # A disconnected vehicle can retain its current energy, but cannot reach a higher target.
    idle = tiny(availability=np.zeros((1, 1)))
    result = model.solve(idle)
    assert result["optimal"] and result["certificate"]
    assert result["directional_margin"] is None
    pair = model.pair(replace(idle, target=np.array([6.])), "unreachable")
    assert pair["qp_status"] == pair["miqp_status"] == "infeasible"
    assert pair["objective_gap"] is None


def test_forced_native_solver_error_discards_stale_schedule_and_can_retry(monkeypatch):
    solution = model.solve(tiny())
    folder = model.ROOT / "runs" / "solves" / solution["key"]

    def license_error(directory, time_limit):
        (directory / "solver.log").write_text("CPLEX Error 1016: size limit exceeded.")
        return .01, 0., False, 1

    with monkeypatch.context() as patch:
        patch.setattr(model, "_run_cplex", license_error)
        failed = model.solve(tiny(), force=True)
    assert failed["status"] == "license_size_error"
    assert not failed["optimal"] and not failed["has_solution"]
    assert not (folder / "solution.sol").exists()
    assert not (folder / "schedule.npz").exists()
    assert model.solve(tiny())["optimal"]


def test_concurrent_identical_solves_share_one_record(monkeypatch):
    calls = []
    run_cplex = model._run_cplex

    def count_native_runs(folder, time_limit):
        calls.append(folder)
        return run_cplex(folder, time_limit)

    start = Barrier(2)

    def solve_from_cold_cache(tag):
        start.wait(timeout=10)
        return model.solve(tiny(), tag=tag)

    monkeypatch.setattr(model, "_run_cplex", count_native_runs)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(solve_from_cold_cache, ["first", "second"]))
    assert len(calls) == 1
    assert [result["tag"] for result in results] == ["first", "second"]
    assert results[0]["key"] == results[1]["key"]


def test_time_limited_cplex_xml_without_primal_solution(tmp_path):
    (tmp_path / "solution.sol").write_text(
        '<CPLEXSolution><header solutionStatusValue="108" '
        'solutionStatusString="time limit exceeded" primalFeasible="0"/></CPLEXSolution>'
    )
    result = {"has_solution": False, "optimal": False, "objective": None}
    model._read_solution(tiny(), result, tmp_path, (), "")
    assert result["status_code"] == "108"
    assert not result["has_solution"] and not result["optimal"]
    assert result["objective"] is None
