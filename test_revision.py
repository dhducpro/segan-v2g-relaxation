import numpy as np
from dataclasses import replace
import revision_model as r


def tiny(**changes):
    d=r.Instance("analytic",np.array([0.]),np.array([1.]),np.array([1.]),np.ones((1,1)),
                 np.array([10.]),np.array([10.]),np.array([5.]),np.array([0.]),np.array([5.]),np.array([5.]),
                 np.array([.95]),np.array([.9]),np.array([0.]),np.ones((1,1)),100.,alpha=0.)
    return replace(d,**changes)


def test_holdout_cannot_change_ecdf():
    train=np.array([[0.,2.],[1.,4.],[3.,6.]])
    fitted=r.fit_ecdf(train)
    first=r.apply_ecdf(fitted,np.array([[2.,5.]]))
    second=r.apply_ecdf(fitted,np.array([[2.,5.],[999.,-999.]]))[:1]
    np.testing.assert_array_equal(first,second)
    np.testing.assert_allclose(first,[[2/3,2/3]])


def test_quantile_finite_sample_boundary():
    assert np.isinf(r.conformal_quantile(np.arange(5),.05))
    assert r.conformal_quantile(np.arange(240),.05)==228


def test_negative_price_analytic_gap():
    d=tiny(buy=np.array([-1.]),sell=np.array([-1.]))
    z=r.pair(d,"test_negative")
    assert z["qp_optimal"] and z["miqp_optimal"]
    assert abs(z["objective_gap"]-1.45)<1e-6
    assert z["classification"]=="positive_value_gap"


def test_export_limit_masks_infeasibility():
    d=tiny(L=np.array([-1.]),capacity_mode="two_sided",rho=np.array([.01]))
    z=r.pair(d,"test_export")
    assert z["classification"]=="infeasibility_masked"
    assert not z["constraint_direction_safe"]


def test_positive_price_zero_penalty_is_certified():
    z=r.pair(tiny(),"test_zero_rho")
    assert z["certificate"] and z["classification"]=="certified_exact"


def test_timestep_energy_and_objective():
    d=tiny(emax=np.array([10.]),target=np.array([6.]),dt=.25)
    z=r.solve(d,True,"test_dt")
    assert z["optimal"]
    assert abs(z["charge_kwh"]-1/.95)<1e-6
    assert abs(z["objective"]-1/.95)<1e-6


def test_cycle_repair_preserves_soc_and_detects_export_violation():
    d=tiny(L=np.array([-1.]),capacity_mode="two_sided")
    x=np.array([[1/(1-.855)]]);y=.855*x
    xx,yy,ee=r.cycle_repair(d,x,y)
    np.testing.assert_allclose(ee,[[5,5]])
    assert r.feasibility_residual(d,xx,yy,ee,True)>.99


def test_copula_uniform_marginals_and_dependence():
    paths=r.uniform_paths(20000,4,.8,7)
    assert np.max(abs(paths.mean(axis=0)-.5))<.01
    assert np.corrcoef(paths.T)[0,1]>.7
