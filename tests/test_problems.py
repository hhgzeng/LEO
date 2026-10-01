"""Unit tests verifying mathematical fidelity of all benchmark and engineering functions."""

import numpy as np

from leo.problems.benchmarks_2d import (
    BealeProblem,
    GoldsteinPriceProblem,
    HimmelblauProblem,
    Rosenbrock2DProblem,
    ScaledSphereProblem,
    Sphere2DProblem,
)
from leo.problems.engineering import (
    HeatTransferProblem,
    NoseConeProblem,
    NozzleShapeProblem,
    WindFarmProblem,
)
from leo.problems.multi_objective import ZDT1Problem, ZDT3Problem
from leo.problems.rosenbrock_nd import ShiftedRosenbrockND


def test_2d_benchmarks_global_minima() -> None:
    # 1. Sphere2D: f(0, 0) = 0
    p_sphere = Sphere2DProblem()
    assert p_sphere.evaluate(np.array([0.0, 0.0])) == 0.0

    # 2. ScaledSphere: f(0, 0) = 0
    p_scaled = ScaledSphereProblem()
    assert p_scaled.evaluate(np.array([0.0, 0.0])) == 0.0

    # 3. Himmelblau: f(3.0, 2.0) = 0
    p_himmel = HimmelblauProblem()
    assert p_himmel.evaluate(np.array([3.0, 2.0])) == 0.0

    # 4. Rosenbrock 2D: f(1.0, 1.0) = 0
    p_rosen = Rosenbrock2DProblem()
    assert p_rosen.evaluate(np.array([1.0, 1.0])) == 0.0

    # 5. Beale: f(3.0, 0.5) = 0
    p_beale = BealeProblem()
    np.testing.assert_allclose(p_beale.evaluate(np.array([3.0, 0.5])), 0.0, atol=1e-12)

    # 6. Goldstein-Price: f(0.0, -1.0) = 3.0
    p_gp = GoldsteinPriceProblem()
    np.testing.assert_allclose(p_gp.evaluate(np.array([0.0, -1.0])), 3.0, atol=1e-12)


def test_shifted_rosenbrock_nd() -> None:
    dim = 5
    a = 0.2913
    prob = ShiftedRosenbrockND(dim=dim, a=a)
    opt_x = np.full(dim, 1.0 + a)
    np.testing.assert_allclose(prob.evaluate(opt_x), 0.0, atol=1e-12)


def test_multi_objective_zdt() -> None:
    zdt1 = ZDT1Problem(dim=2)
    eval_origin = zdt1.evaluate(np.array([0.0, 0.0]))
    # at x1=0, x2=0: f1 = 0, g = 1, f2 = 1*(1 - 0) = 1
    np.testing.assert_allclose(eval_origin, [0.0, 1.0], atol=1e-6)

    zdt3 = ZDT3Problem(dim=2)
    eval_z3 = zdt3.evaluate(np.array([0.0, 0.0]))
    np.testing.assert_allclose(eval_z3, [0.0, 1.0], atol=1e-6)


def test_engineering_problems() -> None:
    # 1. Nozzle
    nozzle = NozzleShapeProblem()
    res_nozzle = nozzle.evaluate(np.array([0.2, 0.6, 0.8, 0.9]))
    assert res_nozzle >= 0.0

    # 2. Heat transfer
    heat = HeatTransferProblem(dim=4)
    exact_x = heat.optimum_x
    res_heat = heat.evaluate(exact_x)
    # Exact analytical solution has near zero residual
    np.testing.assert_allclose(res_heat, 0.0, atol=1e-1)

    # 3. Wind farm
    wind = WindFarmProblem(n_turbines=2)
    res_wind = wind.evaluate(np.array([100.0, 100.0, 800.0, 800.0]))
    assert 0.0 <= res_wind <= 2.0

    # 4. Nosecone
    nose = NoseConeProblem()
    cd = nose.evaluate(np.array([0.75]))
    assert 0.0 < cd < 1.0
