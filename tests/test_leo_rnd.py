"""Tests for LEORndOptimizer ablation baseline."""

from __future__ import annotations

import numpy as np

from leo.baselines.leo_rnd import LEORndOptimizer
from leo.llm.client import LLMClient, LLMSettings
from leo.problems.benchmarks_2d import GoldsteinPriceProblem, Rosenbrock2DProblem
from leo.problems.rosenbrock_nd import ShiftedRosenbrockND


def test_leo_rnd_mock_execution() -> None:
    settings = LLMSettings(leo_mock_llm=True)
    client = LLMClient(settings=settings)
    problem = GoldsteinPriceProblem()

    optimizer = LEORndOptimizer(
        problem=problem,
        pop_size=6,
        max_iters=5,
        sol_to_port=1,
        llm_client=client,
        seed=42,
    )

    res = optimizer.optimize()

    assert res.iterations == 5
    assert res.total_evaluations == 6 + 2 * 6 * 5  # 66
    assert len(res.history_exploit_best_y) == 6
    assert res.best_y <= res.history_exploit_best_y[0]
    assert res.best_x.shape == (2,)


def test_leo_rnd_mock_multi_iterations_progress() -> None:
    settings = LLMSettings(leo_mock_llm=True)
    client = LLMClient(settings=settings)
    problem = Rosenbrock2DProblem()

    optimizer = LEORndOptimizer(
        problem=problem,
        pop_size=10,
        max_iters=10,
        sol_to_port=1,
        llm_client=client,
        seed=123,
    )

    res = optimizer.optimize()
    # Ensure port_and_filter and dynamic generation allow valid population shapes
    assert len(res.history_explore_x) == 11
    assert len(res.history_exploit_x) == 11
    assert res.history_exploit_best_y[-1] <= res.history_exploit_best_y[0]


def test_leo_rnd_10d_mock() -> None:
    settings = LLMSettings(leo_mock_llm=True)
    client = LLMClient(settings=settings)
    problem = ShiftedRosenbrockND(dim=10, a=0.2913)

    optimizer = LEORndOptimizer(
        problem=problem,
        pop_size=8,
        max_iters=4,
        sol_to_port=1,
        llm_client=client,
        seed=99,
    )

    res = optimizer.optimize()
    assert res.best_x.shape == (10,)
    assert np.all(res.best_x >= -2.0) and np.all(res.best_x <= 2.0)
