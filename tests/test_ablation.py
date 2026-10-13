"""Unit tests for AblationLEOAlgorithm modes."""

from __future__ import annotations

import numpy as np

from leo.core.ablation import AblationLEOAlgorithm, AblationMode
from leo.llm.client import LLMClient, LLMSettings
from leo.problems.benchmarks_2d import GoldsteinPriceProblem, Rosenbrock2DProblem
from leo.problems.rosenbrock_nd import ShiftedRosenbrockND


def test_ablation_full_mode() -> None:
    settings = LLMSettings(leo_mock_llm=True)
    client = LLMClient(settings=settings)
    problem = Rosenbrock2DProblem()

    algo = AblationLEOAlgorithm(
        problem=problem,
        mode=AblationMode.FULL,
        pop_size=6,
        max_iters=3,
        llm_client=client,
        seed=42,
    )
    res = algo.optimize()

    assert res.mode == "full"
    assert res.iterations == 3
    assert res.total_evaluations == 6 + 3 * (6 + 6)  # 42
    assert len(res.history_overall_best_y) == 4
    assert res.best_x.shape == (2,)
    assert res.best_y <= res.history_overall_best_y[0]


def test_ablation_no_history_mode() -> None:
    settings = LLMSettings(leo_mock_llm=True)
    client = LLMClient(settings=settings)
    problem = GoldsteinPriceProblem()

    algo = AblationLEOAlgorithm(
        problem=problem,
        mode=AblationMode.NO_HISTORY,
        pop_size=6,
        max_iters=3,
        llm_client=client,
        seed=42,
    )
    res = algo.optimize()

    assert res.mode == "no_history"
    assert res.iterations == 3
    assert res.total_evaluations == 6 + 3 * (6 + 6)
    assert res.best_x.shape == (2,)


def test_ablation_no_elitism_mode() -> None:
    settings = LLMSettings(leo_mock_llm=True)
    client = LLMClient(settings=settings)
    problem = Rosenbrock2DProblem()

    algo = AblationLEOAlgorithm(
        problem=problem,
        mode=AblationMode.NO_ELITISM,
        pop_size=6,
        max_iters=3,
        llm_client=client,
        seed=101,
    )
    res = algo.optimize()

    assert res.mode == "no_elitism"
    assert res.iterations == 3
    assert res.total_evaluations == 6 + 3 * (6 + 6)
    assert res.best_x.shape == (2,)


def test_ablation_random_perturb_mode() -> None:
    # Random perturbation does not even call LLM
    problem = ShiftedRosenbrockND(dim=10, a=0.2913)

    algo = AblationLEOAlgorithm(
        problem=problem,
        mode=AblationMode.RANDOM_PERTURB,
        pop_size=6,
        max_iters=3,
        seed=202,
    )
    res = algo.optimize()

    assert res.mode == "random_perturb"
    assert res.iterations == 3
    assert res.total_evaluations == 6 + 3 * (6 + 6)
    assert res.best_x.shape == (10,)
    assert np.all(res.best_x >= -2.0) and np.all(res.best_x <= 2.0)
