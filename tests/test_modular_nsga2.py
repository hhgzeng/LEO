"""Integration tests for LEO-modular NSGA-II."""

import numpy as np

from leo.llm.client import LLMClient, LLMSettings
from leo.modular.nsga2 import (
    LEOModularNSGA2,
    calculate_crowding_distance,
    fast_non_dominated_sort,
)
from leo.problems.multi_objective import ZDT1Problem


def test_fast_non_dominated_sort() -> None:
    # 3 points: p0=(1, 2), p1=(2, 1), p2=(3, 3)
    # p0 and p1 are non-dominated; p2 is dominated by both
    objs = np.array([[1.0, 2.0], [2.0, 1.0], [3.0, 3.0]])
    fronts = fast_non_dominated_sort(objs)

    assert len(fronts) == 2
    assert set(fronts[0]) == {0, 1}
    assert fronts[1] == [2]


def test_crowding_distance() -> None:
    objs = np.array([[0.0, 1.0], [0.5, 0.5], [1.0, 0.0]])
    front = [0, 1, 2]
    dists = calculate_crowding_distance(objs, front)

    assert dists[0] == np.inf
    assert dists[2] == np.inf
    assert np.isfinite(dists[1])


def test_leo_modular_nsga2_execution() -> None:
    problem = ZDT1Problem(dim=2)
    settings = LLMSettings(leo_mock_llm=True)
    client = LLMClient(settings=settings)

    algo = LEOModularNSGA2(
        problem=problem,
        pop_size=6,
        max_generations=3,
        llm_client=client,
        seed=42,
    )
    res = algo.optimize()

    assert len(res.pareto_f) > 0
    assert res.pareto_f.shape[1] == 2
    assert len(res.history_pareto_f) == 3
