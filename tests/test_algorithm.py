"""End-to-end integration tests for LEOAlgorithm execution loop."""

from leo.core.algorithm import LEOAlgorithm
from leo.llm.client import LLMClient, LLMSettings
from leo.problems.benchmarks_2d import BealeProblem, Sphere2DProblem


def test_leo_algorithm_sphere() -> None:
    problem = Sphere2DProblem()
    settings = LLMSettings(leo_mock_llm=True)
    client = LLMClient(settings=settings)

    algo = LEOAlgorithm(
        problem=problem,
        pop_size=6,
        max_iters=5,
        llm_client=client,
        seed=42,
        verbose=False,
    )
    result = algo.optimize()

    assert result.iterations == 5
    assert len(result.history_exploit_best_y) == 6  # init + 5 iters
    assert len(result.history_explore_x) == 6
    assert result.total_evaluations == 6 + 5 * (6 + 6)  # 66 evals
    assert result.best_y <= result.history_exploit_best_y[0]


def test_leo_algorithm_beale() -> None:
    problem = BealeProblem()
    settings = LLMSettings(leo_mock_llm=True)
    client = LLMClient(settings=settings)

    algo = LEOAlgorithm(
        problem=problem,
        pop_size=6,
        max_iters=5,
        llm_client=client,
        seed=123,
        verbose=False,
    )
    result = algo.optimize()

    assert result.best_y >= 0.0
    assert result.best_x.shape == (2,)
