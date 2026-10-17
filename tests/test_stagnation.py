"""Unit tests for stagnation detection and Adaptive LEO algorithm."""

import numpy as np
import pytest

from leo.core.adaptive_leo import AdaptiveLEOAlgorithm
from leo.core.stagnation import CallingStrategy, StagnationConfig, StagnationDetector
from leo.llm.client import LLMClient, LLMSettings
from leo.problems.benchmarks_2d import Rosenbrock2DProblem


def test_stagnation_detector_progress_and_flat() -> None:
    config = StagnationConfig(window_size=3, rel_tol=1e-3, abs_tol=1e-5, cooldown=1)
    detector = StagnationDetector(config=config)

    # Initially no stagnation with insufficient history
    detector.record(10.0)
    stagnated, _ = detector.is_stagnated()
    assert not stagnated

    # Progressing history: 10.0 -> 8.0 -> 5.0 -> 2.0
    detector.record(8.0)
    detector.record(5.0)
    detector.record(2.0)
    stagnated, rel_imp = detector.is_stagnated()
    assert not stagnated
    assert rel_imp > 0.5  # Significant progress (10.0 -> 2.0)

    # Flat history: 2.0 -> 2.0 -> 2.0 -> 2.0
    detector.record(2.0)
    detector.record(2.0)
    detector.record(2.0)
    stagnated, rel_imp = detector.is_stagnated()
    assert stagnated
    assert rel_imp == 0.0


def test_calling_strategies() -> None:
    config = StagnationConfig(
        window_size=2,
        rel_tol=1e-3,
        cooldown=1,
        periodic_interval=3,
        warmup_llm=True,
    )

    # 1. EVERY strategy
    det = StagnationDetector(config)
    for it in range(1, 6):
        should_call, reason = det.should_call_llm(it, CallingStrategy.EVERY)
        assert should_call
        assert reason == "every_iteration"

    # 2. NEVER strategy
    det = StagnationDetector(config)
    for it in range(1, 6):
        should_call, reason = det.should_call_llm(it, CallingStrategy.NEVER)
        assert not should_call
        assert reason == "never_call"

    # 3. PERIODIC strategy (interval=3)
    det = StagnationDetector(config)
    calls = [
        det.should_call_llm(it, CallingStrategy.PERIODIC)[0] for it in range(1, 10)
    ]
    # Iterations 1..9: calls at 3, 6, 9
    assert calls == [False, False, True, False, False, True, False, False, True]

    # 4. STAGNATION strategy
    det = StagnationDetector(config)
    # Iteration 1: Warmup call
    should_call, reason = det.should_call_llm(1, CallingStrategy.STAGNATION)
    assert should_call
    assert reason == "warmup_initial"

    # Iteration 2: Cooldown active (last_iter=1, cooldown=1)
    det.record(10.0)
    should_call, reason = det.should_call_llm(2, CallingStrategy.STAGNATION)
    assert not should_call
    assert "cooldown_active" in reason

    # Feed flat values to induce stagnation
    det.record(10.0)
    det.record(10.0)
    should_call, reason = det.should_call_llm(3, CallingStrategy.STAGNATION)
    assert should_call
    assert "stagnation_detected" in reason


def test_adaptive_leo_algorithm_offline() -> None:
    problem = Rosenbrock2DProblem()
    settings = LLMSettings(leo_mock_llm=True)
    client = LLMClient(settings=settings)
    pop_size = 6
    max_iters = 6

    # Test EVERY strategy
    algo_every = AdaptiveLEOAlgorithm(
        problem=problem,
        strategy=CallingStrategy.EVERY,
        pop_size=pop_size,
        max_iters=max_iters,
        llm_client=client,
        seed=42,
    )
    res_every = algo_every.optimize()
    assert res_every.total_llm_queries == 2 * max_iters
    assert len(res_every.llm_call_iterations) == max_iters
    assert res_every.total_evaluations == pop_size + 2 * pop_size * max_iters

    # Test NEVER strategy
    algo_never = AdaptiveLEOAlgorithm(
        problem=problem,
        strategy=CallingStrategy.NEVER,
        pop_size=pop_size,
        max_iters=max_iters,
        llm_client=client,
        seed=42,
    )
    res_never = algo_never.optimize()
    assert res_never.total_llm_queries == 0
    assert len(res_never.llm_call_iterations) == 0
    assert res_never.total_evaluations == pop_size + 2 * pop_size * max_iters

    # Test PERIODIC strategy (interval=3)
    cfg_periodic = StagnationConfig(periodic_interval=3)
    algo_periodic = AdaptiveLEOAlgorithm(
        problem=problem,
        strategy=CallingStrategy.PERIODIC,
        stagnation_config=cfg_periodic,
        pop_size=pop_size,
        max_iters=max_iters,
        llm_client=client,
        seed=42,
    )
    res_periodic = algo_periodic.optimize()
    # In 6 iters, called at 3 and 6 -> 2 times -> 4 queries
    assert res_periodic.llm_call_iterations == [3, 6]
    assert res_periodic.total_llm_queries == 4

    # Test STAGNATION strategy
    algo_stag = AdaptiveLEOAlgorithm(
        problem=problem,
        strategy=CallingStrategy.STAGNATION,
        pop_size=pop_size,
        max_iters=max_iters,
        llm_client=client,
        seed=42,
    )
    res_stag = algo_stag.optimize()
    assert res_stag.total_llm_queries <= res_every.total_llm_queries
    assert res_stag.total_evaluations == pop_size + 2 * pop_size * max_iters
    assert np.all(res_stag.best_x >= problem.bounds[0])
    assert np.all(res_stag.best_x <= problem.bounds[1])
