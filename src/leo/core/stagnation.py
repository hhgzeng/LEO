"""Stagnation detection and LLM invocation strategy controllers for LEO.

Enables adaptive, demand-driven invocation of LLM reasoning based on optimization progress.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class CallingStrategy(str, Enum):
    """LLM invocation strategy mode."""

    EVERY = "every"  # Call LLM on every iteration (Standard LEO baseline)
    PERIODIC = "periodic"  # Call LLM every k iterations
    STAGNATION = "stagnation"  # Call LLM adaptively when stagnation is detected
    NEVER = "never"  # Never call LLM (pure heuristic perturbation baseline)


@dataclass
class StagnationConfig:
    """Hyperparameters for stagnation detection and adaptive invocation."""

    window_size: int = 3  # Lookback window W for stagnation detection
    rel_tol: float = 1e-4  # Relative improvement tolerance
    abs_tol: float = 1e-6  # Absolute improvement tolerance
    cooldown: int = 1  # Minimum iterations between LLM calls in stagnation mode
    periodic_interval: int = 3  # Interval k for CallingStrategy.PERIODIC
    warmup_llm: bool = True  # Whether to always call LLM on iteration 1 for initialization


class StagnationDetector:
    """Tracks fitness progress and decides when to trigger LLM reasoning."""

    def __init__(self, config: StagnationConfig | None = None) -> None:
        self.config = config or StagnationConfig()
        self.history_best_y: list[float] = []
        self.last_llm_iter: int = -999

    def record(self, best_y: float) -> None:
        """Record the best-so-far objective loss at the current step."""
        self.history_best_y.append(float(best_y))

    def reset(self) -> None:
        """Reset internal history and state."""
        self.history_best_y.clear()
        self.last_llm_iter = -999

    def calculate_relative_improvement(self) -> tuple[float, float, float]:
        """Compute relative improvement over the sliding lookback window.

        Returns:
            (rel_improvement, old_best, current_best)
        """
        if len(self.history_best_y) <= self.config.window_size:
            return float("inf"), float("nan"), float("nan")

        old_best = self.history_best_y[-(self.config.window_size + 1)]
        current_best = self.history_best_y[-1]
        diff = old_best - current_best

        # Relative improvement normalized by max(1.0, |old_best|)
        scale = max(1.0, abs(old_best))
        rel_improvement = diff / scale
        return rel_improvement, old_best, current_best

    def is_stagnated(self) -> tuple[bool, float]:
        """Check whether optimization has stagnated over the sliding window.

        Returns:
            (is_stagnated, relative_improvement)
        """
        if len(self.history_best_y) <= self.config.window_size:
            return False, float("inf")

        rel_imp, old_best, cur_best = self.calculate_relative_improvement()
        abs_diff = old_best - cur_best

        # Stagnation occurs if neither relative nor absolute improvement threshold is met
        stagnated = (rel_imp <= self.config.rel_tol) and (abs_diff <= self.config.abs_tol)
        return bool(stagnated), rel_imp

    def should_call_llm(
        self, iteration: int, strategy: CallingStrategy | str
    ) -> tuple[bool, str]:
        """Determine whether the current iteration should invoke the LLM.

        Args:
            iteration: Current 1-based iteration number (1..max_iters).
            strategy: LLM invocation strategy.

        Returns:
            (should_call: bool, reason: str)
        """
        strat = CallingStrategy(strategy) if isinstance(strategy, str) else strategy

        if strat == CallingStrategy.EVERY:
            self.last_llm_iter = iteration
            return True, "every_iteration"

        if strat == CallingStrategy.NEVER:
            return False, "never_call"

        if strat == CallingStrategy.PERIODIC:
            if iteration % self.config.periodic_interval == 0:
                self.last_llm_iter = iteration
                return True, f"periodic_interval_{self.config.periodic_interval}"
            return False, f"periodic_skip_{iteration % self.config.periodic_interval}"

        if strat == CallingStrategy.STAGNATION:
            # 1. Warmup call on iteration 1 if configured
            if iteration == 1 and self.config.warmup_llm:
                self.last_llm_iter = iteration
                return True, "warmup_initial"

            # 2. Cooldown guardrail
            if iteration - self.last_llm_iter <= self.config.cooldown:
                return False, f"cooldown_active_(last_iter={self.last_llm_iter})"

            # 3. Check for stagnation over sliding window
            stagnated, rel_imp = self.is_stagnated()
            if stagnated:
                self.last_llm_iter = iteration
                return True, f"stagnation_detected_(rel_imp={rel_imp:.2e})"

            return False, f"making_progress_(rel_imp={rel_imp:.2e})"

        raise ValueError(f"Unknown calling strategy: {strategy}")
