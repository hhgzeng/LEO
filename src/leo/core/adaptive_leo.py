"""Adaptive LEO (Large Language Model-Based Evolutionary Optimizer).

Supports adaptive, demand-driven LLM invocation strategies:
- Every iteration (Standard baseline)
- Fixed periodic interval (e.g. every k generations)
- Stagnation-triggered (Only invoke LLM when search progress stagnates)
- Never (Zero-LLM classical perturbation baseline)
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)

from leo.core.jitter import inject_jitter
from leo.core.population import Population, port_and_filter
from leo.core.stagnation import CallingStrategy, StagnationConfig, StagnationDetector
from leo.llm.client import LLMClient
from leo.llm.parser import parse_candidate_csv
from leo.llm.prompts import build_exploit_prompt, build_explore_prompt
from leo.problems.base import BaseProblem


@dataclass
class AdaptiveLEOResult:
    """Optimization result container for adaptive LEO runs."""

    strategy: str
    best_x: np.ndarray
    best_y: float
    history_exploit_best_y: list[float] = field(default_factory=list)
    history_exploit_mean_y: list[float] = field(default_factory=list)
    history_explore_best_y: list[float] = field(default_factory=list)
    history_explore_mean_y: list[float] = field(default_factory=list)
    history_overall_best_y: list[float] = field(default_factory=list)
    history_explore_x: list[np.ndarray] = field(default_factory=list)
    history_exploit_x: list[np.ndarray] = field(default_factory=list)
    llm_call_iterations: list[int] = field(default_factory=list)
    trigger_reasons: dict[int, str] = field(default_factory=dict)
    total_llm_queries: int = 0
    total_evaluations: int = 0
    iterations: int = 0


class AdaptiveLEOAlgorithm:
    """Adaptive LEO algorithm supporting multiple LLM invocation strategies."""

    def __init__(
        self,
        problem: BaseProblem,
        strategy: CallingStrategy | str = CallingStrategy.STAGNATION,
        stagnation_config: StagnationConfig | None = None,
        pop_size: int = 10,
        max_iters: int = 30,
        sol_to_port: int = 1,
        jitter_scale: float = 1e-4,
        llm_client: LLMClient | None = None,
        seed: int | None = None,
        temperature: float = 0.7,
        perturb_sigma_scale: float = 0.05,
        bypass_cache: bool = False,
        verbose: bool = False,
    ) -> None:
        self.problem = problem
        self.strategy = (
            CallingStrategy(strategy) if isinstance(strategy, str) else strategy
        )
        self.stagnation_config = stagnation_config or StagnationConfig()
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.sol_to_port = sol_to_port
        self.jitter_scale = jitter_scale
        self.llm_client = llm_client or LLMClient()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.temperature = temperature
        self.perturb_sigma_scale = perturb_sigma_scale
        self.bypass_cache = bypass_cache
        self.verbose = verbose

    def optimize(self) -> AdaptiveLEOResult:
        """Run adaptive optimization loop."""
        # 1. Initialization: Sample initial population
        x_init = self.problem.sample_uniform(self.pop_size, rng=self.rng)
        y_init = self.problem.evaluate(x_init)
        total_evals = len(x_init)

        x_explore = x_init.copy()
        y_explore = y_init.copy()
        x_exploit = x_init.copy()
        y_exploit = y_init.copy()

        explore_pop = Population(x=x_explore, y=y_explore).sorted()
        exploit_pop = Population(x=x_exploit, y=y_exploit).sorted()

        overall_best = min(exploit_pop.best_y, explore_pop.best_y)

        # Initialize stagnation detector and record initial state
        detector = StagnationDetector(config=self.stagnation_config)
        detector.record(exploit_pop.best_y)

        # Tracking metrics
        hist_exploit_best = [exploit_pop.best_y]
        hist_exploit_mean = [float(np.mean(exploit_pop.y))]
        hist_explore_best = [explore_pop.best_y]
        hist_explore_mean = [float(np.mean(explore_pop.y))]
        hist_overall_best = [overall_best]
        hist_explore_x = [explore_pop.x.copy()]
        hist_exploit_x = [exploit_pop.x.copy()]

        llm_call_iterations: list[int] = []
        trigger_reasons: dict[int, str] = {}
        total_llm_queries = 0

        progress = Progress(
            SpinnerColumn(),
            TextColumn(
                f"[bold cyan]AdaptiveLEO [{self.strategy.value}|{self.problem.name}][/bold cyan]"
            ),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("Best: {task.fields[best_loss]:.6f}"),
            disable=not self.verbose,
        )

        with progress:
            task_id = progress.add_task(
                "optimizing", total=self.max_iters, best_loss=overall_best
            )

            for iteration in range(1, self.max_iters + 1):
                # 2. Decide whether to call LLM or use routine evolutionary operator
                call_llm, reason = detector.should_call_llm(iteration, self.strategy)

                if call_llm:
                    llm_call_iterations.append(iteration)
                    trigger_reasons[iteration] = reason
                    total_llm_queries += 2

                    # Jitter injection on candidate coordinates
                    explore_x_jitter = inject_jitter(
                        explore_pop.x,
                        jitter_scale=self.jitter_scale,
                        bounds=self.problem.bounds,
                        rng=self.rng,
                    )
                    exploit_best_jitter = inject_jitter(
                        exploit_pop.best_x,
                        jitter_scale=self.jitter_scale,
                        bounds=self.problem.bounds,
                        rng=self.rng,
                    )

                    # Build prompts
                    prompt_explore = build_explore_prompt(
                        x_jitter=explore_x_jitter,
                        y=explore_pop.y,
                        num_new=self.pop_size,
                        bounds=self.problem.bounds,
                    )
                    prompt_exploit = build_exploit_prompt(
                        best_x_jitter=exploit_best_jitter,
                        best_y=exploit_pop.best_y,
                        num_new=self.pop_size,
                        bounds=self.problem.bounds,
                    )

                    cache_extras = [
                        f"adaptive_{self.strategy.value}_explore_s{self.seed}_i{iteration}",
                        f"adaptive_{self.strategy.value}_exploit_s{self.seed}_i{iteration}",
                    ]
                    resp_explore, resp_exploit = self.llm_client.query_parallel(
                        [prompt_explore, prompt_exploit],
                        temperature=self.temperature,
                        cache_key_extras=cache_extras,
                        bypass_cache=self.bypass_cache,
                    )

                    # Parse response CSV tables
                    x_tilde_explore = parse_candidate_csv(
                        resp_explore,
                        expected_rows=self.pop_size,
                        expected_cols=self.problem.dim,
                        bounds=self.problem.bounds,
                        fallback_generator=explore_pop.x,
                    )
                    x_tilde_exploit = parse_candidate_csv(
                        resp_exploit,
                        expected_rows=self.pop_size,
                        expected_cols=self.problem.dim,
                        bounds=self.problem.bounds,
                        fallback_generator=exploit_pop.x,
                    )

                else:
                    # Routine non-LLM candidate generation (Classical Perturbation Operator)
                    # Exploit: Gaussian perturbation around best candidate
                    lower_b, upper_b = self.problem.bounds
                    span = upper_b - lower_b
                    sigma = self.perturb_sigma_scale * span
                    noise = self.rng.normal(
                        0.0, sigma, size=(self.pop_size, self.problem.dim)
                    )
                    x_tilde_exploit = np.clip(
                        exploit_pop.best_x + noise, lower_b, upper_b
                    )

                    # Explore: Uniform random scattering over bounds
                    x_tilde_explore = self.problem.sample_uniform(
                        self.pop_size, rng=self.rng
                    )

                # 3. Evaluate objective functions
                y_tilde_explore = self.problem.evaluate(x_tilde_explore)
                y_tilde_exploit = self.problem.evaluate(x_tilde_exploit)
                total_evals += len(x_tilde_explore) + len(x_tilde_exploit)

                # 4. Update populations (concat to 2*N_pop)
                new_explore = Population(x=x_tilde_explore, y=y_tilde_explore)
                new_exploit = Population(x=x_tilde_exploit, y=y_tilde_exploit)

                explore_pop = explore_pop.concat(new_explore)
                exploit_pop = exploit_pop.concat(new_exploit)

                # 5. Port and Filter guardrail
                explore_pop, exploit_pop = port_and_filter(
                    explore_pop=explore_pop,
                    exploit_pop=exploit_pop,
                    num_port=self.sol_to_port,
                )

                # 6. Truncate both pools back to pop_size
                explore_pop = explore_pop.truncate(self.pop_size)
                exploit_pop = exploit_pop.truncate(self.pop_size)

                # 7. Record current metrics and update stagnation detector
                current_exploit_best = exploit_pop.best_y
                current_explore_best = explore_pop.best_y
                current_overall_best = min(current_exploit_best, current_explore_best)
                overall_best = min(overall_best, current_overall_best)

                detector.record(exploit_pop.best_y)

                hist_exploit_best.append(current_exploit_best)
                hist_exploit_mean.append(float(np.mean(exploit_pop.y)))
                hist_explore_best.append(current_explore_best)
                hist_explore_mean.append(float(np.mean(explore_pop.y)))
                hist_overall_best.append(overall_best)
                hist_explore_x.append(explore_pop.x.copy())
                hist_exploit_x.append(exploit_pop.x.copy())

                progress.update(task_id, advance=1, best_loss=overall_best)

        best_sol = exploit_pop.best_x
        best_val = exploit_pop.best_y
        if explore_pop.best_y < best_val:
            best_sol = explore_pop.best_x
            best_val = explore_pop.best_y

        return AdaptiveLEOResult(
            strategy=self.strategy.value,
            best_x=best_sol,
            best_y=best_val,
            history_exploit_best_y=hist_exploit_best,
            history_exploit_mean_y=hist_exploit_mean,
            history_explore_best_y=hist_explore_best,
            history_explore_mean_y=hist_explore_mean,
            history_overall_best_y=hist_overall_best,
            history_explore_x=hist_explore_x,
            history_exploit_x=hist_exploit_x,
            llm_call_iterations=llm_call_iterations,
            trigger_reasons=trigger_reasons,
            total_llm_queries=total_llm_queries,
            total_evaluations=total_evals,
            iterations=self.max_iters,
        )
