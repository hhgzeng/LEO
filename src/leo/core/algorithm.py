"""LEO (Large Language Model-Based Evolutionary Optimizer) execution loop.

Faithfully implements Algorithm 1 from:
'Large language model-based evolutionary optimizer: Reasoning with elitism'.
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
from leo.llm.client import LLMClient
from leo.llm.parser import parse_candidate_csv
from leo.llm.prompts import build_exploit_prompt, build_explore_prompt
from leo.problems.base import BaseProblem


@dataclass
class LEOResult:
    """Optimization result container."""

    best_x: np.ndarray
    best_y: float
    history_exploit_best_y: list[float] = field(default_factory=list)
    history_exploit_mean_y: list[float] = field(default_factory=list)
    history_explore_best_y: list[float] = field(default_factory=list)
    history_explore_mean_y: list[float] = field(default_factory=list)
    history_explore_x: list[np.ndarray] = field(default_factory=list)
    history_exploit_x: list[np.ndarray] = field(default_factory=list)
    total_evaluations: int = 0
    iterations: int = 0


class LEOAlgorithm:
    """Large Language Model-Based Evolutionary Optimizer (Algorithm 1)."""

    def __init__(
        self,
        problem: BaseProblem,
        pop_size: int = 10,
        max_iters: int = 30,
        sol_to_port: int = 1,
        jitter_scale: float = 1e-4,
        llm_client: LLMClient | None = None,
        seed: int | None = None,
        temperature: float = 0.7,
        verbose: bool = False,
    ) -> None:
        self.problem = problem
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.sol_to_port = sol_to_port
        self.jitter_scale = jitter_scale
        self.llm_client = llm_client or LLMClient()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.temperature = temperature
        self.verbose = verbose

    def optimize(self) -> LEOResult:
        """Run Algorithm 1 optimization loop."""
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

        # History tracking
        hist_exploit_best = [exploit_pop.best_y]
        hist_exploit_mean = [float(np.mean(exploit_pop.y))]
        hist_explore_best = [explore_pop.best_y]
        hist_explore_mean = [float(np.mean(explore_pop.y))]
        hist_explore_x = [explore_pop.x.copy()]
        hist_exploit_x = [exploit_pop.x.copy()]

        progress = Progress(
            SpinnerColumn(),
            TextColumn(f"[bold cyan]LEO [{self.problem.name}][/bold cyan]"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            TextColumn("Best: {task.fields[best_loss]:.6f}"),
            disable=not self.verbose,
        )

        with progress:
            task_id = progress.add_task(
                "optimizing", total=self.max_iters, best_loss=exploit_pop.best_y
            )

            for iteration in range(1, self.max_iters + 1):
                # 2. Jitter injection on candidate coordinates
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

                # 3. Construct explore and exploit prompts
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

                # 4. Asynchronously query LLM in parallel
                resp_explore, resp_exploit = self.llm_client.query_parallel(
                    [prompt_explore, prompt_exploit], temperature=self.temperature
                )

                # 5. Parse response CSV tables
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

                # 6. Evaluate objective functions
                y_tilde_explore = self.problem.evaluate(x_tilde_explore)
                y_tilde_exploit = self.problem.evaluate(x_tilde_exploit)
                total_evals += len(x_tilde_explore) + len(x_tilde_exploit)

                # 7. Update populations (concat to 2*N_pop)
                new_explore = Population(x=x_tilde_explore, y=y_tilde_explore)
                new_exploit = Population(x=x_tilde_exploit, y=y_tilde_exploit)

                explore_pop = explore_pop.concat(new_explore)
                exploit_pop = exploit_pop.concat(new_exploit)

                # 8. Port and Filter guardrail
                explore_pop, exploit_pop = port_and_filter(
                    explore_pop=explore_pop,
                    exploit_pop=exploit_pop,
                    num_port=self.sol_to_port,
                )

                # 9. Truncate both pools back to pop_size
                explore_pop = explore_pop.truncate(self.pop_size)
                exploit_pop = exploit_pop.truncate(self.pop_size)

                # Record metrics
                current_best_exploit = exploit_pop.best_y
                hist_exploit_best.append(current_best_exploit)
                hist_exploit_mean.append(float(np.mean(exploit_pop.y)))
                hist_explore_best.append(explore_pop.best_y)
                hist_explore_mean.append(float(np.mean(explore_pop.y)))
                hist_explore_x.append(explore_pop.x.copy())
                hist_exploit_x.append(exploit_pop.x.copy())

                progress.update(task_id, advance=1, best_loss=current_best_exploit)

        return LEOResult(
            best_x=exploit_pop.best_x,
            best_y=exploit_pop.best_y,
            history_exploit_best_y=hist_exploit_best,
            history_exploit_mean_y=hist_exploit_mean,
            history_explore_best_y=hist_explore_best,
            history_explore_mean_y=hist_explore_mean,
            history_explore_x=hist_explore_x,
            history_exploit_x=hist_exploit_x,
            total_evaluations=total_evals,
            iterations=self.max_iters,
        )
