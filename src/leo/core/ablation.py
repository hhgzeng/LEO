"""Ablation study module for LEO.

Systematically evaluates one-factor-at-a-time (OFAT) ablations:
1. Full-LEO: Algorithm 1 baseline.
2. w/o History: Removes history and loss feedback from prompts.
3. w/o Elitism: Disables PortFilter cross-pool elitism migration.
4. Random-Perturb: Replaces LLM candidate generation with classical random perturbation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

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
from leo.llm.prompts import (
    build_exploit_prompt,
    build_explore_prompt,
    build_rnd_exploit_prompt,
    build_rnd_explore_prompt,
)
from leo.problems.base import BaseProblem


class AblationMode(str, Enum):
    """Ablation configuration mode."""

    FULL = "full"
    NO_HISTORY = "no_history"
    NO_ELITISM = "no_elitism"
    RANDOM_PERTURB = "random_perturb"


@dataclass
class AblationResult:
    """Result container for ablation optimization runs."""

    mode: str
    best_x: np.ndarray
    best_y: float
    history_exploit_best_y: list[float] = field(default_factory=list)
    history_exploit_mean_y: list[float] = field(default_factory=list)
    history_explore_best_y: list[float] = field(default_factory=list)
    history_explore_mean_y: list[float] = field(default_factory=list)
    history_overall_best_y: list[float] = field(default_factory=list)
    history_explore_x: list[np.ndarray] = field(default_factory=list)
    history_exploit_x: list[np.ndarray] = field(default_factory=list)
    eval_checkpoints: list[int] = field(default_factory=list)
    total_evaluations: int = 0
    iterations: int = 0


class AblationLEOAlgorithm:
    """Configurable LEO algorithm supporting one-factor-at-a-time ablation modes."""

    def __init__(
        self,
        problem: BaseProblem,
        mode: AblationMode | str = AblationMode.FULL,
        pop_size: int = 10,
        max_iters: int = 30,
        sol_to_port: int = 1,
        jitter_scale: float = 1e-4,
        llm_client: LLMClient | None = None,
        seed: int | None = None,
        temperature: float = 0.7,
        perturb_sigma_scale: float = 0.05,
        verbose: bool = False,
    ) -> None:
        self.problem = problem
        self.mode = AblationMode(mode) if isinstance(mode, str) else mode
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.sol_to_port = sol_to_port
        self.jitter_scale = jitter_scale
        self.llm_client = llm_client or LLMClient()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.temperature = temperature
        self.perturb_sigma_scale = perturb_sigma_scale
        self.verbose = verbose

    def optimize(self) -> AblationResult:
        """Run ablation optimization loop."""
        # 1. Initialization: Sample initial population
        x_init = self.problem.sample_uniform(self.pop_size, rng=self.rng)
        y_init = self.problem.evaluate(x_init)
        total_evals = len(x_init)

        explore_pop = Population(x=x_init.copy(), y=y_init.copy()).sorted()
        exploit_pop = Population(x=x_init.copy(), y=y_init.copy()).sorted()

        overall_best = float(min(exploit_pop.best_y, explore_pop.best_y))
        hist_exploit_best = [exploit_pop.best_y]
        hist_exploit_mean = [float(np.mean(exploit_pop.y))]
        hist_explore_best = [explore_pop.best_y]
        hist_explore_mean = [float(np.mean(explore_pop.y))]
        hist_overall_best = [overall_best]
        hist_explore_x = [explore_pop.x.copy()]
        hist_exploit_x = [exploit_pop.x.copy()]
        eval_checkpoints = [total_evals]

        progress = Progress(
            SpinnerColumn(),
            TextColumn(
                f"[bold cyan]LEO-Ablation[{self.mode.value} | {self.problem.name}][/bold cyan]"
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
                if self.mode == AblationMode.RANDOM_PERTURB:
                    # Exploit: Gaussian perturbation around best candidate in exploit pool
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

                elif self.mode == AblationMode.NO_HISTORY:
                    # Prompts without candidate coordinates and loss feedback
                    ex_explore = self.problem.sample_uniform(3, rng=self.rng)
                    ex_exploit = self.problem.sample_uniform(3, rng=self.rng)

                    prompt_explore = build_rnd_explore_prompt(
                        num_new=self.pop_size,
                        n_vars=self.problem.dim,
                        bounds=self.problem.bounds,
                        example_x=ex_explore,
                    )
                    prompt_exploit = build_rnd_exploit_prompt(
                        num_new=self.pop_size,
                        n_vars=self.problem.dim,
                        bounds=self.problem.bounds,
                        example_x=ex_exploit,
                    )

                    cache_extras = [
                        f"ablation_no_hist_explore_s{self.seed}_i{iteration}",
                        f"ablation_no_hist_exploit_s{self.seed}_i{iteration}",
                    ]
                    resp_explore, resp_exploit = self.llm_client.query_parallel(
                        [prompt_explore, prompt_exploit],
                        temperature=self.temperature,
                        cache_key_extras=cache_extras,
                    )

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
                    # FULL and NO_ELITISM: LLM with full context & jitter
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
                        f"ablation_{self.mode.value}_explore_s{self.seed}_i{iteration}",
                        f"ablation_{self.mode.value}_exploit_s{self.seed}_i{iteration}",
                    ]
                    resp_explore, resp_exploit = self.llm_client.query_parallel(
                        [prompt_explore, prompt_exploit],
                        temperature=self.temperature,
                        cache_key_extras=cache_extras,
                    )

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

                # Objective evaluation
                y_tilde_explore = self.problem.evaluate(x_tilde_explore)
                y_tilde_exploit = self.problem.evaluate(x_tilde_exploit)
                total_evals += len(x_tilde_explore) + len(x_tilde_exploit)
                eval_checkpoints.append(total_evals)

                # Update population pools
                new_explore = Population(x=x_tilde_explore, y=y_tilde_explore)
                new_exploit = Population(x=x_tilde_exploit, y=y_tilde_exploit)
                explore_pop = explore_pop.concat(new_explore)
                exploit_pop = exploit_pop.concat(new_exploit)

                # PortFilter Elitism Guardrails (disabled in NO_ELITISM)
                if self.mode != AblationMode.NO_ELITISM:
                    explore_pop, exploit_pop = port_and_filter(
                        explore_pop=explore_pop,
                        exploit_pop=exploit_pop,
                        num_port=self.sol_to_port,
                    )

                # Truncate both pools back to pop_size
                explore_pop = explore_pop.truncate(self.pop_size)
                exploit_pop = exploit_pop.truncate(self.pop_size)

                # Record metrics
                current_overall_best = float(
                    min(exploit_pop.best_y, explore_pop.best_y)
                )
                hist_overall_best.append(current_overall_best)
                hist_exploit_best.append(exploit_pop.best_y)
                hist_exploit_mean.append(float(np.mean(exploit_pop.y)))
                hist_explore_best.append(explore_pop.best_y)
                hist_explore_mean.append(float(np.mean(explore_pop.y)))
                hist_explore_x.append(explore_pop.x.copy())
                hist_exploit_x.append(exploit_pop.x.copy())

                progress.update(task_id, advance=1, best_loss=current_overall_best)

        # Select best solution overall
        if exploit_pop.best_y <= explore_pop.best_y:
            best_x = exploit_pop.best_x
            best_y = exploit_pop.best_y
        else:
            best_x = explore_pop.best_x
            best_y = explore_pop.best_y

        return AblationResult(
            mode=self.mode.value,
            best_x=best_x,
            best_y=best_y,
            history_exploit_best_y=hist_exploit_best,
            history_exploit_mean_y=hist_exploit_mean,
            history_explore_best_y=hist_explore_best,
            history_explore_mean_y=hist_explore_mean,
            history_overall_best_y=hist_overall_best,
            history_explore_x=hist_explore_x,
            history_exploit_x=hist_exploit_x,
            eval_checkpoints=eval_checkpoints,
            total_evaluations=total_evals,
            iterations=self.max_iters,
        )
