"""LEO-Rnd ablation baseline from Section 4.1.

Implements LEO-Rnd where the LLM acts solely as a random solution generator
without historical context or elitism guardrails (Table 5).
"""

from __future__ import annotations

import numpy as np

from leo.core.algorithm import LEOResult
from leo.core.population import Population
from leo.llm.client import LLMClient
from leo.llm.parser import parse_candidate_csv
from leo.llm.prompts import build_rnd_exploit_prompt, build_rnd_explore_prompt
from leo.problems.base import BaseProblem


class LEORndOptimizer:
    """LEO-Rnd ablation optimizer (Section 4.1 & Table 5)."""

    def __init__(
        self,
        problem: BaseProblem,
        pop_size: int = 10,
        max_iters: int = 100,
        llm_client: LLMClient | None = None,
        seed: int | None = None,
        temperature: float = 0.7,
    ) -> None:
        self.problem = problem
        self.pop_size = pop_size
        self.max_iters = max_iters
        self.llm_client = llm_client or LLMClient()
        self.rng = np.random.default_rng(seed)
        self.temperature = temperature

    def optimize(self) -> LEOResult:
        # Sample initial population
        x_init = self.problem.sample_uniform(self.pop_size, rng=self.rng)
        y_init = self.problem.evaluate(x_init)
        total_evals = len(x_init)

        explore_pop = Population(x=x_init.copy(), y=y_init.copy()).sorted()
        exploit_pop = Population(x=x_init.copy(), y=y_init.copy()).sorted()

        hist_exploit_best = [exploit_pop.best_y]
        hist_exploit_mean = [float(np.mean(exploit_pop.y))]
        hist_explore_best = [explore_pop.best_y]
        hist_explore_mean = [float(np.mean(explore_pop.y))]
        hist_explore_x = [explore_pop.x.copy()]
        hist_exploit_x = [exploit_pop.x.copy()]

        for iteration in range(1, self.max_iters + 1):
            # Prompts without historical candidate context or elitism
            prompt_explore = build_rnd_explore_prompt(
                num_new=self.pop_size,
                n_vars=self.problem.dim,
                bounds=self.problem.bounds,
            )
            prompt_exploit = build_rnd_exploit_prompt(
                num_new=self.pop_size,
                n_vars=self.problem.dim,
                bounds=self.problem.bounds,
            )

            resp_explore, resp_exploit = self.llm_client.query_parallel(
                [prompt_explore, prompt_exploit], temperature=self.temperature
            )

            x_tilde_explore = parse_candidate_csv(
                resp_explore,
                expected_rows=self.pop_size,
                expected_cols=self.problem.dim,
                bounds=self.problem.bounds,
            )
            x_tilde_exploit = parse_candidate_csv(
                resp_exploit,
                expected_rows=self.pop_size,
                expected_cols=self.problem.dim,
                bounds=self.problem.bounds,
            )

            y_tilde_explore = self.problem.evaluate(x_tilde_explore)
            y_tilde_exploit = self.problem.evaluate(x_tilde_exploit)
            total_evals += len(x_tilde_explore) + len(x_tilde_exploit)

            # In LEO-Rnd: solutions are updated WITHOUT elitism PortFilter guardrail
            explore_pop = Population(x=x_tilde_explore, y=y_tilde_explore).sorted()
            # Exploit pool retains best found so far without importing from explore pool
            combined_exploit = exploit_pop.concat(
                Population(x=x_tilde_exploit, y=y_tilde_exploit)
            )
            exploit_pop = combined_exploit.truncate(self.pop_size)

            hist_exploit_best.append(exploit_pop.best_y)
            hist_exploit_mean.append(float(np.mean(exploit_pop.y)))
            hist_explore_best.append(explore_pop.best_y)
            hist_explore_mean.append(float(np.mean(explore_pop.y)))
            hist_explore_x.append(explore_pop.x.copy())
            hist_exploit_x.append(exploit_pop.x.copy())

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
