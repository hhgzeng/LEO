"""LEO-modular NSGA-II and standard NSGA-II multi-objective optimizers.

Implements the LEO-modular algorithm from Section 3.2 (Fig. 5), replacing crossover
and mutation with LLM in-context queries (Table 9).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from leo.llm.client import LLMClient
from leo.llm.parser import parse_modular_list
from leo.llm.prompts import build_modular_exploit_prompt, build_modular_explore_prompt
from leo.problems.multi_objective import MultiObjectiveProblem


@dataclass
class MultiObjectiveResult:
    """Result container for multi-objective optimization."""

    population_x: np.ndarray  # Shape: (N, D)
    population_f: np.ndarray  # Shape: (N, M)
    pareto_x: np.ndarray  # Shape: (P, D)
    pareto_f: np.ndarray  # Shape: (P, M)
    history_pareto_f: list[np.ndarray] = field(default_factory=list)


def fast_non_dominated_sort(objectives: np.ndarray) -> list[list[int]]:
    """Fast non-dominated sorting (Deb et al., 2002)."""
    n = len(objectives)
    domination_counts = np.zeros(n, dtype=int)
    dominated_solutions: list[list[int]] = [[] for _ in range(n)]
    fronts: list[list[int]] = [[]]

    for p in range(n):
        for q in range(n):
            if p == q:
                continue
            # p dominates q if p <= q for all objectives and p < q for at least one
            p_leq_q = np.all(objectives[p] <= objectives[q])
            p_lt_q = np.any(objectives[p] < objectives[q])

            if p_leq_q and p_lt_q:
                dominated_solutions[p].append(q)
            else:
                q_leq_p = np.all(objectives[q] <= objectives[p])
                q_lt_p = np.any(objectives[q] < objectives[p])
                if q_leq_p and q_lt_p:
                    domination_counts[p] += 1

        if domination_counts[p] == 0:
            fronts[0].append(p)

    i = 0
    while len(fronts[i]) > 0:
        next_front: list[int] = []
        for p in fronts[i]:
            for q in dominated_solutions[p]:
                domination_counts[q] -= 1
                if domination_counts[q] == 0:
                    next_front.append(q)
        i += 1
        fronts.append(next_front)

    return fronts[:-1]


def calculate_crowding_distance(objectives: np.ndarray, front: list[int]) -> np.ndarray:
    """Calculate crowding distance for individuals in a single Pareto front."""
    l = len(front)
    if l == 0:
        return np.array([])
    if l <= 2:
        return np.full(l, np.inf)

    distances = np.zeros(l)
    m = objectives.shape[1]

    for i in range(m):
        # Sort front by objective i
        obj_vals = objectives[front, i]
        sorted_indices = np.argsort(obj_vals)
        distances[sorted_indices[0]] = np.inf
        distances[sorted_indices[-1]] = np.inf

        obj_min = obj_vals[sorted_indices[0]]
        obj_max = obj_vals[sorted_indices[-1]]
        denom = max(obj_max - obj_min, 1e-12)

        for k in range(1, l - 1):
            prev_idx = sorted_indices[k - 1]
            next_idx = sorted_indices[k + 1]
            distances[sorted_indices[k]] += (
                obj_vals[next_idx] - obj_vals[prev_idx]
            ) / denom

    return distances


class LEOModularNSGA2:
    """LEO-modular NSGA-II using LLM for candidate solution generation (Fig. 5)."""

    def __init__(
        self,
        problem: MultiObjectiveProblem,
        pop_size: int = 10,
        max_generations: int = 40,
        llm_client: LLMClient | None = None,
        seed: int | None = None,
    ) -> None:
        self.problem = problem
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.llm_client = llm_client or LLMClient()
        self.rng = np.random.default_rng(seed)

    def optimize(self) -> MultiObjectiveResult:
        """Run LEO-modular NSGA-II optimization loop."""
        lower, upper = self.problem.bounds
        pop_x = self.rng.uniform(lower, upper, size=(self.pop_size, self.problem.dim))
        pop_f = self.problem.evaluate(pop_x)

        history_pareto: list[np.ndarray] = []

        for gen in range(self.max_generations):
            # Mating selection: randomly pair parents
            perm = self.rng.permutation(self.pop_size)
            offspring_x_list: list[np.ndarray] = []

            prompts: list[tuple[str, str]] = []  # (prompt_type, prompt_str)
            for i in range(0, self.pop_size - 1, 2):
                p1_idx, p2_idx = perm[i], perm[i + 1]
                p1_x, p1_f = pop_x[p1_idx], pop_f[p1_idx]
                p2_x, p2_f = pop_x[p2_idx], pop_f[p2_idx]

                # Alternate exploit and explore prompts
                if (i // 2) % 2 == 0:
                    pr = build_modular_exploit_prompt(
                        p1_x, p1_f, p2_x, p2_f, bounds=self.problem.bounds
                    )
                    prompts.append(("exploit", pr))
                else:
                    pr = build_modular_explore_prompt(
                        p1_x, p1_f, p2_x, p2_f, bounds=self.problem.bounds
                    )
                    prompts.append(("explore", pr))

            # Query LLM
            prompt_texts = [p[1] for p in prompts]
            responses = self.llm_client.query_parallel(prompt_texts)

            for resp in responses:
                offspring = parse_modular_list(
                    resp,
                    expected_pairs=2,
                    expected_dim=self.problem.dim,
                    bounds=self.problem.bounds,
                )
                offspring_x_list.append(offspring)

            if offspring_x_list:
                offspring_x = np.vstack(offspring_x_list)
            else:
                offspring_x = self.rng.uniform(
                    lower, upper, size=(self.pop_size, self.problem.dim)
                )

            offspring_f = self.problem.evaluate(offspring_x)

            # Combine parent and offspring populations
            combined_x = np.vstack([pop_x, offspring_x])
            combined_f = np.vstack([pop_f, offspring_f])

            # Non-dominated sorting
            fronts = fast_non_dominated_sort(combined_f)

            # Environmental selection
            new_pop_x: list[np.ndarray] = []
            new_pop_f: list[np.ndarray] = []

            for front in fronts:
                if len(new_pop_x) + len(front) <= self.pop_size:
                    new_pop_x.extend(combined_x[front])
                    new_pop_f.extend(combined_f[front])
                else:
                    remaining = self.pop_size - len(new_pop_x)
                    crowding_dist = calculate_crowding_distance(combined_f, front)
                    sorted_by_dist = np.argsort(-crowding_dist)  # Descending
                    selected = [front[idx] for idx in sorted_by_dist[:remaining]]
                    new_pop_x.extend(combined_x[selected])
                    new_pop_f.extend(combined_f[selected])
                    break

            pop_x = np.array(new_pop_x)
            pop_f = np.array(new_pop_f)

            # Record Pareto front of current generation
            first_front = fast_non_dominated_sort(pop_f)[0]
            history_pareto.append(pop_f[first_front].copy())

        final_fronts = fast_non_dominated_sort(pop_f)
        first_front_idx = final_fronts[0]

        return MultiObjectiveResult(
            population_x=pop_x,
            population_f=pop_f,
            pareto_x=pop_x[first_front_idx],
            pareto_f=pop_f[first_front_idx],
            history_pareto_f=history_pareto,
        )


class StandardNSGA2:
    """Traditional NSGA-II algorithm with SBX crossover and polynomial mutation."""

    def __init__(
        self,
        problem: MultiObjectiveProblem,
        pop_size: int = 10,
        max_generations: int = 40,
        crossover_prob: float = 0.9,
        mutation_prob: float = 0.1,
        seed: int | None = None,
    ) -> None:
        self.problem = problem
        self.pop_size = pop_size
        self.max_generations = max_generations
        self.pc = crossover_prob
        self.pm = mutation_prob
        self.rng = np.random.default_rng(seed)

    def optimize(self) -> MultiObjectiveResult:
        lower, upper = self.problem.bounds
        pop_x = self.rng.uniform(lower, upper, size=(self.pop_size, self.problem.dim))
        pop_f = self.problem.evaluate(pop_x)

        for gen in range(self.max_generations):
            # Tournament selection and offspring creation
            offspring_x: list[np.ndarray] = []
            for _ in range(self.pop_size // 2):
                p1 = pop_x[self.rng.integers(self.pop_size)]
                p2 = pop_x[self.rng.integers(self.pop_size)]

                # SBX Crossover
                if self.rng.random() < self.pc:
                    beta = self.rng.uniform(0.0, 1.0, size=self.problem.dim)
                    c1 = 0.5 * ((1.0 + beta) * p1 + (1.0 - beta) * p2)
                    c2 = 0.5 * ((1.0 - beta) * p1 + (1.0 + beta) * p2)
                else:
                    c1, c2 = p1.copy(), p2.copy()

                # Polynomial Mutation
                for child in (c1, c2):
                    for d in range(self.problem.dim):
                        if self.rng.random() < self.pm:
                            delta = self.rng.normal(0, 0.1 * (upper[d] - lower[d]))
                            child[d] += delta
                    offspring_x.append(np.clip(child, lower, upper))

            off_x = np.array(offspring_x)
            off_f = self.problem.evaluate(off_x)

            combined_x = np.vstack([pop_x, off_x])
            combined_f = np.vstack([pop_f, off_f])

            fronts = fast_non_dominated_sort(combined_f)
            new_pop_x: list[np.ndarray] = []
            new_pop_f: list[np.ndarray] = []

            for front in fronts:
                if len(new_pop_x) + len(front) <= self.pop_size:
                    new_pop_x.extend(combined_x[front])
                    new_pop_f.extend(combined_f[front])
                else:
                    remaining = self.pop_size - len(new_pop_x)
                    crowding_dist = calculate_crowding_distance(combined_f, front)
                    sorted_by_dist = np.argsort(-crowding_dist)
                    selected = [front[idx] for idx in sorted_by_dist[:remaining]]
                    new_pop_x.extend(combined_x[selected])
                    new_pop_f.extend(combined_f[selected])
                    break

            pop_x = np.array(new_pop_x)
            pop_f = np.array(new_pop_f)

        first_front = fast_non_dominated_sort(pop_f)[0]
        return MultiObjectiveResult(
            population_x=pop_x,
            population_f=pop_f,
            pareto_x=pop_x[first_front],
            pareto_f=pop_f[first_front],
        )
