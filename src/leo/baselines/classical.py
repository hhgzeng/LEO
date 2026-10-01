"""Classical gradient-based and gradient-free baseline optimizers from Table 3."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from cmaes import CMA
from scipy.optimize import minimize

from leo.problems.base import BaseProblem


@dataclass
class BaselineResult:
    """Result container for baseline optimizers."""

    best_x: np.ndarray
    best_y: float
    history_best_y: list[float]
    evaluations: int


def estimate_gradient(
    problem: BaseProblem,
    x: np.ndarray,
    eps: float = 1e-5,
) -> tuple[np.ndarray, int]:
    """Central difference numerical gradient estimation."""
    dim = len(x)
    grad = np.zeros(dim)
    for i in range(dim):
        x_plus = x.copy()
        x_minus = x.copy()
        x_plus[i] += eps
        x_minus[i] -= eps
        grad[i] = (problem.evaluate(x_plus) - problem.evaluate(x_minus)) / (2.0 * eps)
    return grad, 2 * dim


class SGDOptimizer:
    """SGD optimizer with momentum (Table 3)."""

    def __init__(
        self,
        problem: BaseProblem,
        lr: float = 0.01,
        momentum: float = 0.0,
        max_evals: int = 1000,
        seed: int | None = None,
    ) -> None:
        self.problem = problem
        self.lr = lr
        self.momentum = momentum
        self.max_evals = max_evals
        self.rng = np.random.default_rng(seed)

    def optimize(self) -> BaselineResult:
        x = self.problem.sample_uniform(1, rng=self.rng)[0]
        y = float(self.problem.evaluate(x))
        evals = 1

        v = np.zeros(self.problem.dim)
        best_x = x.copy()
        best_y = y
        history = [best_y]

        while evals < self.max_evals:
            grad, g_evals = estimate_gradient(self.problem, x)
            evals += g_evals
            if evals > self.max_evals:
                break

            v = self.momentum * v + self.lr * grad
            x = self.problem.clamp(x - v)
            y = float(self.problem.evaluate(x))
            evals += 1

            if y < best_y:
                best_y = y
                best_x = x.copy()
            history.append(best_y)

        return BaselineResult(
            best_x=best_x, best_y=best_y, history_best_y=history, evaluations=evals
        )


class AdamOptimizer:
    """Adam optimizer (Table 3: beta1 in {0.1, 0.5, 0.9}, beta2=0.999)."""

    def __init__(
        self,
        problem: BaseProblem,
        lr: float = 0.05,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
        max_evals: int = 1000,
        seed: int | None = None,
    ) -> None:
        self.problem = problem
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.max_evals = max_evals
        self.rng = np.random.default_rng(seed)

    def optimize(self) -> BaselineResult:
        x = self.problem.sample_uniform(1, rng=self.rng)[0]
        y = float(self.problem.evaluate(x))
        evals = 1

        m = np.zeros(self.problem.dim)
        v = np.zeros(self.problem.dim)
        t = 0

        best_x = x.copy()
        best_y = y
        history = [best_y]

        while evals < self.max_evals:
            t += 1
            grad, g_evals = estimate_gradient(self.problem, x)
            evals += g_evals
            if evals > self.max_evals:
                break

            m = self.beta1 * m + (1.0 - self.beta1) * grad
            v = self.beta2 * v + (1.0 - self.beta2) * (grad**2)

            m_hat = m / (1.0 - self.beta1**t)
            v_hat = v / (1.0 - self.beta2**t)

            step = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            x = self.problem.clamp(x - step)
            y = float(self.problem.evaluate(x))
            evals += 1

            if y < best_y:
                best_y = y
                best_x = x.copy()
            history.append(best_y)

        return BaselineResult(
            best_x=best_x, best_y=best_y, history_best_y=history, evaluations=evals
        )


class LBFGSOptimizer:
    """L-BFGS-B optimizer from SciPy (Table 3)."""

    def __init__(
        self,
        problem: BaseProblem,
        max_evals: int = 1000,
        seed: int | None = None,
    ) -> None:
        self.problem = problem
        self.max_evals = max_evals
        self.rng = np.random.default_rng(seed)

    def optimize(self) -> BaselineResult:
        x0 = self.problem.sample_uniform(1, rng=self.rng)[0]
        evals = 0
        history: list[float] = []
        best_y = float("inf")
        best_x = x0.copy()

        bounds = list(zip(self.problem.lower_bounds, self.problem.upper_bounds))

        def obj(x_vec: np.ndarray) -> float:
            nonlocal evals, best_y, best_x
            evals += 1
            val = float(self.problem.evaluate(x_vec))
            if val < best_y:
                best_y = val
                best_x = x_vec.copy()
            history.append(best_y)
            return val

        minimize(
            obj,
            x0=x0,
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxfun": self.max_evals, "ftol": 1e-12},
        )

        return BaselineResult(
            best_x=best_x, best_y=best_y, history_best_y=history, evaluations=evals
        )


class SimulatedAnnealingOptimizer:
    """Simulated Annealing optimizer with exponential cooling (Table 3: T in {0.1, 1, 10})."""

    def __init__(
        self,
        problem: BaseProblem,
        initial_temp: float = 1.0,
        cooling_rate: float = 0.995,
        max_evals: int = 1000,
        seed: int | None = None,
    ) -> None:
        self.problem = problem
        self.t0 = initial_temp
        self.cooling_rate = cooling_rate
        self.max_evals = max_evals
        self.rng = np.random.default_rng(seed)

    def optimize(self) -> BaselineResult:
        x = self.problem.sample_uniform(1, rng=self.rng)[0]
        y = float(self.problem.evaluate(x))
        evals = 1

        best_x = x.copy()
        best_y = y
        history = [best_y]

        temp = self.t0
        span = self.problem.upper_bounds - self.problem.lower_bounds

        while evals < self.max_evals:
            step = self.rng.normal(0.0, 0.1 * span)
            x_candidate = self.problem.clamp(x + step)
            y_candidate = float(self.problem.evaluate(x_candidate))
            evals += 1

            delta = y_candidate - y
            if delta < 0 or self.rng.random() < np.exp(-delta / max(temp, 1e-12)):
                x = x_candidate
                y = y_candidate

            if y < best_y:
                best_y = y
                best_x = x.copy()

            history.append(best_y)
            temp = max(temp * self.cooling_rate, 1e-8)

        return BaselineResult(
            best_x=best_x, best_y=best_y, history_best_y=history, evaluations=evals
        )


class CMAESOptimizer:
    """Covariance Matrix Adaptation Evolution Strategy (CMA-ES) (Table 3)."""

    def __init__(
        self,
        problem: BaseProblem,
        max_evals: int = 1000,
        seed: int | None = None,
    ) -> None:
        self.problem = problem
        self.max_evals = max_evals
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def optimize(self) -> BaselineResult:
        mean = self.problem.sample_uniform(1, rng=self.rng)[0]
        span = self.problem.upper_bounds - self.problem.lower_bounds
        sigma = float(np.mean(span) / 4.0)

        bounds = np.column_stack([self.problem.lower_bounds, self.problem.upper_bounds])
        optimizer = CMA(mean=mean, sigma=sigma, bounds=bounds, seed=self.seed)

        best_x = mean.copy()
        best_y = float("inf")
        history: list[float] = []
        evals = 0

        while evals < self.max_evals and not optimizer.should_stop():
            solutions = []
            for _ in range(optimizer.population_size):
                x = optimizer.ask()
                y = float(self.problem.evaluate(x))
                evals += 1
                solutions.append((x, y))

                if y < best_y:
                    best_y = y
                    best_x = x.copy()
                history.append(best_y)

                if evals >= self.max_evals:
                    break

            optimizer.tell(solutions)

        return BaselineResult(
            best_x=best_x, best_y=best_y, history_best_y=history, evaluations=evals
        )


class COBYLAOptimizer:
    """COBYLA derivative-free optimizer from SciPy (Table 3)."""

    def __init__(
        self,
        problem: BaseProblem,
        max_evals: int = 1000,
        seed: int | None = None,
    ) -> None:
        self.problem = problem
        self.max_evals = max_evals
        self.rng = np.random.default_rng(seed)

    def optimize(self) -> BaselineResult:
        x0 = self.problem.sample_uniform(1, rng=self.rng)[0]
        evals = 0
        best_y = float("inf")
        best_x = x0.copy()
        history: list[float] = []

        def obj(x_vec: np.ndarray) -> float:
            nonlocal evals, best_y, best_x
            evals += 1
            x_clamped = self.problem.clamp(x_vec)
            val = float(self.problem.evaluate(x_clamped))
            if val < best_y:
                best_y = val
                best_x = x_clamped.copy()
            history.append(best_y)
            return val

        constraints = []
        for i in range(self.problem.dim):
            low = self.problem.lower_bounds[i]
            high = self.problem.upper_bounds[i]
            constraints.append(
                {"type": "ineq", "fun": lambda x, i=i, low=low: x[i] - low}
            )
            constraints.append(
                {"type": "ineq", "fun": lambda x, i=i, high=high: high - x[i]}
            )

        minimize(
            obj,
            x0=x0,
            method="COBYLA",
            constraints=constraints,
            options={"maxiter": self.max_evals, "rhobeg": 0.5},
        )

        return BaselineResult(
            best_x=best_x, best_y=best_y, history_best_y=history, evaluations=evals
        )
