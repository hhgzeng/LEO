"""Multi-objective benchmark functions ZDT1 and ZDT3 from Section 3.2."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class MultiObjectiveProblem(ABC):
    """Abstract base class for multi-objective optimization problems."""

    def __init__(
        self,
        name: str,
        dim: int,
        num_objectives: int,
        bounds: tuple[np.ndarray, np.ndarray],
    ) -> None:
        self.name = name
        self.dim = dim
        self.num_objectives = num_objectives
        self.lower_bounds = np.asarray(bounds[0], dtype=np.float64)
        self.upper_bounds = np.asarray(bounds[1], dtype=np.float64)
        self.bounds = (self.lower_bounds, self.upper_bounds)

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate objectives for candidate points.

        Returns array of shape (N, num_objectives) or (num_objectives,).
        """

    @abstractmethod
    def get_pareto_front(self, n_points: int = 200) -> np.ndarray:
        """Generate reference points along the true Pareto front."""


class ZDT1Problem(MultiObjectiveProblem):
    """ZDT1 benchmark function (Table 4).

    f1(x) = x1
    g(x) = 1 + 9/(n-1) * sum_{i=2}^n x_i
    f2(x) = g * (1 - sqrt(f1 / g))
    """

    def __init__(self, dim: int = 2) -> None:
        super().__init__(
            name="ZDT1",
            dim=dim,
            num_objectives=2,
            bounds=(np.zeros(dim), np.ones(dim)),
        )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.atleast_2d(x)
        f1 = x_arr[:, 0]
        if self.dim > 1:
            g = 1.0 + (9.0 / (self.dim - 1)) * np.sum(x_arr[:, 1:], axis=1)
        else:
            g = np.ones(len(x_arr))

        # Safe sqrt
        ratio = np.clip(f1 / np.maximum(g, 1e-12), 0.0, 1.0)
        f2 = g * (1.0 - np.sqrt(ratio))

        objs = np.column_stack([f1, f2])
        return objs if x.ndim > 1 else objs[0]

    def get_pareto_front(self, n_points: int = 200) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        f2 = 1.0 - np.sqrt(f1)
        return np.column_stack([f1, f2])


class ZDT3Problem(MultiObjectiveProblem):
    """ZDT3 benchmark function with disconnected Pareto front segments (Table 4).

    f1(x) = x1
    g(x) = 1 + 9/(n-1) * sum_{i=2}^n x_i
    f2(x) = g * (1 - sqrt(f1 / g) - (f1 / g) * sin(10*pi*f1))
    """

    def __init__(self, dim: int = 2) -> None:
        super().__init__(
            name="ZDT3",
            dim=dim,
            num_objectives=2,
            bounds=(np.zeros(dim), np.ones(dim)),
        )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.atleast_2d(x)
        f1 = x_arr[:, 0]
        if self.dim > 1:
            g = 1.0 + (9.0 / (self.dim - 1)) * np.sum(x_arr[:, 1:], axis=1)
        else:
            g = np.ones(len(x_arr))

        ratio = np.clip(f1 / np.maximum(g, 1e-12), 0.0, 1.0)
        f2 = g * (1.0 - np.sqrt(ratio) - ratio * np.sin(10.0 * np.pi * f1))

        objs = np.column_stack([f1, f2])
        return objs if x.ndim > 1 else objs[0]

    def get_pareto_front(self, n_points: int = 500) -> np.ndarray:
        f1 = np.linspace(0.0, 1.0, n_points)
        f2 = 1.0 - np.sqrt(f1) - f1 * np.sin(10.0 * np.pi * f1)
        front = np.column_stack([f1, f2])

        # Filter non-dominated points
        is_dominated = np.zeros(n_points, dtype=bool)
        for i in range(n_points):
            for j in range(n_points):
                if (
                    i != j
                    and (front[j, 0] <= front[i, 0] and front[j, 1] <= front[i, 1])
                    and (front[j, 0] < front[i, 0] or front[j, 1] < front[i, 1])
                ):
                    is_dominated[i] = True
                    break
        return front[~is_dominated]
