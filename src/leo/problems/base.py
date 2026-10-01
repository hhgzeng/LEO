"""Base class for optimization problems."""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class BaseProblem(ABC):
    """Abstract base class for numerical optimization problems."""

    def __init__(
        self,
        name: str,
        dim: int,
        bounds: tuple[np.ndarray, np.ndarray],
        global_min: float | None = None,
        optimum_x: np.ndarray | None = None,
    ) -> None:
        self.name = name
        self.dim = dim
        self.lower_bounds = np.asarray(bounds[0], dtype=np.float64)
        self.upper_bounds = np.asarray(bounds[1], dtype=np.float64)
        self.bounds = (self.lower_bounds, self.upper_bounds)
        self.global_min = global_min
        self.optimum_x = (
            np.asarray(optimum_x, dtype=np.float64) if optimum_x is not None else None
        )

        if len(self.lower_bounds) != dim or len(self.upper_bounds) != dim:
            raise ValueError(
                f"Bounds dimension mismatch: expected {dim}, got {len(self.lower_bounds)}"
            )

    @abstractmethod
    def evaluate(self, x: np.ndarray) -> np.ndarray:
        """Evaluate objective function on candidate points.

        Args:
            x: Array of shape (N, D) or (D,).

        Returns:
            1D array of shape (N,) containing objective values.
        """

    def sample_uniform(
        self, pop_size: int, rng: np.random.Generator | None = None
    ) -> np.ndarray:
        """Sample candidate points uniformly within bounds."""
        if rng is None:
            rng = np.random.default_rng()
        return rng.uniform(
            self.lower_bounds, self.upper_bounds, size=(pop_size, self.dim)
        )

    def clamp(self, x: np.ndarray) -> np.ndarray:
        """Clamp coordinates within domain bounds."""
        return np.clip(x, self.lower_bounds, self.upper_bounds)
