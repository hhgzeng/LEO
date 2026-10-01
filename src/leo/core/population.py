"""Population container and PortFilter elitism guardrails for LEO.

Maintains candidate coordinates and corresponding objective function values,
providing sorting, truncation, and the PortFilter operation described in Algorithm 1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class Population:
    """Container representing a pool of candidate solutions."""

    x: np.ndarray  # Shape: (N, D)
    y: np.ndarray  # Shape: (N,)

    def __post_init__(self) -> None:
        self.x = np.asarray(self.x, dtype=np.float64)
        self.y = np.asarray(self.y, dtype=np.float64)
        if self.x.ndim == 1:
            self.x = self.x.reshape(1, -1)
        if self.y.ndim == 0:
            self.y = self.y.reshape(1)
        if len(self.x) != len(self.y):
            raise ValueError(
                f"Mismatch in population size: x has {len(self.x)} rows, y has {len(self.y)} elements"
            )

    def __len__(self) -> int:
        return len(self.y)

    @property
    def size(self) -> int:
        return len(self.y)

    @property
    def dim(self) -> int:
        return self.x.shape[1] if self.x.ndim > 1 else 0

    @property
    def best_x(self) -> np.ndarray:
        """Return coordinate of the best individual."""
        idx = int(np.argmin(self.y))
        return self.x[idx].copy()

    @property
    def best_y(self) -> float:
        """Return objective value of the best individual."""
        return float(np.min(self.y))

    def sorted(self) -> Population:
        """Return a new Population sorted ascending by objective value y."""
        sort_indices = np.argsort(self.y)
        return Population(x=self.x[sort_indices].copy(), y=self.y[sort_indices].copy())

    def truncate(self, target_size: int) -> Population:
        """Sort and truncate population to retain the top target_size solutions."""
        sorted_pop = self.sorted()
        return Population(
            x=sorted_pop.x[:target_size].copy(),
            y=sorted_pop.y[:target_size].copy(),
        )

    def concat(self, other: Population) -> Population:
        """Concatenate with another population."""
        new_x = np.vstack([self.x, other.x])
        new_y = np.concatenate([self.y, other.y])
        return Population(x=new_x, y=new_y)


def port_and_filter(
    explore_pop: Population,
    exploit_pop: Population,
    num_port: int,
) -> tuple[Population, Population]:
    """Execute Port and Filter operation (Algorithm 1, Procedure PortFilter).

    Copies the top `num_port` candidates from the explore pool and overwrites
    the worst `num_port` candidates of the exploit pool.

    Args:
        explore_pop: Current explore population.
        exploit_pop: Current exploit population.
        num_port: Number of elite candidates to transfer from explore to exploit.

    Returns:
        Tuple of (explore_pop, updated_exploit_pop).
    """
    sorted_explore = explore_pop.sorted()
    sorted_exploit = exploit_pop.sorted()

    num_port = min(num_port, len(sorted_explore), len(sorted_exploit))
    if num_port <= 0:
        return sorted_explore, sorted_exploit

    # Extract top num_port from explore pool
    x_top = sorted_explore.x[:num_port]
    y_top = sorted_explore.y[:num_port]

    # Copy exploit pool and replace the worst num_port
    x_exploit = sorted_exploit.x.copy()
    y_exploit = sorted_exploit.y.copy()

    x_exploit[-num_port:] = x_top
    y_exploit[-num_port:] = y_top

    return sorted_explore, Population(x=x_exploit, y=y_exploit)
