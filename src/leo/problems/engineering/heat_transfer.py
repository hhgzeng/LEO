"""1D Steady-State Heat Transfer Optimization (Section 3.4.2 & Appendix A.4).

Formulates the discretized heat conduction equation as an optimization problem:
d^2 T / dx^2 + q/k = 0, with T(0)=0, T(L)=1, q/k=10.
"""

from __future__ import annotations

import numpy as np

from leo.problems.base import BaseProblem


class HeatTransferProblem(BaseProblem):
    """1D Steady-State Heat Conduction residual minimization.

    Variables: Internal temperature values T = [T_1, T_2, ..., T_n].
    """

    def __init__(
        self, dim: int = 4, q_over_k: float = 10.0, length: float = 1.0
    ) -> None:
        self.q_over_k = q_over_k
        self.length = length
        self.dx = length / (dim + 1)
        self.grid_x = np.linspace(0.0, length, dim + 2)

        # Internal grid points
        self.internal_x = self.grid_x[1:-1]

        # Bounds for non-dimensional temperatures: [0.0, 3.0]
        lower = np.zeros(dim, dtype=np.float64)
        upper = np.full(dim, 3.0, dtype=np.float64)

        # Compute exact discrete solution
        exact_internal = self.exact_solution(self.internal_x)

        super().__init__(
            name=f"HeatTransfer_{dim}D",
            dim=dim,
            bounds=(lower, upper),
            global_min=0.0,
            optimum_x=exact_internal,
        )

    def exact_solution(self, x: np.ndarray) -> np.ndarray:
        """Exact analytical temperature profile: T(x) = -(q/2k)x^2 + (1/L + qL/2k)x."""
        c1 = 1.0 / self.length + 0.5 * self.q_over_k * self.length
        return -0.5 * self.q_over_k * (x**2) + c1 * x

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.atleast_2d(x)
        n_samples = len(x_arr)

        # Pad with boundary conditions T_0 = 0.0, T_{n+1} = 1.0
        t_full = np.zeros((n_samples, self.dim + 2), dtype=np.float64)
        t_full[:, 1:-1] = x_arr
        t_full[:, -1] = 1.0  # T_R = 1.0

        # Residual: (T_{i-1} - 2*T_i + T_{i+1}) / dx^2 + q/k
        t_prev = t_full[:, :-2]
        t_curr = t_full[:, 1:-1]
        t_next = t_full[:, 2:]

        laplacian = (t_prev - 2.0 * t_curr + t_next) / (self.dx**2)
        residual = laplacian + self.q_over_k
        loss = np.mean(residual**2, axis=1)

        return loss if x.ndim > 1 else loss[0]
