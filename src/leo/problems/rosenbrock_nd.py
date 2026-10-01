"""Shifted N-dimensional Rosenbrock benchmark function from Section 3.3."""

from __future__ import annotations

import numpy as np

from leo.problems.base import BaseProblem


class ShiftedRosenbrockND(BaseProblem):
    """Shifted N-dimensional Rosenbrock function (Eq. 2 in paper).

    f(x) = sum_{i=0}^{n-2} [ 100 * ((x_{i+1} - a) - (x_i - a)^2)^2 + (1 - (x_i - a))^2 ]
    where a = 0.2913.
    Global minimum: f(1+a, ..., 1+a) = 0.
    """

    def __init__(
        self,
        dim: int = 10,
        a: float = 0.2913,
        bound_range: tuple[float, float] = (-2.0, 2.0),
    ) -> None:
        self.a = a
        lower = np.full(dim, bound_range[0], dtype=np.float64)
        upper = np.full(dim, bound_range[1], dtype=np.float64)
        optimum = np.full(dim, 1.0 + a, dtype=np.float64)

        super().__init__(
            name=f"ShiftedRosenbrock_{dim}D",
            dim=dim,
            bounds=(lower, upper),
            global_min=0.0,
            optimum_x=optimum,
        )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.atleast_2d(x)
        # Shift variables: z_i = x_i - a
        z = x_arr - self.a
        z_curr = z[:, :-1]
        z_next = z[:, 1:]

        term1 = 100.0 * (z_next - z_curr**2) ** 2
        term2 = (1.0 - z_curr) ** 2
        val = np.sum(term1 + term2, axis=1)

        return val if x.ndim > 1 else val[0]
