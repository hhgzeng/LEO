"""Nose Cone Shape Optimization toy problem (Section 2.1 & Appendix A.2).

Optimizes power-law index n_pl in y = a * x^(n_pl) to minimize Newtonian hypersonic drag.
"""

from __future__ import annotations

import numpy as np
from scipy import integrate

from leo.problems.base import BaseProblem


class NoseConeProblem(BaseProblem):
    """Nose Cone Shape Optimization using Newtonian impact theory.

    Variable: n_pl in [0.001, 1.0].
    """

    def __init__(self, fineness_ratio: float = 1.0) -> None:
        self.length = 1.0
        self.radius = 0.5 / fineness_ratio
        self.a = self.radius / (self.length**1.0)

        super().__init__(
            name="NoseConeOptimization",
            dim=1,
            bounds=(np.array([0.001]), np.array([1.0])),
            global_min=0.0,
            optimum_x=np.array([0.75]),
        )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.atleast_2d(x)
        losses = []

        for row in x_arr:
            n_pl = float(np.clip(row[0], 0.001, 1.0))

            def integrand(xi: float, n_val: float = n_pl) -> float:
                if xi <= 1e-6:
                    return 0.0
                y = self.a * (xi**n_val)
                dy = self.a * n_val * (xi ** (n_val - 1.0))
                return (y * (dy**3)) / (1.0 + dy**2)

            integral, _ = integrate.quad(integrand, 1e-5, self.length, limit=50)
            cd = (4.0 / (self.radius**2)) * integral
            losses.append(cd)

        res = np.array(losses, dtype=np.float64)
        return res if x.ndim > 1 else res[0]
