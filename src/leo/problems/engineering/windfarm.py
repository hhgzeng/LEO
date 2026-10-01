"""Wind Farm Layout Optimization (Section 3.4.3 & Appendix A.5).

Maximizes Annual Energy Production (AEP) by optimizing coordinates of
2, 4, or 8 wind turbines (D=126m) using the Jensen kinematic wake model.
"""

from __future__ import annotations

import numpy as np

from leo.problems.base import BaseProblem


class WindFarmProblem(BaseProblem):
    """Wind Farm Layout Optimization.

    Domain: 1000m x 1000m.
    Rotor diameter D = 126m.
    Number of turbines: n_turbines in {2, 4, 8}.
    Dimension: 2 * n_turbines (x_1, y_1, ..., x_N, y_N).
    """

    def __init__(
        self,
        n_turbines: int = 4,
        domain_size: float = 1000.0,
        rotor_diameter: float = 126.0,
        ct: float = 0.8,
        wake_decay_k: float = 0.075,
    ) -> None:
        self.n_turbines = n_turbines
        self.domain_size = domain_size
        self.d = rotor_diameter
        self.r0 = rotor_diameter / 2.0
        self.ct = ct
        self.wake_k = wake_decay_k
        self.min_dist = 2.0 * rotor_diameter  # Minimal proximity distance 2D

        dim = 2 * n_turbines
        lower = np.zeros(dim, dtype=np.float64)
        upper = np.full(dim, domain_size, dtype=np.float64)

        # 72 wind directions (every 5 degrees)
        self.wind_directions = np.radians(np.linspace(0.0, 355.0, 72))
        self.v0 = 8.0  # mean wind speed 8 m/s

        super().__init__(
            name=f"WindFarm_{n_turbines}Turbines",
            dim=dim,
            bounds=(lower, upper),
            global_min=0.0,
        )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.atleast_2d(x)
        losses = []

        for row in x_arr:
            coords = row.reshape(self.n_turbines, 2)
            loss = self._evaluate_single_layout(coords)
            losses.append(loss)

        res = np.array(losses, dtype=np.float64)
        return res if x.ndim > 1 else res[0]

    def _evaluate_single_layout(self, coords: np.ndarray) -> float:
        # 1. Proximity penalty: distance between any pair should be >= 2*D
        penalty = 0.0
        for i in range(self.n_turbines):
            for j in range(i + 1, self.n_turbines):
                dist = np.linalg.norm(coords[i] - coords[j])
                if dist < self.min_dist:
                    penalty += 50.0 * ((self.min_dist - dist) / self.min_dist) ** 2

        # 2. Wake deficit calculation over 72 wind directions
        total_relative_power = 0.0

        for theta in self.wind_directions:
            cos_t, sin_t = np.cos(theta), np.sin(theta)
            # Downwind (x_rot) and crosswind (y_rot)
            x_rot = coords[:, 0] * cos_t + coords[:, 1] * sin_t
            y_rot = -coords[:, 0] * sin_t + coords[:, 1] * cos_t

            # Order by downwind coordinate
            order = np.argsort(x_rot)
            v_turbines = np.full(self.n_turbines, self.v0)

            for idx_rank, j in enumerate(order):
                deficits_sq = 0.0
                for i in order[:idx_rank]:
                    dx = x_rot[j] - x_rot[i]
                    dy = abs(y_rot[j] - y_rot[i])
                    r_wake = self.r0 + self.wake_k * dx
                    if dy < r_wake:
                        # Jensen velocity deficit
                        denom = (1.0 + 2.0 * self.wake_k * dx / self.d) ** 2
                        deficit = (1.0 - np.sqrt(max(0.0, 1.0 - self.ct))) / denom
                        deficits_sq += deficit**2
                v_turbines[j] = self.v0 * (1.0 - np.sqrt(deficits_sq))

            # Power proportional to v^3
            power_dir = np.sum(np.maximum(0.0, v_turbines) ** 3)
            total_relative_power += power_dir

        ideal_power = len(self.wind_directions) * self.n_turbines * (self.v0**3)
        power_efficiency = total_relative_power / ideal_power

        # Objective is to minimize: (1.0 - efficiency) + penalty
        return float(1.0 - power_efficiency + penalty)
