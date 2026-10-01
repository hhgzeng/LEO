"""Supersonic Nozzle Shape Optimization (Section 3.4.1 & Appendix A.3).

Parameterizes a 2D supersonic nozzle upper wall using a Cubic Bezier curve
with 4 design variables (x1, y1, x2, y2) to minimize exit radial velocity.
"""

from __future__ import annotations

import numpy as np

from leo.problems.base import BaseProblem


class NozzleShapeProblem(BaseProblem):
    """Supersonic Nozzle Shape Optimization.

    Control points:
    P0 = (0.0, r_inlet) [Fixed throat]
    P1 = (x1, y1)       [Variable]
    P2 = (x2, y2)       [Variable]
    P3 = (L, r_exit)    [Fixed exit]

    Variables: [x1, y1, x2, y2]
    """

    def __init__(
        self,
        length: float = 1.0,
        r_inlet: float = 0.5,
        r_exit: float = 1.0,
    ) -> None:
        self.length = length
        self.r_inlet = r_inlet
        self.r_exit = r_exit

        # Bounds: x1 in [0, L], y1 in [r_inlet, r_exit], x2 in [0, L], y2 in [r_inlet, r_exit]
        lower = np.array([0.0, r_inlet, 0.0, r_inlet], dtype=np.float64)
        upper = np.array([length, r_exit, length, r_exit], dtype=np.float64)

        super().__init__(
            name="NozzleShapeOptimization",
            dim=4,
            bounds=(lower, upper),
            global_min=0.0,
        )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.atleast_2d(x)
        losses = []

        for row in x_arr:
            x1, y1, x2, y2 = row

            # Exit slope at t=1: dy/dx = (y3 - y2) / (x3 - x2)
            dx_exit = max(self.length - x2, 1e-5)
            dy_exit = self.r_exit - y2
            slope_exit = dy_exit / dx_exit
            theta_exit = np.arctan(slope_exit)

            # Radial velocity component is proportional to sin(theta_exit)
            v_radial_loss = (np.sin(theta_exit)) ** 2

            # Shape regularity penalty: ensure monotonic longitudinal progression
            penalty = 0.0
            if x1 > x2:
                penalty += 10.0 * (x1 - x2) ** 2
            if y1 < self.r_inlet or y2 < y1:
                penalty += (
                    10.0 * max(0.0, self.r_inlet - y1) ** 2
                    + 10.0 * max(0.0, y1 - y2) ** 2
                )

            losses.append(v_radial_loss + penalty)

        res = np.array(losses, dtype=np.float64)
        return res if x.ndim > 1 else res[0]

    def get_contour(
        self, params: np.ndarray, num_points: int = 100
    ) -> tuple[np.ndarray, np.ndarray]:
        """Compute the (x, y) coordinates of the Bezier nozzle wall."""
        x1, y1, x2, y2 = params
        t = np.linspace(0.0, 1.0, num_points)

        x_coords = (
            (1 - t) ** 3 * 0.0
            + 3 * (1 - t) ** 2 * t * x1
            + 3 * (1 - t) * t**2 * x2
            + t**3 * self.length
        )
        y_coords = (
            (1 - t) ** 3 * self.r_inlet
            + 3 * (1 - t) ** 2 * t * y1
            + 3 * (1 - t) * t**2 * y2
            + t**3 * self.r_exit
        )
        return x_coords, y_coords
