"""2D benchmark optimization functions matching Table 2 in LEO paper."""

from __future__ import annotations

import numpy as np

from leo.problems.base import BaseProblem


class ScaledSphereProblem(BaseProblem):
    """ScaledSphere: f(x, y) = x^2 + y^4, bounds [-1, 4]^2, min f(0, 0) = 0."""

    def __init__(self) -> None:
        super().__init__(
            name="ScaledSphere",
            dim=2,
            bounds=(np.array([-1.0, -1.0]), np.array([4.0, 4.0])),
            global_min=0.0,
            optimum_x=np.array([0.0, 0.0]),
        )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.atleast_2d(x)
        val = x_arr[:, 0] ** 2 + x_arr[:, 1] ** 4
        return val if x.ndim > 1 else val[0]


class HimmelblauProblem(BaseProblem):
    """Himmelblau: f(x, y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2, bounds [-4, 4]^2, min = 0."""

    def __init__(self) -> None:
        super().__init__(
            name="Himmelblau",
            dim=2,
            bounds=(np.array([-4.0, -4.0]), np.array([4.0, 4.0])),
            global_min=0.0,
            optimum_x=np.array([3.0, 2.0]),
        )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.atleast_2d(x)
        val = (x_arr[:, 0] ** 2 + x_arr[:, 1] - 11.0) ** 2 + (
            x_arr[:, 0] + x_arr[:, 1] ** 2 - 7.0
        ) ** 2
        return val if x.ndim > 1 else val[0]


class Rosenbrock2DProblem(BaseProblem):
    """Rosenbrock 2D: f(x, y) = 100(y - x^2)^2 + (1 - x)^2, bounds [0, 2]^2, min f(1, 1) = 0."""

    def __init__(self) -> None:
        super().__init__(
            name="Rosenbrock2D",
            dim=2,
            bounds=(np.array([0.0, 0.0]), np.array([2.0, 2.0])),
            global_min=0.0,
            optimum_x=np.array([1.0, 1.0]),
        )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.atleast_2d(x)
        val = 100.0 * (x_arr[:, 1] - x_arr[:, 0] ** 2) ** 2 + (1.0 - x_arr[:, 0]) ** 2
        return val if x.ndim > 1 else val[0]


class Sphere2DProblem(BaseProblem):
    """Sphere 2D: f(x, y) = x^2 + y^2, bounds [-1, 1]^2, min f(0, 0) = 0."""

    def __init__(self) -> None:
        super().__init__(
            name="Sphere2D",
            dim=2,
            bounds=(np.array([-1.0, -1.0]), np.array([1.0, 1.0])),
            global_min=0.0,
            optimum_x=np.array([0.0, 0.0]),
        )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.atleast_2d(x)
        val = x_arr[:, 0] ** 2 + x_arr[:, 1] ** 2
        return val if x.ndim > 1 else val[0]


class BealeProblem(BaseProblem):
    """Beale: f(x, y) = (1.5 - x + xy)^2 + (2.25 - x + xy^2)^2 + (2.625 - x + xy^3)^2, bounds [0, 5]^2, min f(3, 0.5) = 0."""

    def __init__(self) -> None:
        super().__init__(
            name="Beale",
            dim=2,
            bounds=(np.array([0.0, 0.0]), np.array([5.0, 5.0])),
            global_min=0.0,
            optimum_x=np.array([3.0, 0.5]),
        )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.atleast_2d(x)
        x_val = x_arr[:, 0]
        y_val = x_arr[:, 1]
        term1 = (1.5 - x_val + x_val * y_val) ** 2
        term2 = (2.25 - x_val + x_val * (y_val**2)) ** 2
        term3 = (2.625 - x_val + x_val * (y_val**3)) ** 2
        val = term1 + term2 + term3
        return val if x.ndim > 1 else val[0]


class GoldsteinPriceProblem(BaseProblem):
    """Goldstein-Price function: bounds [-2, 2]^2, global min f(0, -1) = 3.0."""

    def __init__(self) -> None:
        super().__init__(
            name="GoldsteinPrice",
            dim=2,
            bounds=(np.array([-2.0, -2.0]), np.array([2.0, 2.0])),
            global_min=3.0,
            optimum_x=np.array([0.0, -1.0]),
        )

    def evaluate(self, x: np.ndarray) -> np.ndarray:
        x_arr = np.atleast_2d(x)
        x_val = x_arr[:, 0]
        y_val = x_arr[:, 1]

        part1 = 1.0 + (x_val + y_val + 1.0) ** 2 * (
            19.0
            - 14.0 * x_val
            + 3.0 * x_val**2
            - 14.0 * y_val
            + 6.0 * x_val * y_val
            + 3.0 * y_val**2
        )
        part2 = 30.0 + (2.0 * x_val - 3.0 * y_val) ** 2 * (
            18.0
            - 32.0 * x_val
            + 12.0 * x_val**2
            + 48.0 * y_val
            - 36.0 * x_val * y_val
            + 27.0 * y_val**2
        )
        val = part1 * part2
        return val if x.ndim > 1 else val[0]


BENCHMARKS_2D: dict[str, type[BaseProblem]] = {
    "scaled_sphere": ScaledSphereProblem,
    "himmelblau": HimmelblauProblem,
    "rosenbrock2d": Rosenbrock2DProblem,
    "sphere2d": Sphere2DProblem,
    "beale": BealeProblem,
    "goldstein_price": GoldsteinPriceProblem,
}
