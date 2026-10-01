"""Jitter perturbation generator for LEO.

Adds a small decimal perturbation (~10^-4 to 10^-3) to numerical coordinates
before inserting them into prompt templates to prevent autoregressive mode collapse
and token hallucination.
"""

from __future__ import annotations

import numpy as np


def inject_jitter(
    x: np.ndarray,
    jitter_scale: float = 1e-4,
    bounds: tuple[np.ndarray, np.ndarray] | None = None,
    rng: np.random.Generator | None = None,
) -> np.ndarray:
    """Inject small perturbation into candidate coordinates.

    Args:
        x: Candidate coordinates array of shape (N, D) or (D,).
        jitter_scale: Magnitude of jitter noise (e.g. 1e-4).
        bounds: Optional (lower_bounds, upper_bounds) tuple for clamping.
        rng: Optional NumPy random generator instance.

    Returns:
        Jittered coordinates array with the same shape as x.
    """
    if rng is None:
        rng = np.random.default_rng()

    # Sample uniform noise in [-jitter_scale, jitter_scale]
    jitter = rng.uniform(-jitter_scale, jitter_scale, size=x.shape)
    x_jitter = x + jitter

    if bounds is not None:
        lower, upper = bounds
        x_jitter = np.clip(x_jitter, lower, upper)

    return x_jitter
