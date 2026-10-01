"""Unit tests for jitter perturbation."""

import numpy as np

from leo.core.jitter import inject_jitter


def test_jitter_shape_and_bounds() -> None:
    x = np.array([[1.0, 2.0], [3.0, 4.0]])
    bounds = (np.array([0.0, 0.0]), np.array([5.0, 5.0]))
    rng = np.random.default_rng(42)

    jittered = inject_jitter(x, jitter_scale=1e-4, bounds=bounds, rng=rng)

    assert jittered.shape == x.shape
    assert not np.array_equal(jittered, x)
    assert np.all(np.abs(jittered - x) <= 1e-4 + 1e-9)
    assert np.all(jittered >= bounds[0])
    assert np.all(jittered <= bounds[1])


def test_jitter_clamping_at_boundaries() -> None:
    x = np.array([[0.0, 5.0]])
    bounds = (np.array([0.0, 0.0]), np.array([5.0, 5.0]))
    jittered = inject_jitter(x, jitter_scale=0.1, bounds=bounds)

    assert jittered[0, 0] >= 0.0
    assert jittered[0, 1] <= 5.0
