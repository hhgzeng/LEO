"""Unit tests for LLM response parser."""

import numpy as np

from leo.llm.parser import parse_candidate_csv, parse_modular_list


def test_parse_candidate_csv_clean() -> None:
    raw = """
    0.123456, 0.654321
    -1.500000, 1.200000
    """
    bounds = (np.array([-2.0, -2.0]), np.array([2.0, 2.0]))
    parsed = parse_candidate_csv(raw, expected_rows=2, expected_cols=2, bounds=bounds)

    assert parsed.shape == (2, 2)
    np.testing.assert_allclose(parsed[0], [0.123456, 0.654321])
    np.testing.assert_allclose(parsed[1], [-1.500000, 1.200000])


def test_parse_candidate_csv_with_fences_and_headers() -> None:
    raw = """
    Here are the candidates:
    ```csv
    var1, var2
    0.500000, -0.800000
    1.100000, 0.900000
    ```
    I hope this helps!
    """
    bounds = (np.array([-2.0, -2.0]), np.array([2.0, 2.0]))
    parsed = parse_candidate_csv(raw, expected_rows=2, expected_cols=2, bounds=bounds)

    assert parsed.shape == (2, 2)
    np.testing.assert_allclose(parsed[0], [0.5, -0.8])
    np.testing.assert_allclose(parsed[1], [1.1, 0.9])


def test_parse_candidate_csv_clamping() -> None:
    raw = """
    -10.000000, 50.000000
    """
    bounds = (np.array([-2.0, -2.0]), np.array([2.0, 2.0]))
    parsed = parse_candidate_csv(raw, expected_rows=1, expected_cols=2, bounds=bounds)

    assert parsed[0, 0] == -2.0
    assert parsed[0, 1] == 2.0


def test_parse_modular_list() -> None:
    raw = "[0.1234, 0.5678, 0.9101, 0.1121]"
    bounds = (np.array([0.0, 0.0]), np.array([1.0, 1.0]))
    parsed = parse_modular_list(raw, expected_pairs=2, expected_dim=2, bounds=bounds)

    assert parsed.shape == (2, 2)
    np.testing.assert_allclose(parsed[0], [0.1234, 0.5678])
    np.testing.assert_allclose(parsed[1], [0.9101, 0.1121])
