"""Parser for LLM generated CSV tables and numerical lists.

Extracts floating-point candidate matrices, strips markdown formatting,
validates shapes, and clamps coordinates to variable domain bounds.
"""

from __future__ import annotations

import re

import numpy as np

# Matches standard float representations including scientific notation
FLOAT_REGEX = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def parse_candidate_csv(
    response_text: str,
    expected_rows: int,
    expected_cols: int,
    bounds: tuple[np.ndarray, np.ndarray],
    fallback_generator: np.ndarray | None = None,
) -> np.ndarray:
    """Parse candidate matrix from LLM text response.

    Args:
        response_text: Raw response string from LLM.
        expected_rows: Expected number of candidate solutions (N_pop).
        expected_cols: Expected dimension of variables (num_vars).
        bounds: (lower_bounds, upper_bounds) tuple for coordinate clamping.
        fallback_generator: Optional (N, D) array to draw fallback rows from if parsing fails.

    Returns:
        np.ndarray of shape (expected_rows, expected_cols), clamped to bounds.
    """
    lower, upper = bounds
    lower = np.asarray(lower, dtype=np.float64)
    upper = np.asarray(upper, dtype=np.float64)

    # 1. Clean response: remove markdown code blocks
    text = response_text.strip()
    # Strip markdown fences ```csv ... ``` or ``` ... ```
    text = re.sub(r"```(?:csv|text|python)?\s*", "", text, flags=re.IGNORECASE)
    text = text.replace("```", "")

    # 2. Strategy A: Line-by-line parsing to avoid header contamination
    candidates: list[list[float]] = []
    lines = text.splitlines()

    for line in lines:
        line = line.strip()
        if not line:
            continue
        # Skip pure header strings (like "var1, var2")
        if re.search(r"[a-zA-Z]{3,}", line) and not any(
            kw in line.lower() for kw in ["e-", "e+"]
        ):
            continue

        numbers = [float(match.group(0)) for match in FLOAT_REGEX.finditer(line)]
        if len(numbers) == expected_cols:
            candidates.append(numbers)
        elif len(numbers) > expected_cols and len(candidates) < expected_rows:
            # Maybe multiple rows or semicolon-delimited on one line
            for chunk_start in range(
                0, len(numbers) - expected_cols + 1, expected_cols
            ):
                chunk = numbers[chunk_start : chunk_start + expected_cols]
                candidates.append(chunk)
                if len(candidates) == expected_rows:
                    break

    # 3. Strategy B: Global regex extract if line-by-line yielded insufficient rows
    if len(candidates) < expected_rows:
        all_numbers = [float(match.group(0)) for match in FLOAT_REGEX.finditer(text)]
        total_needed = expected_rows * expected_cols
        if len(all_numbers) >= total_needed:
            candidates = []
            for i in range(expected_rows):
                start = i * expected_cols
                candidates.append(all_numbers[start : start + expected_cols])

    # 4. Fallback handling for missing rows
    parsed_array = (
        np.array(candidates, dtype=np.float64)
        if candidates
        else np.empty((0, expected_cols))
    )

    if len(parsed_array) < expected_rows:
        missing_count = expected_rows - len(parsed_array)
        if fallback_generator is not None and len(fallback_generator) >= missing_count:
            # Use provided fallback points with tiny perturbation
            fallback_rows = fallback_generator[:missing_count] + np.random.uniform(
                -1e-3, 1e-3, size=(missing_count, expected_cols)
            )
        else:
            # Sample uniformly within bounds
            fallback_rows = np.random.uniform(
                lower, upper, size=(missing_count, expected_cols)
            )

        if len(parsed_array) > 0:
            parsed_array = np.vstack([parsed_array, fallback_rows])
        else:
            parsed_array = fallback_rows

    # Ensure shape (expected_rows, expected_cols)
    result = parsed_array[:expected_rows, :expected_cols]

    # Clamp to domain bounds
    return np.clip(result, lower, upper)


def parse_modular_list(
    response_text: str,
    expected_pairs: int = 2,
    expected_dim: int = 2,
    bounds: tuple[np.ndarray, np.ndarray] = (
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
    ),
) -> np.ndarray:
    """Parse comma-separated list of numbers for LEO-modular NSGA-II offspring.

    Expected output: [a1, b1, a2, b2] -> shape (2, 2).
    """
    total_nums = expected_pairs * expected_dim
    lower, upper = bounds

    # Extract all floats
    numbers = [float(m.group(0)) for m in FLOAT_REGEX.finditer(response_text)]

    if len(numbers) >= total_nums:
        selected = numbers[:total_nums]
        arr = np.array(selected, dtype=np.float64).reshape(expected_pairs, expected_dim)
    else:
        # Fallback to random uniform
        arr = np.random.uniform(lower, upper, size=(expected_pairs, expected_dim))

    return np.clip(arr, lower, upper)
