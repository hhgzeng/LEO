"""Prompt templates and builders for LEO.

Faithfully implements prompt structures from Tables 7, 8, 9, 10 in:
'Large language model-based evolutionary optimizer: Reasoning with elitism'.
"""

from __future__ import annotations

import numpy as np


def format_point_csv(x: np.ndarray, y: float | None = None) -> str:
    """Format single candidate coordinate and optional loss as comma-separated string."""
    coords = ", ".join(f"{v:.6f}" for v in x)
    if y is not None:
        return f"{coords}, {y:.6f}"
    return coords


def build_explore_prompt(
    x_jitter: np.ndarray,
    y: np.ndarray,
    num_new: int,
    bounds: tuple[np.ndarray, np.ndarray],
    var_names: list[str] | None = None,
) -> str:
    """Build LEO Explore prompt (Table 7).

    Instructs LLM to analyze the sorted population points and losses, deduce (min, max)
    limits, and sample scattered candidates in far-away regions to explore the landscape.
    """
    n_vars = x_jitter.shape[1]
    if var_names is None:
        var_names = [f"var{i + 1}" for i in range(n_vars)]

    var_header = ", ".join(var_names) + ", loss"
    rows = []
    for xi, yi in zip(x_jitter, y):
        row_str = ",".join(f"{v:.6f}" for v in xi) + f",{yi:.6f}"
        rows.append(row_str)
    points_csv = f"{var_header}; " + "; ".join(rows)

    prompt = (
        f"You are an optimization researcher tasked to minimize the value of loss. "
        f"Current candidate solutions for N={n_vars} variables in the order {', '.join(var_names)} "
        f"with their respective function loss in csv format are: [{points_csv}]. "
        f"You have to look at the above points and think of the (min, max) values for each variable "
        f"that might reduce or minimize the loss. With these limits of (min, max) values for each variable, "
        f"you must provide exactly {num_new} new, but completely different and scattered sets of values "
        f"from the above ones, to explore away regions for minimizing the function loss. "
        f"Generate the result like a csv file with {num_new} rows and {n_vars} columns, "
        f"where each row represents a candidate and each column represents a variable. "
        f"The response must only contain these numerical values in the csv format without column headers. "
        f"Do not provide additional text or explanation. Strictly, provide only the variable values "
        f"as floating point numbers with precision format %.6f."
    )
    return prompt


def build_exploit_prompt(
    best_x_jitter: np.ndarray,
    best_y: float,
    num_new: int,
    bounds: tuple[np.ndarray, np.ndarray],
    var_names: list[str] | None = None,
) -> str:
    """Build LEO Exploit prompt (Table 8).

    Instructs LLM to generate localized perturbations around the best candidate
    within domain bounds to exploit promising regions.
    """
    if best_x_jitter.ndim == 2:
        best_x_jitter = best_x_jitter[0]

    n_vars = len(best_x_jitter)
    if var_names is None:
        var_names = [f"var{i + 1}" for i in range(n_vars)]

    var_header = ", ".join(var_names) + ", loss"
    row_str = ", ".join(f"{v:.6f}" for v in best_x_jitter) + f", {best_y:.6f}"
    best_csv = f"{var_header}; {row_str}"

    lower_b, upper_b = bounds
    min_val_str = f"{float(np.min(lower_b)):.4g}"
    max_val_str = f"{float(np.max(upper_b)):.4g}"

    prompt = (
        f"You are an optimization researcher tasked to minimize the value of loss. "
        f"Current best candidate solution for N={n_vars} variables in the order {', '.join(var_names)}, "
        f"with their respective function loss in csv format are: [{best_csv}]. "
        f"Please provide exactly {num_new} new but different candidates of values for all variables, "
        f"in between {min_val_str} and {max_val_str}, to exploit close by regions for minimizing the function loss. "
        f"Generate the result like a csv file with {num_new} rows and {n_vars} columns, "
        f"where each row represents a candidate and each column represents a variable. "
        f"The response must only contain these numerical values in the csv format without column headers and row index. "
        f"Do not provide additional text or explanation. Strictly, provide only the variable values "
        f"as floating point numbers with precision format %.6f."
    )
    return prompt


def build_rnd_explore_prompt(
    num_new: int,
    n_vars: int,
    bounds: tuple[np.ndarray, np.ndarray],
) -> str:
    """Build LEO-Rnd Explore prompt (Table 5 & Table 7)."""
    lower_b, upper_b = bounds
    min_val_str = f"{float(np.min(lower_b)):.4g}"
    max_val_str = f"{float(np.max(upper_b)):.4g}"

    return (
        f"You must provide exactly {num_new} new, but completely different and scattered sets of values "
        f"within all variables, in between {min_val_str} and {max_val_str}. "
        f"Generate the result like a csv file with {num_new} rows and {n_vars} columns, "
        f"where each row represents a candidate and each column represents a variable. "
        f"The response must only contain these numerical values in the csv format without column headers. "
        f"Do not provide additional text or explanation. Strictly, provide only the variable values "
        f"as floating point numbers with precision format %.6f."
    )


def build_rnd_exploit_prompt(
    num_new: int,
    n_vars: int,
    bounds: tuple[np.ndarray, np.ndarray],
) -> str:
    """Build LEO-Rnd Exploit prompt (Table 5 & Table 8)."""
    lower_b, upper_b = bounds
    min_val_str = f"{float(np.min(lower_b)):.4g}"
    max_val_str = f"{float(np.max(upper_b)):.4g}"

    return (
        f"Please provide exactly {num_new} new but different candidates of values for all variables, "
        f"in between {min_val_str} and {max_val_str} to exploit close by regions. "
        f"Generate the result like a csv file with {num_new} rows and {n_vars} columns, "
        f"where each row represents a candidate and each column represents a variable. "
        f"The response must only contain these numerical values in the csv format without column headers and row index. "
        f"Do not provide additional text or explanation. Strictly, provide only the variable values "
        f"as floating point numbers with precision format %.6f."
    )


def build_modular_exploit_prompt(
    parent1_x: np.ndarray,
    parent1_f: np.ndarray,
    parent2_x: np.ndarray,
    parent2_f: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] = (
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
    ),
) -> str:
    """Build LEO-modular NSGA-II Exploit prompt (Table 9)."""
    p1_a, p1_b = parent1_x[0], parent1_x[1]
    p1_f1, p1_f2 = parent1_f[0], parent1_f[1]
    p2_a, p2_b = parent2_x[0], parent2_x[1]
    p2_f1, p2_f2 = parent2_f[0], parent2_f[1]

    return (
        f"You are an intelligent assistant who can understand great technical details. "
        f"You will help me minimize two functions by taking cues from given information. "
        f"Current candidate solutions (a, b) with their function loss are: "
        f"Candidate 1: a={p1_a:.4f}, b={p1_b:.4f}, function loss f1={p1_f1:.4f}, function loss f2={p1_f2:.4f}; "
        f"Candidate 2: a={p2_a:.4f}, b={p2_b:.4f}, function loss f1={p2_f1:.4f}, function loss f2={p2_f2:.4f}. "
        f"Please provide exactly 2 new but different pairs of values for 0<'a'<1 and 0<'b'<1 "
        f"to exploit closeby regions for minimizing the function loss1 and function loss2. "
        f"Take cues from the above points. The response should contain 4 values, "
        f"corresponding to 2 pairs in the format: 'a1, b1, a2, b2'. "
        f"Provide only these 4 numerical values in order as python [list], separated by commas, "
        f"with no additional text or explanation."
    )


def build_modular_explore_prompt(
    parent1_x: np.ndarray,
    parent1_f: np.ndarray,
    parent2_x: np.ndarray,
    parent2_f: np.ndarray,
    bounds: tuple[np.ndarray, np.ndarray] = (
        np.array([0.0, 0.0]),
        np.array([1.0, 1.0]),
    ),
) -> str:
    """Build LEO-modular NSGA-II Explore prompt (Table 9)."""
    p1_a, p1_b = parent1_x[0], parent1_x[1]
    p1_f1, p1_f2 = parent1_f[0], parent1_f[1]
    p2_a, p2_b = parent2_x[0], parent2_x[1]
    p2_f1, p2_f2 = parent2_f[0], parent2_f[1]

    return (
        f"You are an intelligent assistant who can understand great technical details. "
        f"You will help me minimize two functions by taking cues from given information. "
        f"Current candidate solutions (a, b) with their function loss are: "
        f"Candidate 1: a={p1_a:.4f}, b={p1_b:.4f}, function Loss f1={p1_f1:.4f}, function Loss f2={p1_f2:.4f}; "
        f"Candidate 2: a={p2_a:.4f}, b={p2_b:.4f}, function Loss f1={p2_f1:.4f}, function Loss f2={p2_f2:.4f}. "
        f"Please provide exactly 2 new but different pairs of values for 0<'a'<1 and 0<'b'<1 "
        f"to explore away regions for minimizing the function loss1 and function loss2. "
        f"The response should contain 4 values, corresponding to 2 pairs in the format: 'a1, b1, a2, b2'. "
        f"Provide only these 4 numerical values in order, separated by commas, with no additional text or explanation."
    )
