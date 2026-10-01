"""Plotting utilities for multi-objective optimization (Section 3.2, Figure 6)."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np


def setup_plot_style() -> None:
    """Configure matplotlib for clean publication-quality aesthetic."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True


def plot_pareto_panel(
    ax: plt.Axes,
    true_front: np.ndarray,
    std_pareto_f: np.ndarray,
    leo_pareto_f: np.ndarray,
    title: str,
    is_zdt3: bool = False,
) -> None:
    """Plot a single Pareto front panel matching Figure 6 style exactly."""
    # 1. Pymoo True/Reference Front (Black dashed curve)
    if is_zdt3:
        sorted_indices = np.argsort(true_front[:, 0])
        front_sorted = true_front[sorted_indices]
        # Insert NaNs where gaps occur between disconnected segments
        diffs = np.diff(front_sorted[:, 0])
        split_idx = np.where(diffs > 0.02)[0] + 1
        front_with_nans = np.insert(front_sorted, split_idx, np.nan, axis=0)
        (h_pymoo,) = ax.plot(
            front_with_nans[:, 0],
            front_with_nans[:, 1],
            "k--",
            linewidth=1.2,
            label="NSGAII Pymoo",
            zorder=2,
        )
    else:
        sorted_indices = np.argsort(true_front[:, 0])
        front_sorted = true_front[sorted_indices]
        (h_pymoo,) = ax.plot(
            front_sorted[:, 0],
            front_sorted[:, 1],
            "k--",
            linewidth=1.2,
            label="NSGAII Pymoo",
            zorder=2,
        )

    # 2. LEO-modular (Dark blue circles with black edge)
    h_leo = ax.scatter(
        leo_pareto_f[:, 0],
        leo_pareto_f[:, 1],
        facecolor="#0b409c",
        edgecolor="black",
        linewidth=0.8,
        marker="o",
        s=80,
        label="LEO-modular",
        zorder=4,
    )

    # 3. Traditional NSGA-II (Purple cross)
    h_nsga = ax.scatter(
        std_pareto_f[:, 0],
        std_pareto_f[:, 1],
        color="#8e24aa",
        marker="x",
        s=85,
        linewidths=2.0,
        label="NSGAII",
        zorder=3,
    )

    ax.set_title(title, fontsize=14, pad=8)
    ax.set_xlabel("Objective function, $f_1$", fontsize=13)
    ax.set_ylabel("Objective function, $f_2$", fontsize=13)

    if is_zdt3:
        ax.set_xlim(-0.15, 1.05)
        ax.set_ylim(-1.15, 1.25)
        ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
        ax.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    else:
        ax.set_xlim(-0.15, 1.10)
        ax.set_ylim(-0.15, 1.10)
        ax.set_xticks([0.00, 0.25, 0.50, 0.75, 1.00])
        ax.set_yticks([0.00, 0.25, 0.50, 0.75, 1.00])

    # Legend order matching paper: LEO-modular, NSGAII, NSGAII Pymoo
    ax.legend(
        handles=[h_leo, h_nsga, h_pymoo],
        labels=["LEO-modular", "NSGAII", "NSGAII Pymoo"],
        loc="upper right",
        frameon=True,
        fontsize=12,
    )
