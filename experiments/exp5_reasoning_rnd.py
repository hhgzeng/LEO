"""Experiment 5: Proof of LLM Reasoning & Diminishing Variance (Section 4.1, Table 6, Fig. 9).

Compares LEO against LEO-Rnd ablation on the 2D Goldstein-Price problem over 100 iterations,
computing convergence statistics (Table 6) and Kernel Density Estimation (KDE) of the explore pool.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import gaussian_kde

from leo.baselines.leo_rnd import LEORndOptimizer
from leo.core.algorithm import LEOAlgorithm
from leo.llm.client import LLMClient, LLMSettings
from leo.problems.benchmarks_2d import GoldsteinPriceProblem


def run_experiment(num_seeds: int = 5, use_mock: bool = True) -> None:
    fig_path = Path("experiments/figures")
    fig_path.mkdir(parents=True, exist_ok=True)
    out_path = Path("experiments/results")
    out_path.mkdir(parents=True, exist_ok=True)

    settings = LLMSettings(leo_mock_llm=use_mock)
    client = LLMClient(settings=settings)

    print(
        f"\n>>> Running Experiment 5: Reasoning vs LEO-Rnd on Goldstein-Price (Seeds: {num_seeds})"
    )

    problem = GoldsteinPriceProblem()
    max_iters = 100
    pop_size = 10

    leo_finals: list[float] = []
    leo_rnd_finals: list[float] = []

    leo_best_trajectories: list[list[float]] = []
    leo_rnd_best_trajectories: list[list[float]] = []

    # Store explore pool coordinates across iterations for seed 0 to plot KDE
    leo_explore_history: list[np.ndarray] = []
    leo_rnd_explore_history: list[np.ndarray] = []

    for seed in range(num_seeds):
        # 1. Run LEO
        algo = LEOAlgorithm(
            problem=problem,
            pop_size=pop_size,
            max_iters=max_iters,
            llm_client=client,
            seed=seed,
            verbose=False,
        )
        res_leo = algo.optimize()
        leo_finals.append(res_leo.best_y)
        leo_best_trajectories.append(res_leo.history_exploit_best_y)
        if seed == 0:
            leo_explore_history = res_leo.history_explore_x

        # 2. Run LEO-Rnd
        rnd_algo = LEORndOptimizer(
            problem=problem,
            pop_size=pop_size,
            max_iters=max_iters,
            llm_client=client,
            seed=seed,
        )
        res_rnd = rnd_algo.optimize()
        leo_rnd_finals.append(res_rnd.best_y)
        leo_rnd_best_trajectories.append(res_rnd.history_exploit_best_y)
        if seed == 0:
            leo_rnd_explore_history = res_rnd.history_explore_x

    # Compute Table 6 Statistics
    leo_mean, leo_std = float(np.mean(leo_finals)), float(np.std(leo_finals))
    leo_med = float(np.median(leo_finals))

    rnd_mean, rnd_std = float(np.mean(leo_rnd_finals)), float(np.std(leo_rnd_finals))
    rnd_med = float(np.median(leo_rnd_finals))

    t6_data = [
        {
            "Method": "LEO",
            "Mean ± std": f"{leo_mean:.3f} ± {leo_std:.3f}",
            "Median": f"{leo_med:.3f}",
        },
        {
            "Method": "LEO-Rnd",
            "Mean ± std": f"{rnd_mean:.3f} ± {rnd_std:.3f}",
            "Median": f"{rnd_med:.3f}",
        },
    ]
    df_t6 = pd.DataFrame(t6_data)
    print("\n=== Table 6 Reproduction (Goldstein-Price) ===")
    print(df_t6.to_markdown(index=False))

    with open(out_path / "table6_reasoning.json", "w") as f:
        json.dump(t6_data, f, indent=2)

    # Plot Fig. 9: 6-Panel Diagnostic Figure
    _fig, axes = plt.subplots(3, 2, figsize=(15, 14))

    # (a) Exploit pool mean convergence
    leo_mean_traj = np.mean(leo_best_trajectories, axis=0)
    rnd_mean_traj = np.mean(leo_rnd_best_trajectories, axis=0)
    axes[0, 0].plot(leo_mean_traj, color="crimson", linewidth=2.2, label="LEO")
    axes[0, 0].plot(
        rnd_mean_traj, color="royalblue", linestyle="--", linewidth=2.2, label="LEO-Rnd"
    )
    axes[0, 0].set_title(
        "(a) Exploit pool (best): mean", fontsize=12, fontweight="bold"
    )
    axes[0, 0].set_xlabel("Optimization iterations")
    axes[0, 0].set_ylabel("Mean function value")
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend()
    axes[0, 0].grid(True, linestyle=":")

    # (b) Exploit pool median convergence
    leo_med_traj = np.median(leo_best_trajectories, axis=0)
    rnd_med_traj = np.median(leo_rnd_best_trajectories, axis=0)
    axes[0, 1].plot(leo_med_traj, color="crimson", linewidth=2.2, label="LEO")
    axes[0, 1].plot(
        rnd_med_traj, color="royalblue", linestyle="--", linewidth=2.2, label="LEO-Rnd"
    )
    axes[0, 1].set_title(
        "(b) Exploit pool (best): median", fontsize=12, fontweight="bold"
    )
    axes[0, 1].set_xlabel("Optimization iterations")
    axes[0, 1].set_ylabel("Median function value")
    axes[0, 1].set_yscale("log")
    axes[0, 1].legend()
    axes[0, 1].grid(True, linestyle=":")

    # KDE Plots for explore pool across iterations (every 5th iteration: 5, 10, ..., 50)
    iter_indices = list(range(0, min(len(leo_explore_history), 50), 5))
    x_eval = np.linspace(-2.0, 2.0, 200)
    colors = plt.cm.plasma(np.linspace(0.1, 0.9, len(iter_indices)))

    def plot_kdes(ax, histories, var_idx, title):
        for it_idx, col in zip(iter_indices, colors):
            pop_coords = histories[it_idx][:, var_idx]
            # Add small noise to avoid singular covariance matrix in KDE
            perturbed = pop_coords + np.random.normal(0, 1e-3, size=len(pop_coords))
            kde = gaussian_kde(perturbed, bw_method=0.4)
            ax.plot(
                x_eval,
                kde(x_eval),
                color=col,
                alpha=0.7,
                label=f"Iter {it_idx}" if it_idx in [0, 25, 45] else None,
            )
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Variable value")
        ax.set_ylabel("Density")
        ax.set_xlim(-2.0, 2.0)
        ax.grid(True, linestyle=":")
        if ax.get_legend_handles_labels()[0]:
            ax.legend(fontsize=8)

    # (c) LEO (var1)
    plot_kdes(
        axes[1, 0], leo_explore_history, 0, "(c) LEO (var1): Diminishing Variance"
    )
    # (d) LEO (var2)
    plot_kdes(
        axes[1, 1], leo_explore_history, 1, "(d) LEO (var2): Diminishing Variance"
    )
    # (e) LEO-Rnd (var1)
    plot_kdes(
        axes[2, 0], leo_rnd_explore_history, 0, "(e) LEO-Rnd (var1): Non-Diminishing"
    )
    # (f) LEO-Rnd (var2)
    plot_kdes(
        axes[2, 1], leo_rnd_explore_history, 1, "(f) LEO-Rnd (var2): Non-Diminishing"
    )

    plt.tight_layout()
    fig_file = fig_path / "fig9_reasoning_diminishing_variance.png"
    plt.savefig(fig_file, dpi=200)
    plt.close()
    print(f"\n[+] Saved Figure 9 to {fig_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Reasoning vs LEO-Rnd Experiment (Exp 5)"
    )
    parser.add_argument("--seeds", type=int, default=5, help="Number of random seeds")
    parser.add_argument(
        "--mock", action="store_true", default=True, help="Use mock LLM"
    )
    parser.add_argument(
        "--live", dest="mock", action="store_false", help="Use live API"
    )
    args = parser.parse_args()

    run_experiment(num_seeds=args.seeds, use_mock=args.mock)


if __name__ == "__main__":
    main()
