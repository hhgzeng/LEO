"""Experiment 1: 2D Benchmark Functions Comparison (Section 3.1, Table 3, Fig. 4).

Compares LEO against 6 classical baseline families (SGD, Adam, L-BFGS-B, SA, CMA-ES, COBYLA)
across 6 2D benchmark functions over multiple seeds, plotting convergence curves.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from leo.baselines.classical import (
    AdamOptimizer,
    CMAESOptimizer,
    COBYLAOptimizer,
    LBFGSOptimizer,
    SGDOptimizer,
    SimulatedAnnealingOptimizer,
)
from leo.core.algorithm import LEOAlgorithm
from leo.llm.client import LLMClient, LLMSettings
from leo.problems.benchmarks_2d import BENCHMARKS_2D


def run_experiment(
    num_seeds: int = 5, use_mock: bool = True, output_dir: str = "experiments/results"
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    fig_path = Path("experiments/figures")
    fig_path.mkdir(parents=True, exist_ok=True)

    settings = LLMSettings(leo_mock_llm=use_mock)
    client = LLMClient(settings=settings)

    results: dict[str, dict[str, list[float]]] = {}
    convergence_histories: dict[str, dict[str, list[float]]] = {}

    optimizers = [
        ("SGD", lambda p, s: SGDOptimizer(p, lr=0.01, momentum=0.0, seed=s)),
        ("SGD (nu=0.5)", lambda p, s: SGDOptimizer(p, lr=0.01, momentum=0.5, seed=s)),
        ("SGD (nu=0.9)", lambda p, s: SGDOptimizer(p, lr=0.01, momentum=0.9, seed=s)),
        ("Adam (b1=0.1)", lambda p, s: AdamOptimizer(p, lr=0.05, beta1=0.1, seed=s)),
        ("Adam (b1=0.5)", lambda p, s: AdamOptimizer(p, lr=0.05, beta1=0.5, seed=s)),
        ("Adam (b1=0.9)", lambda p, s: AdamOptimizer(p, lr=0.05, beta1=0.9, seed=s)),
        ("L-BFGS-B", lambda p, s: LBFGSOptimizer(p, seed=s)),
        (
            "SA (T=0.1)",
            lambda p, s: SimulatedAnnealingOptimizer(p, initial_temp=0.1, seed=s),
        ),
        (
            "SA (T=1)",
            lambda p, s: SimulatedAnnealingOptimizer(p, initial_temp=1.0, seed=s),
        ),
        (
            "SA (T=10)",
            lambda p, s: SimulatedAnnealingOptimizer(p, initial_temp=10.0, seed=s),
        ),
        ("CMA-ES", lambda p, s: CMAESOptimizer(p, seed=s)),
        ("COBYLA", lambda p, s: COBYLAOptimizer(p, seed=s)),
        (
            "LEO",
            lambda p, s: LEOAlgorithm(
                p, pop_size=10, max_iters=30, llm_client=client, seed=s
            ),
        ),
    ]

    for prob_cls in BENCHMARKS_2D.values():
        prob_name = prob_cls().name
        print(f"\n>>> Running Benchmark: {prob_name} (Seeds: {num_seeds})")
        results[prob_name] = {}
        convergence_histories[prob_name] = {}

        for opt_name, opt_factory in optimizers:
            final_losses: list[float] = []
            histories: list[list[float]] = []

            for seed in range(num_seeds):
                problem = prob_cls()
                opt = opt_factory(problem, seed)
                res = opt.optimize()

                best_val = res.best_y
                final_losses.append(float(best_val))

                if opt_name == "LEO":
                    histories.append(res.history_exploit_best_y)
                else:
                    histories.append(res.history_best_y)

            results[prob_name][opt_name] = final_losses

            # Compute median trajectory for plotting
            max_len = max(len(h) for h in histories)
            padded = np.full((len(histories), max_len), np.nan)
            for i, h in enumerate(histories):
                padded[i, : len(h)] = h
                padded[i, len(h) :] = h[-1]
            median_traj = np.nanmedian(padded, axis=0)
            convergence_histories[prob_name][opt_name] = median_traj.tolist()

    # Create Summary DataFrame of Medians (matching Table 3)
    summary_data = []
    for prob_name, opt_dict in results.items():
        row = {"Problem": prob_name}
        for opt_name, vals in opt_dict.items():
            row[opt_name] = float(np.median(vals))
        summary_data.append(row)

    df_summary = pd.DataFrame(summary_data)
    print("\n=== Table 3 Reproduction (Median Objective Values) ===")
    print(df_summary.to_markdown(index=False))

    # Save to JSON
    with open(out_path / "table3_benchmarks.json", "w") as f:
        json.dump(summary_data, f, indent=2)

    # Plot Fig. 4: Convergence plots for the 6 functions
    _fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()

    for idx, (prob_name, opt_hist) in enumerate(convergence_histories.items()):
        ax = axes[idx]
        for opt_name, traj in opt_hist.items():
            # Highlight LEO, CMA-ES, L-BFGS-B
            if opt_name == "LEO":
                ax.plot(traj, label=opt_name, color="red", linewidth=2.5)
            elif opt_name in ["CMA-ES", "L-BFGS-B", "Adam (b1=0.9)"]:
                ax.plot(traj, label=opt_name, linestyle="--", alpha=0.8)

        ax.set_title(f"Convergence: {prob_name}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Iterations / Evaluations")
        ax.set_ylabel("f_min")
        ax.set_yscale("log")
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper right", fontsize=9)

    plt.tight_layout()
    fig_file = fig_path / "fig4_benchmarks_2d.png"
    plt.savefig(fig_file, dpi=200)
    plt.close()
    print(f"\n[+] Saved Figure 4 to {fig_file}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run 2D Benchmarks Experiment (Exp 1)")
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
