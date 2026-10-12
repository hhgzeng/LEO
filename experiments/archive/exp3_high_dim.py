"""Experiment 3: High-Dimensional Benchmark Optimization (Section 3.3, Fig. 7).

Evaluates LEO on Shifted N-dimensional Rosenbrock (a = 0.2913) across
dimensions: 2, 4, 6, 8, 10, 20, 25 for 100 optimization iterations.
Produces publication-quality convergence plots matching Figure 7.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from leo.core.algorithm import LEOAlgorithm
from leo.llm.client import LLMClient, LLMSettings
from leo.problems.rosenbrock_nd import ShiftedRosenbrockND


def setup_plot_style() -> None:
    """Configure matplotlib for clean publication-quality aesthetic."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 12
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True


def run_experiment(
    num_seeds: int = 1,
    dimensions: list[int] | None = None,
    use_mock: bool = True,
    output_dir: str = "tmp_results",
) -> None:
    setup_plot_style()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig_archive_path = Path("experiments/figures")
    fig_archive_path.mkdir(parents=True, exist_ok=True)

    settings = LLMSettings(leo_mock_llm=use_mock)
    client = LLMClient(settings=settings)

    if dimensions is None:
        dimensions = [2, 4, 6, 8, 10, 20, 25]

    max_iters = 100
    pop_size = 10
    mode_suffix = "mock" if use_mock else "live"

    print(f"\n{'=' * 70}")
    print(
        f" Running Section 3.3 High-Dim Rosenbrock [{mode_suffix.upper()}] (iters={max_iters}, pop={pop_size})"
    )
    print(
        f" Dimensions: {dimensions} | Seeds count: {num_seeds} | Output dir: {out_path.resolve()}"
    )
    print(f"{'=' * 70}")

    dim_curves: dict[int, np.ndarray] = {}

    for d in dimensions:
        print(f"\n>>> Evaluating dimension D = {d} over {num_seeds} seed(s)...")
        histories: list[list[float]] = []

        for seed in range(num_seeds):
            problem = ShiftedRosenbrockND(dim=d)
            algo = LEOAlgorithm(
                problem=problem,
                pop_size=pop_size,
                max_iters=max_iters,
                llm_client=client,
                seed=seed,
                verbose=False,
            )
            res = algo.optimize()
            histories.append(res.history_exploit_best_y)
            print(f"    Seed {seed + 1}/{num_seeds}: final f_min = {res.best_y:.4e}")

        # Average trajectory over seeds
        mean_curve = np.mean(histories, axis=0)
        dim_curves[d] = mean_curve

    # Plot Fig. 7: High-dimensional convergence trajectories
    plt.figure(figsize=(9, 6))
    colors = plt.cm.plasma(np.linspace(0.1, 0.85, len(dimensions)))

    for (d, curve), col in zip(dim_curves.items(), colors):
        plt.plot(curve, label=f"$n_{{dim}} = {d}$", color=col, linewidth=2.0)

    plt.title(
        "Convergence Pattern of $f_{min}$ for High-Dimensional Rosenbrock (Fig. 7)",
        fontsize=13,
        pad=10,
    )
    plt.xlabel("Optimization Iterations", fontsize=12)
    plt.ylabel("Mean $f_{min}$ (Exploit Pool)", fontsize=12)
    plt.yscale("log")
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend(loc="upper right", frameon=True, fontsize=11)
    plt.tight_layout()

    fig_file = out_path / f"fig7_high_dim_rosenbrock_{mode_suffix}.png"
    plt.savefig(fig_file, dpi=200)
    plt.close()

    # Archive copy to experiments/figures
    shutil.copy(
        fig_file, fig_archive_path / f"fig7_high_dim_rosenbrock_{mode_suffix}.png"
    )
    print(f"\n[+] Successfully saved Figure 7 to: {fig_file}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run High-Dimensional Rosenbrock Experiment (Section 3.3, Fig. 7)"
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of random seeds to average over (default: 1 for quick test, 30 for paper reproduction)",
    )
    parser.add_argument(
        "--dim",
        "-d",
        type=str,
        default="all",
        help="Dimension to test: single int e.g. '10', comma-separated '2,4,8', or 'all' (default: all)",
    )
    parser.add_argument(
        "--mock", action="store_true", default=True, help="Use deterministic mock LLM"
    )
    parser.add_argument(
        "--live", dest="mock", action="store_false", help="Use live API"
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="tmp_results",
        help="Directory to save output plots (default: 'tmp_results')",
    )
    args = parser.parse_args()

    if args.dim.lower() == "all":
        dims = [2, 4, 6, 8, 10, 20, 25]
    else:
        dims = [int(x.strip()) for x in args.dim.split(",")]

    run_experiment(
        num_seeds=args.seeds,
        dimensions=dims,
        use_mock=args.mock,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
