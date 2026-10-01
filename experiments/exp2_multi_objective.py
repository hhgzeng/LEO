"""Experiment 2: Multi-Objective Optimization with LEO-Modular NSGA-II (Section 3.2, Fig. 6).

Evaluates LEO-modular NSGA-II vs Standard NSGA-II and Pymoo Reference on ZDT1 and ZDT3.
Produces publication-quality Pareto front plots matching Figure 6.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np

from leo.llm.client import LLMClient, LLMSettings
from leo.modular.nsga2 import LEOModularNSGA2, StandardNSGA2
from leo.modular.plotting import plot_pareto_panel, setup_plot_style
from leo.problems.multi_objective import ZDT1Problem, ZDT3Problem


def run_experiment(
    problem: str = "all",
    use_mock: bool = True,
    seed: int = 42,
    output_dir: str = "tmp_results",
) -> None:
    """Run ZDT1, ZDT3 or both, and generate Figure 6 Pareto plots."""
    setup_plot_style()
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    fig_archive_path = Path("experiments/figures")
    fig_archive_path.mkdir(parents=True, exist_ok=True)

    settings = LLMSettings(leo_mock_llm=use_mock)
    client = LLMClient(settings=settings)

    run_zdt1 = problem.lower() in ["all", "zdt1"]
    run_zdt3 = problem.lower() in ["all", "zdt3"]

    mode_str = "MOCK" if use_mock else "LIVE (API)"
    print(f"\n{'='*60}")
    print(f" Running Section 3.2 Multi-Objective Experiment [{mode_str}] (seed={seed})")
    print(f" Target: {problem.upper()} | Output dir: {out_path.resolve()}")
    print(f"{'='*60}")

    zdt1_data = None
    zdt3_data = None

    # --- Run ZDT1 ---
    if run_zdt1:
        print("\n[1/2] Optimizing ZDT1 (Pop=10, Gens=40)...")
        prob_zdt1 = ZDT1Problem(dim=2)
        print("  -> Running LEO-modular NSGA-II...")
        leo_zdt1 = LEOModularNSGA2(
            prob_zdt1, pop_size=10, max_generations=40, llm_client=client, seed=seed
        ).optimize()
        print(f"     Found {len(leo_zdt1.pareto_f)} non-dominated solutions.")

        print("  -> Running Standard NSGA-II baseline...")
        std_zdt1 = StandardNSGA2(
            prob_zdt1, pop_size=10, max_generations=40, seed=seed
        ).optimize()
        true_front_zdt1 = prob_zdt1.get_pareto_front(n_points=300)

        zdt1_data = (true_front_zdt1, std_zdt1.pareto_f, leo_zdt1.pareto_f)

        # Plot individual ZDT1 figure
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_pareto_panel(
            ax,
            true_front_zdt1,
            std_zdt1.pareto_f,
            leo_zdt1.pareto_f,
            title="(a) test case: ZDT1",
            is_zdt3=False,
        )
        plt.tight_layout()
        zdt1_fig_file = out_path / f"zdt1_pareto_front_{mode_str.lower().split()[0]}.png"
        plt.savefig(zdt1_fig_file, dpi=200)
        plt.close()
        print(f"  [+] Saved ZDT1 plot to: {zdt1_fig_file}")

    # --- Run ZDT3 ---
    if run_zdt3:
        print("\n[2/2] Optimizing ZDT3 (Pop=30, Gens=40)...")
        prob_zdt3 = ZDT3Problem(dim=2)
        print("  -> Running LEO-modular NSGA-II...")
        leo_zdt3 = LEOModularNSGA2(
            prob_zdt3, pop_size=30, max_generations=40, llm_client=client, seed=seed
        ).optimize()
        print(f"     Found {len(leo_zdt3.pareto_f)} non-dominated solutions.")

        print("  -> Running Standard NSGA-II baseline...")
        std_zdt3 = StandardNSGA2(
            prob_zdt3, pop_size=30, max_generations=40, seed=seed
        ).optimize()
        true_front_zdt3 = prob_zdt3.get_pareto_front(n_points=1000)

        zdt3_data = (true_front_zdt3, std_zdt3.pareto_f, leo_zdt3.pareto_f)

        # Plot individual ZDT3 figure
        fig, ax = plt.subplots(figsize=(7, 6))
        plot_pareto_panel(
            ax,
            true_front_zdt3,
            std_zdt3.pareto_f,
            leo_zdt3.pareto_f,
            title="(b) test case: ZDT3",
            is_zdt3=True,
        )
        plt.tight_layout()
        zdt3_fig_file = out_path / f"zdt3_pareto_front_{mode_str.lower().split()[0]}.png"
        plt.savefig(zdt3_fig_file, dpi=200)
        plt.close()
        print(f"  [+] Saved ZDT3 plot to: {zdt3_fig_file}")

    # --- Plot Combined Figure 6 (if both were run) ---
    if zdt1_data is not None and zdt3_data is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))
        plot_pareto_panel(
            ax1,
            zdt1_data[0],
            zdt1_data[1],
            zdt1_data[2],
            title="(a) test case: ZDT1",
            is_zdt3=False,
        )
        plot_pareto_panel(
            ax2,
            zdt3_data[0],
            zdt3_data[1],
            zdt3_data[2],
            title="(b) test case: ZDT3",
            is_zdt3=True,
        )
        plt.tight_layout()

        suffix = mode_str.lower().split()[0]
        combined_fig_file = out_path / f"fig6_pareto_fronts_{suffix}.png"
        plt.savefig(combined_fig_file, dpi=200)
        plt.close()

        # Archive to experiments/figures
        shutil.copy(combined_fig_file, fig_archive_path / f"fig6_pareto_fronts_{suffix}.png")
        print(f"\n[+] Successfully generated Figure 6 reproduction: {combined_fig_file}")

    print(f"\nAll requested figures are saved in: {out_path.resolve()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Multi-Objective Experiment (Section 3.2, ZDT1 & ZDT3)"
    )
    parser.add_argument(
        "--problem",
        "-p",
        type=str,
        default="all",
        choices=["all", "zdt1", "zdt3"],
        help="Target problem: 'zdt1', 'zdt3', or 'all' (default: all)",
    )
    parser.add_argument(
        "--mock", action="store_true", default=True, help="Use deterministic mock LLM"
    )
    parser.add_argument(
        "--live", dest="mock", action="store_false", help="Use live API"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="tmp_results",
        help="Directory to save output plots (default: 'tmp_results')",
    )
    args = parser.parse_args()

    run_experiment(
        problem=args.problem,
        use_mock=args.mock,
        seed=args.seed,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
