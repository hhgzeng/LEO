"""Experiment 4: Industry-Relevant Engineering Applications (Section 3.4, Fig. 8, 13, 14).

Evaluates LEO across:
1. Supersonic nozzle contour Bezier optimization (4D, Fig. 8a-c, Fig. 13).
2. 1D steady-state heat equation temperature profiles (dim = 2, 4, 8, Fig. 8d-f).
3. Wind farm layout optimization for 2, 4, 8 turbines (dim = 4, 8, 16, Fig. 8g-i, Fig. 14).
"""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import minimize

from leo.core.algorithm import LEOAlgorithm
from leo.llm.client import LLMClient, LLMSettings
from leo.problems.engineering import (
    HeatTransferProblem,
    NozzleShapeProblem,
    WindFarmProblem,
)


def setup_plot_style() -> None:
    """Configure matplotlib for clean publication-quality aesthetic."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.size"] = 11
    plt.rcParams["axes.linewidth"] = 1.0
    plt.rcParams["xtick.direction"] = "in"
    plt.rcParams["ytick.direction"] = "in"
    plt.rcParams["xtick.top"] = True
    plt.rcParams["ytick.right"] = True


def run_experiment(
    app: str = "all",
    num_seeds: int = 1,
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

    mode_suffix = "mock" if use_mock else "live"
    run_all = app.lower() == "all"
    run_nozzle = run_all or app.lower() in ["nozzle", "nozzle_shape"]
    run_heat = run_all or app.lower() in ["heat", "heat_transfer"]
    run_wind = run_all or app.lower() in ["wind", "windfarm", "wind_farm"]

    print(f"\n{'='*70}")
    print(f" Running Section 3.4 Engineering Applications [{mode_suffix.upper()}]")
    print(f" Target app: {app.upper()} | Seeds: {num_seeds} | Output dir: {out_path.resolve()}")
    print(f"{'='*70}")

    # Data holders for combined Fig 8
    nozzle_data = None
    heat_data = None
    wind_data = None

    # -------------------------------------------------------------
    # 1. Nozzle Shape Optimization
    # -------------------------------------------------------------
    if run_nozzle:
        print("\n[1/3] Running Nozzle Shape Optimization (4D Bezier, 30 iters)...")
        nozzle_prob = NozzleShapeProblem()
        nozzle_algo = LEOAlgorithm(
            nozzle_prob,
            pop_size=10,
            max_iters=30,
            llm_client=client,
            seed=42,
            verbose=False,
        )
        nozzle_res = nozzle_algo.optimize()

        # Compute contours
        init_params = nozzle_prob.sample_uniform(10, rng=np.random.default_rng(42))
        initial_contours = [nozzle_prob.get_contour(p) for p in init_params]
        final_explore_contours = [
            nozzle_prob.get_contour(p) for p in nozzle_res.history_explore_x[-1]
        ]
        optimal_contour = nozzle_prob.get_contour(nozzle_res.best_x)

        nozzle_data = (initial_contours, final_explore_contours, optimal_contour, nozzle_res)

        # Plot individual nozzle figure (Fig 8a-c)
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4.2))
        for xc, yc in initial_contours:
            ax1.plot(xc, yc, color="gray", alpha=0.5)
        ax1.set_title("(a) Initial Nozzle Shapes", fontsize=11, fontweight="bold")
        ax1.set_xlabel("x")
        ax1.set_ylabel("y")
        ax1.grid(True, linestyle=":", alpha=0.6)

        for xc, yc in final_explore_contours:
            ax2.plot(xc, yc, color="darkorange", alpha=0.6)
        ax2.set_title("(b) Explore Pool Shapes", fontsize=11, fontweight="bold")
        ax2.set_xlabel("x")
        ax2.set_ylabel("y")
        ax2.grid(True, linestyle=":", alpha=0.6)

        ax3.plot(optimal_contour[0], optimal_contour[1], color="#c92a2a", linewidth=2.5, label="Optimal Bell Contour")
        ax3.set_title("(c) Exploit Pool Optimal Shape", fontsize=11, fontweight="bold")
        ax3.set_xlabel("x")
        ax3.set_ylabel("y")
        ax3.legend(frameon=True)
        ax3.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        nozzle_fig_file = out_path / f"nozzle_shapes_{mode_suffix}.png"
        plt.savefig(nozzle_fig_file, dpi=200)
        plt.close()
        print(f"  [+] Saved Nozzle Shapes plot to: {nozzle_fig_file}")

    # -------------------------------------------------------------
    # 2. 1D Heat Transfer Optimization
    # -------------------------------------------------------------
    if run_heat:
        print("\n[2/3] Running Heat Transfer Optimization for Dim in {2, 4, 8} (100 iters)...")
        heat_results = {}
        last_heat_res = None
        for d in [2, 4, 8]:
            heat_prob = HeatTransferProblem(dim=d)
            heat_algo = LEOAlgorithm(
                heat_prob,
                pop_size=10,
                max_iters=100,
                llm_client=client,
                seed=42,
                verbose=False,
            )
            heat_res = heat_algo.optimize()
            last_heat_res = heat_res
            heat_results[d] = {
                "problem": heat_prob,
                "best_t": heat_res.best_x,
                "grid_x": heat_prob.grid_x,
                "exact_t": heat_prob.exact_solution(heat_prob.grid_x),
            }
            print(f"    Dim={d}: final residual loss = {heat_res.best_y:.4e}")

        heat_data = (heat_results, last_heat_res)

        # Plot individual heat transfer figure (Fig 8d-f)
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        for col_idx, d in enumerate([2, 4, 8]):
            ax_ht = axes[col_idx]
            h_data = heat_results[d]
            gx = h_data["grid_x"]
            exact = h_data["exact_t"]
            best_t_padded = np.zeros(d + 2)
            best_t_padded[1:-1] = h_data["best_t"]
            best_t_padded[-1] = 1.0

            ax_ht.plot(gx, exact, "k--", label="Exact Solution", linewidth=2.0)
            ax_ht.plot(gx, best_t_padded, "o-", color="#0b409c", label=f"LEO (Dim={d})", linewidth=1.8)
            ax_ht.set_title(f"Heat Profile: Dim={d}", fontsize=11, fontweight="bold")
            ax_ht.set_xlabel("x / L")
            ax_ht.set_ylabel("Temperature T")
            ax_ht.grid(True, linestyle=":", alpha=0.6)
            if col_idx == 0:
                ax_ht.legend(frameon=True)

        plt.tight_layout()
        heat_fig_file = out_path / f"heat_transfer_profiles_{mode_suffix}.png"
        plt.savefig(heat_fig_file, dpi=200)
        plt.close()
        print(f"  [+] Saved Heat Transfer Profiles to: {heat_fig_file}")

    # -------------------------------------------------------------
    # 3. Wind Farm Layout Optimization
    # -------------------------------------------------------------
    if run_wind:
        print(f"\n[3/3] Running Wind Farm Optimization for 2, 4, 8 turbines ({num_seeds} seed(s))...")
        wind_positions_leo: dict[int, list[np.ndarray]] = {2: [], 4: [], 8: []}
        wind_positions_slsqp: dict[int, list[np.ndarray]] = {2: [], 4: [], 8: []}

        for n_t in [2, 4, 8]:
            wind_prob = WindFarmProblem(n_turbines=n_t)
            for s in range(num_seeds):
                # LEO run
                w_algo = LEOAlgorithm(
                    wind_prob,
                    pop_size=10,
                    max_iters=30,
                    llm_client=client,
                    seed=s,
                    verbose=False,
                )
                w_res = w_algo.optimize()
                wind_positions_leo[n_t].append(w_res.best_x.reshape(n_t, 2))

                # SLSQP reference
                x0 = wind_prob.sample_uniform(1, rng=np.random.default_rng(s))[0]
                bnds = list(zip(wind_prob.lower_bounds, wind_prob.upper_bounds))
                slsqp_res = minimize(
                    lambda x, prob=wind_prob: prob.evaluate(x),
                    x0=x0,
                    method="SLSQP",
                    bounds=bnds,
                    options={"maxiter": 30},
                )
                wind_positions_slsqp[n_t].append(slsqp_res.x.reshape(n_t, 2))

        wind_data = (wind_positions_leo, wind_positions_slsqp)

        # Plot individual wind farm comparison
        fig, axes = plt.subplots(2, 3, figsize=(15, 9.5))
        for col_idx, n_t in enumerate([2, 4, 8]):
            # LEO row
            ax_l = axes[0, col_idx]
            coords_l = np.vstack(wind_positions_leo[n_t])
            ax_l.scatter(coords_l[:, 0], coords_l[:, 1], color="#0b409c", alpha=0.7, s=40, edgecolors="black")
            ax_l.set_xlim(0, 1000)
            ax_l.set_ylim(0, 1000)
            ax_l.set_title(f"LEO Turbine Layout: N={n_t}", fontsize=11, fontweight="bold")
            ax_l.set_xlabel("X (m)")
            ax_l.set_ylabel("Y (m)")
            ax_l.grid(True, linestyle=":", alpha=0.6)

            # SLSQP row
            ax_s = axes[1, col_idx]
            coords_s = np.vstack(wind_positions_slsqp[n_t])
            ax_s.scatter(coords_s[:, 0], coords_s[:, 1], color="#495057", alpha=0.7, s=40, edgecolors="black")
            ax_s.set_xlim(0, 1000)
            ax_s.set_ylim(0, 1000)
            ax_s.set_title(f"SLSQP Reference: N={n_t}", fontsize=11, fontweight="bold")
            ax_s.set_xlabel("X (m)")
            ax_s.set_ylabel("Y (m)")
            ax_s.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        wind_fig_file = out_path / f"windfarm_layout_{mode_suffix}.png"
        plt.savefig(wind_fig_file, dpi=200)
        plt.close()
        print(f"  [+] Saved Wind Farm layout to: {wind_fig_file}")

    # -------------------------------------------------------------
    # Plot Full Combined Fig. 8 (if all ran)
    # -------------------------------------------------------------
    if nozzle_data is not None and heat_data is not None and wind_data is not None:
        initial_contours, final_explore_contours, optimal_contour, nozzle_res = nozzle_data
        heat_results, last_heat_res = heat_data
        wind_positions_leo, wind_positions_slsqp = wind_data

        plt.figure(figsize=(18, 14))

        # Row 1: Nozzle shapes
        ax_nz1 = plt.subplot2grid((4, 3), (0, 0))
        for xc, yc in initial_contours:
            ax_nz1.plot(xc, yc, color="gray", alpha=0.5)
        ax_nz1.set_title("(a) Initial Nozzle Shapes", fontsize=11, fontweight="bold")
        ax_nz1.set_xlabel("x")
        ax_nz1.set_ylabel("y")
        ax_nz1.grid(True, linestyle=":", alpha=0.6)

        ax_nz2 = plt.subplot2grid((4, 3), (0, 1))
        for xc, yc in final_explore_contours:
            ax_nz2.plot(xc, yc, color="darkorange", alpha=0.6)
        ax_nz2.set_title("(b) Explore Pool Nozzle Shapes", fontsize=11, fontweight="bold")
        ax_nz2.set_xlabel("x")
        ax_nz2.set_ylabel("y")
        ax_nz2.grid(True, linestyle=":", alpha=0.6)

        ax_nz3 = plt.subplot2grid((4, 3), (0, 2))
        ax_nz3.plot(optimal_contour[0], optimal_contour[1], color="#c92a2a", linewidth=2.5, label="Optimal Bell Contour")
        ax_nz3.set_title("(c) Exploit Pool Optimal Nozzle Shape", fontsize=11, fontweight="bold")
        ax_nz3.set_xlabel("x")
        ax_nz3.set_ylabel("y")
        ax_nz3.legend(frameon=True)
        ax_nz3.grid(True, linestyle=":", alpha=0.6)

        # Row 2: Heat transfer temperature profiles
        for col_idx, d in enumerate([2, 4, 8]):
            ax_ht = plt.subplot2grid((4, 3), (1, col_idx))
            h_data = heat_results[d]
            gx = h_data["grid_x"]
            exact = h_data["exact_t"]
            best_t_padded = np.zeros(d + 2)
            best_t_padded[1:-1] = h_data["best_t"]
            best_t_padded[-1] = 1.0

            ax_ht.plot(gx, exact, "k--", label="Exact Solution", linewidth=2.0)
            ax_ht.plot(gx, best_t_padded, "o-", color="#0b409c", label=f"LEO (Dim={d})", linewidth=1.8)
            letter = chr(ord('d') + col_idx)
            ax_ht.set_title(f"({letter}) Heat Profile: Dim={d}", fontsize=11, fontweight="bold")
            ax_ht.set_xlabel("x / L")
            ax_ht.set_ylabel("Temperature T")
            ax_ht.grid(True, linestyle=":", alpha=0.6)
            if col_idx == 0:
                ax_ht.legend(frameon=True)

        # Row 3: Wind farm LEO KDE scatter
        for col_idx, n_t in enumerate([2, 4, 8]):
            ax_wf_leo = plt.subplot2grid((4, 3), (2, col_idx))
            coords = np.vstack(wind_positions_leo[n_t])
            ax_wf_leo.scatter(coords[:, 0], coords[:, 1], color="#0b409c", alpha=0.7, s=40, edgecolors="black")
            ax_wf_leo.set_xlim(0, 1000)
            ax_wf_leo.set_ylim(0, 1000)
            letter = chr(ord('g') + col_idx)
            ax_wf_leo.set_title(f"({letter}) LEO Turbine: N={n_t}", fontsize=11, fontweight="bold")
            ax_wf_leo.set_xlabel("X (m)")
            ax_wf_leo.set_ylabel("Y (m)")
            ax_wf_leo.grid(True, linestyle=":", alpha=0.6)

        # Row 4: Wind farm SLSQP KDE scatter
        for col_idx, n_t in enumerate([2, 4, 8]):
            ax_wf_ref = plt.subplot2grid((4, 3), (3, col_idx))
            coords = np.vstack(wind_positions_slsqp[n_t])
            ax_wf_ref.scatter(coords[:, 0], coords[:, 1], color="#495057", alpha=0.7, s=40, edgecolors="black")
            ax_wf_ref.set_xlim(0, 1000)
            ax_wf_ref.set_ylim(0, 1000)
            letter = chr(ord('j') + col_idx)
            ax_wf_ref.set_title(f"({letter}) SLSQP Turbine: N={n_t}", fontsize=11, fontweight="bold")
            ax_wf_ref.set_xlabel("X (m)")
            ax_wf_ref.set_ylabel("Y (m)")
            ax_wf_ref.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        fig8_file = out_path / f"fig8_engineering_solutions_{mode_suffix}.png"
        plt.savefig(fig8_file, dpi=200)
        plt.close()

        shutil.copy(fig8_file, fig_archive_path / f"fig8_engineering_solutions_{mode_suffix}.png")
        print(f"\n[+] Successfully generated Figure 8 reproduction: {fig8_file}")

        # Plot Fig 13 & 14 convergence curves
        _fig, (ax_c1, ax_c2) = plt.subplots(1, 2, figsize=(14, 5))
        ax_c1.plot(nozzle_res.history_exploit_best_y, label="Exploit Pool (Min)", color="red", linewidth=2)
        ax_c1.plot(nozzle_res.history_exploit_mean_y, label="Exploit Pool (Mean)", color="darkred", linestyle="--")
        ax_c1.plot(nozzle_res.history_explore_best_y, label="Explore Pool (Min)", color="blue", linewidth=2)
        ax_c1.plot(nozzle_res.history_explore_mean_y, label="Explore Pool (Mean)", color="darkblue", linestyle="--")
        ax_c1.set_title("Nozzle Optimization Convergence (Fig. 13)", fontsize=12, fontweight="bold")
        ax_c1.set_xlabel("Iterations")
        ax_c1.set_ylabel("Objective Loss")
        ax_c1.set_yscale("log")
        ax_c1.legend(frameon=True)
        ax_c1.grid(True, linestyle=":", alpha=0.6)

        if last_heat_res is not None:
            ax_c2.plot(last_heat_res.history_exploit_best_y, label="Heat Conduction (Exploit Best)", color="darkgreen", linewidth=2)
            ax_c2.set_title("Heat Conduction Convergence", fontsize=12, fontweight="bold")
            ax_c2.set_xlabel("Iterations")
            ax_c2.set_ylabel("Residual Loss")
            ax_c2.set_yscale("log")
            ax_c2.legend(frameon=True)
            ax_c2.grid(True, linestyle=":", alpha=0.6)

        plt.tight_layout()
        fig13_file = out_path / f"fig13_14_engineering_convergence_{mode_suffix}.png"
        plt.savefig(fig13_file, dpi=200)
        plt.close()

        shutil.copy(fig13_file, fig_archive_path / f"fig13_14_engineering_convergence_{mode_suffix}.png")
        print(f"[+] Saved Figures 13 & 14 to: {fig13_file}")

    print(f"\nAll engineering figures are saved in: {out_path.resolve()}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Engineering Applications Experiment (Section 3.4, Fig. 8, 13, 14)"
    )
    parser.add_argument(
        "--app",
        "-a",
        type=str,
        default="all",
        choices=["all", "nozzle", "heat", "windfarm"],
        help="Target application: 'nozzle', 'heat', 'windfarm', or 'all' (default: all)",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        default=1,
        help="Number of random seeds (default: 1 for fast verification, 30 for full paper statistics)",
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

    run_experiment(
        app=args.app,
        num_seeds=args.seeds,
        use_mock=args.mock,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()
