"""Command-line interface for LEO optimization and experiment execution."""

from __future__ import annotations

import numpy as np
import typer
from rich.console import Console
from rich.table import Table

from leo.core.algorithm import LEOAlgorithm
from leo.llm.client import LLMClient, LLMSettings
from leo.problems import (
    BENCHMARKS_2D,
    HeatTransferProblem,
    NoseConeProblem,
    NozzleShapeProblem,
    ShiftedRosenbrockND,
    WindFarmProblem,
)

app = typer.Typer(
    name="leo",
    help="LEO: Large Language Model-Based Evolutionary Optimizer (Neurocomputing 622, 2025)",
    add_completion=False,
)
console = Console()


def get_problem(name: str, dim: int = 2):
    name_clean = name.lower().replace("-", "_")
    if name_clean in BENCHMARKS_2D:
        return BENCHMARKS_2D[name_clean]()  # type: ignore
    elif name_clean in ["rosenbrock_nd", "shifted_rosenbrock"]:
        return ShiftedRosenbrockND(dim=dim)
    elif name_clean in ["nozzle", "nozzle_shape"]:
        return NozzleShapeProblem()
    elif name_clean in ["heat", "heat_transfer"]:
        return HeatTransferProblem(dim=dim)
    elif name_clean in ["windfarm", "wind_farm"]:
        return WindFarmProblem(n_turbines=max(dim // 2, 2))
    elif name_clean in ["nosecone", "nose_cone"]:
        return NoseConeProblem()
    else:
        raise ValueError(
            f"Unknown problem '{name}'. Available: {list(BENCHMARKS_2D.keys())} + ['rosenbrock_nd', 'nozzle', 'heat_transfer', 'windfarm', 'nosecone']"
        )


@app.command()
def optimize(
    problem: str = typer.Option(
        "sphere2d", "--problem", "-p", help="Target problem name"
    ),
    dim: int = typer.Option(2, "--dim", "-d", help="Dimension for scalable problems"),
    pop_size: int = typer.Option(10, "--pop", help="Population size"),
    max_iters: int = typer.Option(
        30, "--iters", "-i", help="Maximum optimization iterations"
    ),
    mock: bool = typer.Option(
        True, "--mock/--live", help="Use deterministic mock LLM or live API"
    ),
    seed: int = typer.Option(42, "--seed", "-s", help="Random seed"),
    verbose: bool = typer.Option(
        True, "--verbose/--quiet", help="Display progress bar"
    ),
) -> None:
    """Run LEO optimization on a specific benchmark or engineering problem."""
    problem_clean = problem.lower().replace("-", "_")
    if problem_clean in ["zdt1", "zdt3"]:
        from pathlib import Path

        import matplotlib.pyplot as plt

        from leo.modular.nsga2 import LEOModularNSGA2, StandardNSGA2
        from leo.modular.plotting import plot_pareto_panel, setup_plot_style
        from leo.problems.multi_objective import ZDT1Problem, ZDT3Problem

        mo_prob = (
            ZDT1Problem(dim=dim) if problem_clean == "zdt1" else ZDT3Problem(dim=dim)
        )
        mo_pop = (
            pop_size
            if pop_size != 10 or problem_clean == "zdt1"
            else (30 if problem_clean == "zdt3" else 10)
        )
        mo_gens = max_iters if max_iters != 30 else 40

        settings = LLMSettings(leo_mock_llm=mock)
        client = LLMClient(settings=settings)

        console.print(
            f"[bold green]Starting LEO-modular optimization[/bold green] on [cyan]{mo_prob.name}[/cyan] (dim={mo_prob.dim})"
        )
        console.print(
            f"Settings: pop_size={mo_pop}, max_generations={mo_gens}, mock={mock}, seed={seed}"
        )

        leo_res = LEOModularNSGA2(
            mo_prob,
            pop_size=mo_pop,
            max_generations=mo_gens,
            llm_client=client,
            seed=seed,
        ).optimize()
        std_res = StandardNSGA2(
            mo_prob, pop_size=mo_pop, max_generations=mo_gens, seed=seed
        ).optimize()
        true_front = mo_prob.get_pareto_front(
            n_points=1000 if problem_clean == "zdt3" else 300
        )

        mode_suffix = "mock" if mock else "live"
        out_dir = Path("tmp_results")
        out_dir.mkdir(parents=True, exist_ok=True)
        img_path = out_dir / f"{problem_clean}_pareto_front_{mode_suffix}.png"

        setup_plot_style()
        _fig, ax = plt.subplots(figsize=(7, 6))
        plot_pareto_panel(
            ax,
            true_front,
            std_res.pareto_f,
            leo_res.pareto_f,
            title=f"test case: {mo_prob.name}",
            is_zdt3=(problem_clean == "zdt3"),
        )
        plt.tight_layout()
        plt.savefig(img_path, dpi=200)
        plt.close()

        table = Table(title=f"Multi-Objective Pareto Results: {mo_prob.name}")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green")
        table.add_row("Non-dominated Solutions Found", str(len(leo_res.pareto_f)))
        table.add_row("Generations", str(mo_gens))
        table.add_row("Output Image", str(img_path.resolve()))
        console.print(table)
        console.print(f"[bold green]Saved Pareto front plot to:[/bold green] [cyan]{img_path}[/cyan]")
        return

    prob = get_problem(problem, dim=dim)
    settings = LLMSettings(leo_mock_llm=mock)
    client = LLMClient(settings=settings)

    console.print(
        f"[bold green]Starting LEO optimization[/bold green] on [cyan]{prob.name}[/cyan] (dim={prob.dim})"
    )
    console.print(
        f"Settings: pop_size={pop_size}, max_iters={max_iters}, mock={mock}, seed={seed}"
    )

    algo = LEOAlgorithm(
        problem=prob,
        pop_size=pop_size,
        max_iters=max_iters,
        llm_client=client,
        seed=seed,
        verbose=verbose,
    )
    res = algo.optimize()

    table = Table(title=f"Optimization Results: {prob.name}")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green")
    table.add_row("Best Objective Value", f"{res.best_y:.6e}")
    if prob.global_min is not None:
        table.add_row("Known Global Min", f"{prob.global_min:.6e}")
        table.add_row("Absolute Gap", f"{abs(res.best_y - prob.global_min):.6e}")
    table.add_row("Total Evaluations", str(res.total_evaluations))
    table.add_row(
        "Best Coordinates",
        f"[{', '.join(f'{v:.4f}' for v in res.best_x[:6])}...]"
        if len(res.best_x) > 6
        else str(np.round(res.best_x, 4)),
    )

    console.print(table)

    # Save convergence trajectory plot to tmp_results/
    from pathlib import Path
    import matplotlib.pyplot as plt

    out_dir = Path("tmp_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    mode_suffix = "mock" if mock else "live"
    plot_name = (
        f"{problem_clean}_{dim}d_convergence_{mode_suffix}.png"
        if dim != 2
        else f"{problem_clean}_convergence_{mode_suffix}.png"
    )
    img_path = out_dir / plot_name

    plt.figure(figsize=(7, 4.5))
    plt.plot(
        res.history_exploit_best_y,
        label=f"LEO Exploit $f_{{min}}$ ({mode_suffix})",
        color="#0b409c",
        linewidth=2.0,
    )
    plt.xlabel("Optimization Iterations", fontsize=11)
    plt.ylabel("Best Function Value ($f_{min}$)", fontsize=11)
    plt.title(f"Convergence: {prob.name} (dim={prob.dim})", fontsize=12)
    plt.tick_params(direction="in", which="both", top=True, right=True)
    plt.grid(True, linestyle=":", alpha=0.6)
    hist_arr = np.array(res.history_exploit_best_y)
    if len(hist_arr) > 0 and np.all(hist_arr > 0):
        plt.yscale("log")
    plt.legend(frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(img_path, dpi=200)
    plt.close()

    console.print(f"[bold green]Saved convergence plot to:[/bold green] [cyan]{img_path}[/cyan]")

    # Problem-specific engineering visualizer for Section 3.4
    if problem_clean in ["nozzle", "nozzle_shape"] and hasattr(prob, "get_contour"):
        eng_img_path = out_dir / f"nozzle_shape_{mode_suffix}.png"
        xc, yc = prob.get_contour(res.best_x)
        plt.figure(figsize=(7, 4.5))
        plt.plot(xc, yc, color="#c92a2a", linewidth=2.5, label="Optimal Bell Contour")
        plt.title("LEO Optimal Supersonic Nozzle Contour (Fig. 8c)", fontsize=12)
        plt.xlabel("Longitudinal coordinate x", fontsize=11)
        plt.ylabel("Radial coordinate y", fontsize=11)
        plt.tick_params(direction="in", which="both", top=True, right=True)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True, fontsize=10)
        plt.tight_layout()
        plt.savefig(eng_img_path, dpi=200)
        plt.close()
        console.print(f"[bold green]Saved Nozzle shape contour to:[/bold green] [cyan]{eng_img_path}[/cyan]")
    elif problem_clean in ["heat", "heat_transfer"] and hasattr(prob, "exact_solution"):
        eng_img_path = out_dir / f"heat_{dim}d_profile_{mode_suffix}.png"
        gx = prob.grid_x
        exact_t = prob.exact_solution(gx)
        best_t_padded = np.zeros(dim + 2)
        best_t_padded[1:-1] = res.best_x
        best_t_padded[-1] = 1.0
        plt.figure(figsize=(7, 4.5))
        plt.plot(gx, exact_t, "k--", label="Exact Solution", linewidth=2.0)
        plt.plot(gx, best_t_padded, "o-", color="#0b409c", label=f"LEO (Dim={dim})", linewidth=1.8)
        plt.title(f"1D Steady-State Heat Profile (Dim={dim}, Fig. 8d-f)", fontsize=12)
        plt.xlabel("x / L", fontsize=11)
        plt.ylabel("Temperature T", fontsize=11)
        plt.tick_params(direction="in", which="both", top=True, right=True)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True, fontsize=10)
        plt.tight_layout()
        plt.savefig(eng_img_path, dpi=200)
        plt.close()
        console.print(f"[bold green]Saved Heat profile plot to:[/bold green] [cyan]{eng_img_path}[/cyan]")
    elif problem_clean in ["windfarm", "wind_farm"] and hasattr(prob, "n_turbines"):
        n_turbines = prob.n_turbines
        eng_img_path = out_dir / f"windfarm_{n_turbines}turbines_layout_{mode_suffix}.png"
        coords = res.best_x.reshape(n_turbines, 2)
        plt.figure(figsize=(6, 6))
        plt.scatter(coords[:, 0], coords[:, 1], color="#0b409c", s=100, edgecolors="black", label=f"LEO Turbines (N={n_turbines})")
        plt.xlim(0, prob.domain_size)
        plt.ylim(0, prob.domain_size)
        plt.title(f"Optimal Wind Turbine Layout (N={n_turbines}, Fig. 8g-i)", fontsize=12)
        plt.xlabel("X (m)", fontsize=11)
        plt.ylabel("Y (m)", fontsize=11)
        plt.tick_params(direction="in", which="both", top=True, right=True)
        plt.grid(True, linestyle=":", alpha=0.6)
        plt.legend(frameon=True, fontsize=10)
        plt.tight_layout()
        plt.savefig(eng_img_path, dpi=200)
        plt.close()
        console.print(f"[bold green]Saved Wind Farm layout plot to:[/bold green] [cyan]{eng_img_path}[/cyan]")


@app.command()
def experiment(
    name: str = typer.Argument(
        ..., help="Experiment to run: exp1 | exp2 | exp3 | exp4 | exp5 | all"
    ),
    mock: bool = typer.Option(True, "--mock/--live", help="Use mock LLM or live API"),
    seeds: int = typer.Option(5, "--seeds", "-s", help="Number of random seeds"),
) -> None:
    """Run paper reproduction experiments."""
    import subprocess
    import sys

    exp_map = {
        "exp1": "experiments/exp1_benchmarks_2d.py",
        "exp2": "experiments/exp2_multi_objective.py",
        "exp3": "experiments/exp3_high_dim.py",
        "exp4": "experiments/exp4_engineering.py",
        "exp5": "experiments/exp5_reasoning_rnd.py",
    }

    targets = list(exp_map.keys()) if name.lower() == "all" else [name.lower()]

    for target in targets:
        if target not in exp_map:
            console.print(
                f"[bold red]Unknown experiment '{target}'. Choices: {list(exp_map.keys())} or 'all'[/bold red]"
            )
            raise typer.Exit(code=1)

        script_path = exp_map[target]
        cmd = [sys.executable, script_path, "--seeds", str(seeds)]
        if mock:
            cmd.append("--mock")
        else:
            cmd.append("--live")

        console.print(f"\n[bold yellow]Executing {target}: {script_path}[/bold yellow]")
        subprocess.run(cmd, check=True)


def main() -> None:
    app()


if __name__ == "__main__":
    main()
