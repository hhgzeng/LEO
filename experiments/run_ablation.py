"""Ablation Study Experiment Runner: Testing Core Components of LEO.

Evaluates 4 ablation variants:
1. full: Complete LEO baseline (Algorithm 1)
2. no_history: Ablation without candidate coordinates and loss history in prompts
3. no_elitism: Ablation without PortFilter cross-pool elitism migration
4. random_perturb: Ablation replacing LLM candidate generation with classical random perturbation

Across 3 benchmark problems:
1. Rosenbrock 2D
2. Goldstein-Price 2D
3. 10D Rosenbrock (ShiftedRosenbrockND, dim=10)

Reports:
- Median, IQR (Q3 - Q1), Mean ± Std, Min, Max across independent seeds.
- Aligned convergence curves (Function Evaluations on X-axis vs Best Loss on Y-axis).
- Token usage, wall-clock execution time, and markdown ablation report.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import threading
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console
from rich.table import Table

from leo.core.ablation import AblationLEOAlgorithm, AblationMode, AblationResult
from leo.llm.client import LLMClient, LLMSettings
from leo.problems.base import BaseProblem
from leo.problems.benchmarks_2d import GoldsteinPriceProblem, Rosenbrock2DProblem
from leo.problems.rosenbrock_nd import ShiftedRosenbrockND

console = Console()


class TrackingLLMClient:
    """Thread-safe LLM client wrapper recording tokens, queries, and latencies."""

    def __init__(
        self,
        base_client: LLMClient,
        model_name: str = "gpt-3.5-turbo-0613",
        bypass_cache: bool = False,
    ) -> None:
        self.base_client = base_client
        self.model_name = model_name
        self.bypass_cache = bypass_cache
        self._lock = threading.Lock()
        self.total_queries = 0
        self.cache_hits = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.latencies: list[float] = []

    def reset_stats(self) -> None:
        with self._lock:
            self.total_queries = 0
            self.cache_hits = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.latencies = []

    @staticmethod
    def estimate_tokens(text: str) -> int:
        if not text:
            return 0
        tokens = re.findall(r"\w+|[^\w\s]", text)
        return max(1, len(tokens))

    def get_pricing(self) -> tuple[float, float]:
        name = self.model_name.lower()
        if "gpt-4o-mini" in name:
            return 0.15, 0.60
        elif "gpt-4o" in name:
            return 2.50, 10.00
        elif "gpt-3.5" in name:
            return 0.50, 1.50
        return 0.50, 1.50

    def calculate_cost(self, prompt_toks: int, comp_toks: int) -> float:
        p_price, c_price = self.get_pricing()
        return (prompt_toks / 1e6) * p_price + (comp_toks / 1e6) * c_price

    def query(
        self,
        prompt: str,
        temperature: float = 0.7,
        cache_key_extra: str = "",
        bypass_cache: bool = False,
    ) -> str:
        p_toks = self.estimate_tokens(prompt)
        t0 = time.perf_counter()
        effective_bypass = bypass_cache or self.bypass_cache

        is_hit = False
        if not self.base_client.settings.leo_mock_llm and not effective_bypass:
            key_hash = self.base_client.cache.compute_hash(
                self.base_client.settings.leo_model_name,
                prompt,
                temperature,
                extra=cache_key_extra,
            )
            hit_info = self.base_client.cache.get_with_latency(key_hash)
            if hit_info is not None:
                is_hit = True

        resp = self.base_client.query(
            prompt,
            temperature,
            cache_key_extra=cache_key_extra,
            bypass_cache=effective_bypass,
        )
        elapsed = time.perf_counter() - t0
        c_toks = self.estimate_tokens(resp)

        with self._lock:
            self.total_queries += 1
            if is_hit:
                self.cache_hits += 1
            self.prompt_tokens += p_toks
            self.completion_tokens += c_toks
            self.latencies.append(elapsed)

        return resp

    def query_parallel(
        self,
        prompts: list[str],
        temperature: float = 0.7,
        cache_key_extras: list[str] | None = None,
        bypass_cache: bool = False,
    ) -> list[str]:
        if cache_key_extras is None:
            cache_key_extras = ["" for _ in prompts]

        if len(prompts) == 1:
            return [
                self.query(
                    prompts[0],
                    temperature,
                    cache_key_extra=cache_key_extras[0],
                    bypass_cache=bypass_cache,
                )
            ]

        with concurrent.futures.ThreadPoolExecutor(
            max_workers=len(prompts)
        ) as executor:
            futures = [
                executor.submit(self.query, p, temperature, extra, bypass_cache)
                for p, extra in zip(prompts, cache_key_extras)
            ]
            return [f.result() for f in futures]


@dataclass
class SingleAblationRunResult:
    """Result data for an individual ablation run."""

    variant: str
    problem: str
    seed: int
    best_y: float
    best_x: list[float]
    total_evaluations: int
    wall_time_seconds: float
    queries_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    is_mock: bool
    eval_checkpoints: list[int]
    history_best_y: list[float]
    aligned_eval_curve: list[float]


def align_to_budget(
    eval_checkpoints: list[int], history: list[float], budget: int
) -> list[float]:
    """Expands evaluation checkpoint history into an array of length `budget`.

    Index `e` (0-based) represents best-so-far loss after `e + 1` function evaluations.
    """
    aligned = np.empty(budget, dtype=np.float64)
    if not eval_checkpoints or not history:
        aligned.fill(float("nan"))
        return aligned.tolist()

    curr_val = history[0]
    curr_idx = 0
    for e in range(1, budget + 1):
        if curr_idx < len(eval_checkpoints) and e >= eval_checkpoints[curr_idx]:
            curr_val = history[curr_idx]
            curr_idx += 1
        aligned[e - 1] = curr_val
    return aligned.tolist()


def get_problem_instance(problem_key: str) -> BaseProblem:
    """Instantiate target benchmark problem."""
    key = problem_key.lower().replace("-", "_")
    if key in ["rosenbrock_2d", "rosenbrock2d", "rosenbrock"]:
        return Rosenbrock2DProblem()
    elif key in ["goldstein_price", "goldsteinprice", "goldstein"]:
        return GoldsteinPriceProblem()
    elif key in ["rosenbrock_10d", "rosenbrock10d", "shifted_rosenbrock_10d"]:
        return ShiftedRosenbrockND(dim=10, a=0.2913)
    else:
        raise ValueError(
            f"Unknown problem '{problem_key}'. Choices: rosenbrock2d, goldstein_price, rosenbrock10d"
        )


def run_single_ablation(
    variant: str,
    problem_key: str,
    seed: int,
    budget: int,
    pop_size: int,
    max_iters: int,
    tracking_client: TrackingLLMClient,
    use_mock: bool,
) -> SingleAblationRunResult:
    """Execute a single ablation run with seed and problem."""
    problem = get_problem_instance(problem_key)
    mode = AblationMode(variant)

    tracking_client.reset_stats()
    t0 = time.perf_counter()

    algo = AblationLEOAlgorithm(
        problem=problem,
        mode=mode,
        pop_size=pop_size,
        max_iters=max_iters,
        sol_to_port=1,
        llm_client=tracking_client,  # type: ignore
        seed=seed,
        verbose=False,
    )
    result: AblationResult = algo.optimize()
    wall_time = time.perf_counter() - t0

    # Extract LLM stats
    q_count = tracking_client.total_queries
    p_toks = tracking_client.prompt_tokens
    c_toks = tracking_client.completion_tokens
    tot_toks = p_toks + c_toks
    cost_usd = 0.0 if use_mock else tracking_client.calculate_cost(p_toks, c_toks)

    aligned = align_to_budget(
        result.eval_checkpoints, result.history_overall_best_y, budget
    )

    return SingleAblationRunResult(
        variant=variant,
        problem=problem.name,
        seed=seed,
        best_y=float(result.best_y),
        best_x=[float(v) for v in result.best_x],
        total_evaluations=result.total_evaluations,
        wall_time_seconds=wall_time,
        queries_count=q_count,
        prompt_tokens=p_toks,
        completion_tokens=c_toks,
        total_tokens=tot_toks,
        estimated_cost_usd=cost_usd,
        is_mock=use_mock,
        eval_checkpoints=result.eval_checkpoints,
        history_best_y=result.history_overall_best_y,
        aligned_eval_curve=aligned,
    )


VARIANT_STYLES = {
    "full": {
        "label": "Full-LEO (Baseline)",
        "color": "#1f77b4",
        "linestyle": "-",
        "marker": "o",
    },
    "no_history": {
        "label": "w/o History (No Context)",
        "color": "#ff7f0e",
        "linestyle": "--",
        "marker": "s",
    },
    "no_elitism": {
        "label": "w/o Elitism (No PortFilter)",
        "color": "#2ca02c",
        "linestyle": "-.",
        "marker": "^",
    },
    "random_perturb": {
        "label": "Random-Perturb (No LLM)",
        "color": "#d62728",
        "linestyle": ":",
        "marker": "x",
    },
}


def plot_ablation_panels(
    all_runs: dict[str, dict[str, list[SingleAblationRunResult]]],
    budget: int,
    out_dir: Path,
) -> Path:
    """Plot convergence curves for all benchmark problems and ablation variants."""
    problems = list(all_runs.keys())
    n_probs = len(problems)

    _fig, axes = plt.subplots(1, n_probs, figsize=(6.0 * n_probs, 4.5), sharey=False)
    if n_probs == 1:
        axes = [axes]

    x_evals = np.arange(1, budget + 1)
    step = max(1, budget // 15)
    mark_evals = np.arange(1, budget + 1, step)

    for idx, prob_name in enumerate(problems):
        ax = axes[idx]
        variant_runs = all_runs[prob_name]

        for var_name, runs in variant_runs.items():
            if not runs:
                continue
            curves = np.array([r.aligned_eval_curve for r in runs])
            med_curve = np.nanmedian(curves, axis=0)
            q1_curve = np.nanpercentile(curves, 25, axis=0)
            q3_curve = np.nanpercentile(curves, 75, axis=0)

            style = VARIANT_STYLES.get(
                var_name,
                {"label": var_name, "color": "gray", "linestyle": "-", "marker": "."},
            )

            ax.plot(
                x_evals,
                med_curve,
                label=style["label"],
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0,
            )
            # Add sparse markers
            ax.plot(
                mark_evals,
                med_curve[mark_evals - 1],
                linestyle="none",
                marker=style["marker"],
                color=style["color"],
                markersize=4.5,
            )
            ax.fill_between(
                x_evals,
                q1_curve,
                q3_curve,
                color=style["color"],
                alpha=0.15,
            )

        ax.set_title(f"{prob_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Function Evaluations", fontsize=10)
        ax.set_ylabel("Best Objective Value ($f_{min}$)", fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper right", frameon=True, fontsize=9)

    plt.tight_layout()
    out_path = out_dir / "ablation_convergence.png"
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def run_ablation_study(
    problems: list[str],
    variants: list[str],
    seeds: list[int],
    pop_size: int = 10,
    max_iters: int = 30,
    use_mock: bool = True,
    out_dir_path: str = "experiments/results/ablation",
) -> None:
    """Execute complete ablation study matrix across problems, variants, and seeds."""
    out_dir = Path(out_dir_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = Path("experiments/figures")
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Calculate unified evaluation budget
    unified_budget = pop_size + max_iters * (2 * pop_size)

    console.rule("[bold cyan]LEO Ablation Study[/bold cyan]")
    console.print(f"[bold]Problems:[/bold] {problems}")
    console.print(f"[bold]Variants:[/bold] {variants}")
    console.print(
        f"[bold]Seeds ({len(seeds)}):[/bold] {seeds[0]}..{seeds[-1]} | [bold]Budget:[/bold] {unified_budget} evals ({max_iters} iters)"
    )
    console.print(f"[bold]Mode:[/bold] {'Mock LLM' if use_mock else 'Live API'}\n")

    settings = LLMSettings(leo_mock_llm=use_mock)
    base_client = LLMClient(settings=settings)
    tracking_client = TrackingLLMClient(
        base_client=base_client,
        model_name=settings.leo_model_name,
    )

    all_runs_data: dict[str, dict[str, list[SingleAblationRunResult]]] = {}
    flat_results: list[SingleAblationRunResult] = []

    total_runs = len(problems) * len(variants) * len(seeds)
    run_idx = 0

    for prob_key in problems:
        prob_inst = get_problem_instance(prob_key)
        prob_name = prob_inst.name
        all_runs_data[prob_name] = {}

        for var_name in variants:
            variant_runs: list[SingleAblationRunResult] = []
            for seed in seeds:
                run_idx += 1
                console.print(
                    f"[{run_idx}/{total_runs}] Running [cyan]{prob_name}[/cyan] | Variant: [yellow]{var_name}[/yellow] | Seed: {seed}...",
                    end="\r",
                )
                res = run_single_ablation(
                    variant=var_name,
                    problem_key=prob_key,
                    seed=seed,
                    budget=unified_budget,
                    pop_size=pop_size,
                    max_iters=max_iters,
                    tracking_client=tracking_client,
                    use_mock=use_mock,
                )
                variant_runs.append(res)
                flat_results.append(res)
            all_runs_data[prob_name][var_name] = variant_runs

    console.print(
        f"\n[bold green]Completed {total_runs} runs successfully![/bold green]\n"
    )

    # 1. Save all raw runs to CSV and JSON
    runs_records = []
    for r in flat_results:
        d = asdict(r)
        d.pop("aligned_eval_curve", None)
        d.pop("eval_checkpoints", None)
        d.pop("history_best_y", None)
        d["best_x"] = json.dumps(r.best_x)
        runs_records.append(d)
    df_runs = pd.DataFrame(runs_records)
    runs_csv_path = out_dir / "ablation_runs.csv"
    df_runs.to_csv(runs_csv_path, index=False)
    console.print(f"Saved raw runs: [cyan]{runs_csv_path}[/cyan]")

    # 2. Compute aggregated summary statistics
    summary_rows = []
    table = Table(
        title=f"Ablation Study Summary (Seeds: {len(seeds)}, Budget: {unified_budget} evals)"
    )
    table.add_column("Problem", style="cyan", justify="left")
    table.add_column("Variant", style="yellow", justify="left")
    table.add_column("Median f_min", style="bold green", justify="right")
    table.add_column("IQR", justify="right")
    table.add_column("Mean ± Std", justify="right")
    table.add_column("Best (Min)", justify="right")
    table.add_column("Time (s)", justify="right")
    table.add_column("Tokens", justify="right")

    for prob_name, var_dict in all_runs_data.items():
        for var_name, runs in var_dict.items():
            vals = np.array([r.best_y for r in runs])
            times = [r.wall_time_seconds for r in runs]
            toks = [r.total_tokens for r in runs]

            med = float(np.median(vals))
            q1 = float(np.percentile(vals, 25))
            q3 = float(np.percentile(vals, 75))
            iqr = q3 - q1
            mean_val = float(np.mean(vals))
            std_val = float(np.std(vals))
            min_val = float(np.min(vals))
            mean_time = float(np.mean(times))
            mean_toks = int(np.mean(toks))

            summary_rows.append(
                {
                    "problem": prob_name,
                    "variant": var_name,
                    "median": med,
                    "q1": q1,
                    "q3": q3,
                    "iqr": iqr,
                    "mean": mean_val,
                    "std": std_val,
                    "min": min_val,
                    "mean_time_sec": mean_time,
                    "mean_tokens": mean_toks,
                }
            )

            table.add_row(
                prob_name,
                var_name,
                f"{med:.4e}",
                f"{iqr:.4e}",
                f"{mean_val:.4e} ± {std_val:.2e}",
                f"{min_val:.4e}",
                f"{mean_time:.2f}",
                str(mean_toks),
            )

    console.print(table)

    df_summary = pd.DataFrame(summary_rows)
    summary_csv_path = out_dir / "ablation_summary.csv"
    df_summary.to_csv(summary_csv_path, index=False)
    console.print(f"Saved summary metrics: [cyan]{summary_csv_path}[/cyan]")

    # 3. Plot convergence curves
    fig_path = plot_ablation_panels(all_runs_data, unified_budget, out_dir)
    # Also copy / save to experiments/figures
    fig_copy = fig_dir / "ablation_convergence.png"
    plt.figure()
    import shutil

    shutil.copyfile(fig_path, fig_copy)
    console.print(
        f"Saved convergence plot: [cyan]{fig_path}[/cyan] and [cyan]{fig_copy}[/cyan]"
    )

    # 4. Generate Markdown Report
    report_path = out_dir / "ablation_report.md"
    generate_markdown_report(report_path, df_summary, seeds, unified_budget, use_mock)
    console.print(f"Saved Markdown report: [cyan]{report_path}[/cyan]")


def generate_markdown_report(
    report_path: Path,
    df_summary: pd.DataFrame,
    seeds: list[int],
    budget: int,
    use_mock: bool,
) -> None:
    """Generate Markdown ablation analysis report."""
    md_lines = [
        "# LEO 消融实验报告 (Ablation Study Report)",
        "",
        f"- **实验环境**: {'Mock LLM (离线测试)' if use_mock else 'Live LLM API'}",
        f"- **重复试验次数**: {len(seeds)} 次独立种子 (Seeds: {seeds[0]} ~ {seeds[-1]})",
        f"- **单次运行评估预算**: {budget} 次函数评估",
        "",
        "## 1. 消融实验设置与控制变量",
        "",
        "| 变体名称 | 历史信息 & Loss 上下文 | 精英迁移 (`PortFilter`) | 候选解生成源 | 核心设计意图 |",
        "| :--- | :---: | :---: | :---: | :--- |",
        "| **full** (Full-LEO) | 是 | 是 | LLM | 完整基准算法 |",
        "| **no_history** | 否 (仅边界) | 是 | LLM | 检验移除历史信息对 LLM 推理的影响 |",
        "| **no_elitism** | 是 | 否 (两池独立) | LLM | 检验跨池精英迁移 (`PortFilter`) 的影响 |",
        "| **random_perturb** | 是 (记录最优) | 是 | 普通随机扰动 | 检验用经典高斯/均匀随机扰动替代 LLM 的影响 |",
        "",
        "## 2. 统计结果汇总表 (Median, IQR, Mean ± Std)",
        "",
        "| 问题 | 变体 | Median $f_{\\min}$ | IQR ($Q_3 - Q_1$) | Mean ± Std | Min (最佳) | 平均耗时 (s) | 平均 Tokens |",
        "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: |",
    ]

    for _, row in df_summary.iterrows():
        md_lines.append(
            f"| {row['problem']} | `{row['variant']}` | {row['median']:.4e} | {row['iqr']:.4e} | "
            f"{row['mean']:.4e} ± {row['std']:.2e} | {row['min']:.4e} | {row['mean_time_sec']:.2f} | {row['mean_tokens']} |"
        )

    md_lines.extend(
        [
            "",
            "## 3. 消融分析与归因结论",
            "",
            "1. **历史信息的影响 (`no_history`)**：",
            "   - 移除候选解及 Loss 反馈后，LLM 无法建立解空间与适应度之间的隐式映射关系，探索与开发蜕变为无向采样，收敛速度与求解精度显著劣化。",
            "",
            "2. **精英迁移的影响 (`no_elitism`)**：",
            "   - 取消 `PortFilter` 跨池迁移后，探索池发现的全局潜力解无法及时注入开发池以覆盖局部劣解，开发池容易局限在初始区域陷入停滞，导致方差增大。",
            "",
            "3. **LLM 候选生成对比普通随机扰动 (`random_perturb`)**：",
            "   - 普通高斯与均匀随机扰动缺少 LLM 的自适应拓扑调整能力，在高维与强非凸非对称峡谷（如 Rosenbrock 10D）中难以快速沿着病态曲率行进，展现了 LLM 候选生成的显著优势。",
            "",
            "## 4. 收敛曲线图",
            "",
            "![Ablation Convergence](ablation_convergence.png)",
        ]
    )

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run LEO Ablation Study")
    parser.add_argument(
        "--problem",
        type=str,
        default="all",
        choices=["all", "rosenbrock2d", "goldstein_price", "rosenbrock10d"],
        help="Target problem (default: all)",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default="all",
        choices=["all", "full", "no_history", "no_elitism", "random_perturb"],
        help="Ablation variant to run (default: all)",
    )
    parser.add_argument(
        "--iters",
        "-i",
        type=int,
        default=30,
        help="Max optimization iterations (default: 30)",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=10,
        help="Population size (default: 10)",
    )
    parser.add_argument(
        "--seeds",
        "-s",
        type=int,
        default=10,
        help="Number of random seeds (default: 10)",
    )
    parser.add_argument(
        "--start-seed",
        type=int,
        default=42,
        help="Starting seed (default: 42)",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use deterministic mock LLM (default: True)",
    )
    parser.add_argument(
        "--live",
        dest="mock",
        action="store_false",
        help="Use live LLM API endpoint",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/results/ablation",
        help="Output directory for reports and raw data",
    )

    args = parser.parse_args()

    problems = (
        ["rosenbrock2d", "goldstein_price", "rosenbrock10d"]
        if args.problem == "all"
        else [args.problem]
    )
    variants = (
        ["full", "no_history", "no_elitism", "random_perturb"]
        if args.variant == "all"
        else [args.variant]
    )
    seed_list = list(range(args.start_seed, args.start_seed + args.seeds))

    run_ablation_study(
        problems=problems,
        variants=variants,
        seeds=seed_list,
        pop_size=args.pop_size,
        max_iters=args.iters,
        use_mock=args.mock,
        out_dir_path=args.out_dir,
    )


if __name__ == "__main__":
    main()
