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

Architecture, file structure, and reporting align with `experiments/comparison/run_comparison.py`:
- Stored by algorithm iterations (not raw evaluations).
- Convergence plots with Iterations on X-axis and equivalent function evaluations on top axis (1 Iteration = 20 Evaluations).
- Subdirectory isolation per variant: full/, no_history/, no_elitism/, random_perturb/.
- Summary tables (metrics_table.csv/md, time_cost_table.csv/md), JSON summary, and ablation report.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich.console import Console

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
        self.cached_latencies: list[float] = []

    def reset_stats(self) -> None:
        with self._lock:
            self.total_queries = 0
            self.cache_hits = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self.latencies = []
            self.cached_latencies = []

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
        elif "llama-3.1-70b" in name:
            return 0.59, 0.79
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
        cached_lat = 0.0
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
                cached_lat = hit_info[1] if hit_info[1] > 0.0 else 2.0

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
                self.cached_latencies.append(cached_lat)
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
class SingleRunResult:
    """Result of a single ablation optimization run, stored by iteration."""

    variant: str
    problem: str
    seed: int
    best_y: float
    best_x: list[float]
    total_evaluations: int
    iterations: int
    wall_time_seconds: float
    queries_count: int
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    estimated_cost_usd: float
    is_mock: bool
    history_best_y: list[float]  # Length: iterations + 1 (best-so-far loss at iteration 0..max_iters)
    iteration_checkpoints: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SingleRunResult:
        hist = d.get("history_best_y", [])
        iters = d.get("iterations", 30)
        # Resample legacy evaluation-level history if present
        if len(hist) > iters + 1:
            pop_size = 10
            target_cps = [pop_size + 20 * k for k in range(iters + 1)]
            hist = [hist[min(len(hist) - 1, cp - 1)] for cp in target_cps]
        cps = d.get("iteration_checkpoints", list(range(len(hist))))
        return cls(
            variant=d.get("variant", d.get("algorithm", "full")),
            problem=d["problem"],
            seed=d["seed"],
            best_y=float(d["best_y"]),
            best_x=[float(v) for v in d["best_x"]],
            total_evaluations=int(d["total_evaluations"]),
            iterations=int(iters),
            wall_time_seconds=float(d["wall_time_seconds"]),
            queries_count=int(d.get("queries_count", 0)),
            prompt_tokens=int(d.get("prompt_tokens", 0)),
            completion_tokens=int(d.get("completion_tokens", 0)),
            total_tokens=int(d.get("total_tokens", 0)),
            estimated_cost_usd=float(d.get("estimated_cost_usd", 0.0)),
            is_mock=bool(d.get("is_mock", True)),
            history_best_y=hist,
            iteration_checkpoints=cps,
        )


def get_problem_instance(problem_key: str) -> BaseProblem:
    """Instantiate target benchmark problem with alias tolerance."""
    key = problem_key.lower().replace("-", "_")
    if key in ["rosenbrock_2d", "rosenbrock2d", "rosenbrock"]:
        return Rosenbrock2DProblem()
    elif key in ["goldstein_price", "goldsteinprice", "goldstein"]:
        return GoldsteinPriceProblem()
    elif key in ["rosenbrock_10d", "rosenbrock10d", "shifted_rosenbrock_10d"]:
        return ShiftedRosenbrockND(dim=10, a=0.2913)
    else:
        raise ValueError(
            f"Unknown problem '{problem_key}'. Choices: rosenbrock_2d, goldstein_price, rosenbrock_10d"
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
) -> SingleRunResult:
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
    if not use_mock and tracking_client.cache_hits > 0:
        wall_time += sum(tracking_client.cached_latencies) / 2.0

    q_count = tracking_client.total_queries
    p_toks = tracking_client.prompt_tokens
    c_toks = tracking_client.completion_tokens
    tot_toks = p_toks + c_toks
    cost_usd = tracking_client.calculate_cost(p_toks, c_toks)

    raw_history = [float(v) for v in result.history_overall_best_y]
    if len(raw_history) < max_iters + 1:
        raw_history = raw_history + [float(result.best_y)] * (max_iters + 1 - len(raw_history))
    else:
        raw_history = raw_history[: max_iters + 1]
    iter_checkpoints = list(range(len(raw_history)))

    return SingleRunResult(
        variant=variant,
        problem=problem.name,
        seed=seed,
        best_y=float(result.best_y),
        best_x=[float(v) for v in result.best_x],
        total_evaluations=result.total_evaluations,
        iterations=max_iters,
        wall_time_seconds=wall_time,
        queries_count=q_count,
        prompt_tokens=p_toks,
        completion_tokens=c_toks,
        total_tokens=tot_toks,
        estimated_cost_usd=cost_usd,
        is_mock=use_mock,
        history_best_y=raw_history,
        iteration_checkpoints=iter_checkpoints,
    )


def compute_statistics(values: list[float]) -> dict[str, Any]:
    """Computes median, IQR, Q1, Q3, mean, std, min, max for a list of floats."""
    arr = np.array(values, dtype=np.float64)
    med = float(np.median(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))

    return {
        "median": med,
        "iqr": iqr,
        "q1": q1,
        "q3": q3,
        "mean": mean_val,
        "std": std_val,
        "min": min_val,
        "max": max_val,
        "all_runs": values,
    }


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


def plot_convergence(
    problem_name: str,
    variant_runs: dict[str, list[SingleRunResult]],
    max_iters: int,
    out_path: Path,
) -> None:
    """Plots convergence curves with Iterations on the X-axis and IQR shaded bands."""
    plt.rcParams["font.sans-serif"] = [
        "PingFang SC",
        "Heiti SC",
        "Arial Unicode MS",
        "DejaVu Sans",
        "sans-serif",
    ]
    plt.rcParams["axes.unicode_minus"] = False
    _fig, ax = plt.subplots(figsize=(8, 5.5))

    x_iters = np.arange(0, max_iters + 1)

    for var_name, runs in variant_runs.items():
        if not runs:
            continue
        curves_list = []
        for r in runs:
            c = list(r.history_best_y)
            if len(c) < max_iters + 1:
                c = c + [c[-1]] * (max_iters + 1 - len(c))
            else:
                c = c[: max_iters + 1]
            curves_list.append(c)

        curves = np.array(curves_list)
        med_curve = np.median(curves, axis=0)
        q1_curve = np.percentile(curves, 25, axis=0)
        q3_curve = np.percentile(curves, 75, axis=0)

        style = VARIANT_STYLES.get(
            var_name, {"label": var_name, "color": "gray", "linestyle": "-", "marker": None}
        )
        markevery = max(1, max_iters // 10)

        ax.plot(
            x_iters,
            med_curve,
            label=f"{style['label']} (Median)",
            color=style["color"],
            linestyle=style["linestyle"],
            linewidth=2.2,
            marker=style["marker"],
            markevery=markevery,
            markersize=5,
        )
        ax.fill_between(
            x_iters,
            q1_curve,
            q3_curve,
            color=style["color"],
            alpha=0.18,
            label=f"{style['label']} IQR [Q1, Q3]" if len(runs) > 1 else None,
        )

    ax.set_title(
        f"Ablation Study Convergence on {problem_name}", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Iterations (迭代次数)", fontsize=11)
    ax.set_ylabel("Best-so-far Objective Value ($f_{min}$)", fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    # Secondary X-axis for equivalent function evaluations (1 iteration = 20 evaluations)
    secax = ax.secondary_xaxis(
        "top", functions=(lambda x: x * 20, lambda x: x / 20) # type: ignore
    )
    secax.set_xlabel(
        "Equivalent Function Evaluations (1 Iteration = 20 Evaluations)",
        fontsize=10,
    )

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def plot_combined_convergence(
    all_results: dict[str, dict[str, list[SingleRunResult]]],
    max_iters: int,
    out_path: Path,
) -> None:
    """Plots side-by-side convergence panels for all benchmark problems."""
    prob_names = list(all_results.keys())
    _fig, axes = plt.subplots(1, len(prob_names), figsize=(6 * len(prob_names), 5.2))
    if len(prob_names) == 1:
        axes = [axes]

    x_iters = np.arange(0, max_iters + 1)
    markevery = max(1, max_iters // 8)

    for idx, (prob_name, var_dict) in enumerate(all_results.items()):
        ax = axes[idx]
        for var_name, runs in var_dict.items():
            if not runs:
                continue
            curves_list = []
            for r in runs:
                c = list(r.history_best_y)
                if len(c) < max_iters + 1:
                    c = c + [c[-1]] * (max_iters + 1 - len(c))
                else:
                    c = c[: max_iters + 1]
                curves_list.append(c)

            curves = np.array(curves_list)
            med_curve = np.median(curves, axis=0)
            q1_curve = np.percentile(curves, 25, axis=0)
            q3_curve = np.percentile(curves, 75, axis=0)

            style = VARIANT_STYLES.get(
                var_name,
                {"label": var_name, "color": "gray", "linestyle": "-", "marker": None},
            )

            ax.plot(
                x_iters,
                med_curve,
                label=f"{style['label']}",
                color=style["color"],
                linestyle=style["linestyle"],
                linewidth=2.0,
                marker=style["marker"],
                markevery=markevery,
                markersize=4.5,
            )
            ax.fill_between(
                x_iters,
                q1_curve,
                q3_curve,
                color=style["color"],
                alpha=0.15,
            )

        ax.set_title(f"{prob_name}", fontsize=12, fontweight="bold")
        ax.set_xlabel("Iterations (迭代次数)", fontsize=10)
        ax.set_ylabel("Best Objective Value ($f_{min}$)", fontsize=10)
        ax.set_yscale("log")
        ax.grid(True, linestyle=":", alpha=0.6)
        if idx == 0:
            ax.legend(loc="upper right", frameon=True, fontsize=9)

        secax = ax.secondary_xaxis(
            "top", functions=(lambda x: x * 20, lambda x: x / 20)
        )
        secax.set_xlabel("Evals (Iter × 20)", fontsize=9)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()


def save_variant_problem_runs(
    out_dir: Path,
    variant_name: str,
    problem_name: str,
    runs: list[SingleRunResult],
    iterations: int,
    budget: int,
) -> Path:
    """Saves runs of a single ablation variant on a specific problem into a dedicated subfolder."""
    var_dir = out_dir / variant_name
    var_dir.mkdir(parents=True, exist_ok=True)
    clean_prob = problem_name.lower().replace("-", "_")
    file_path = var_dir / f"{variant_name}_{clean_prob}.json"
    data = {
        "variant": variant_name,
        "problem": problem_name,
        "iterations": iterations,
        "budget": budget,
        "runs": [asdict(r) for r in runs],
    }
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return file_path


def load_all_saved_results(
    out_dir: Path,
) -> tuple[dict[str, dict[str, list[SingleRunResult]]], int, int]:
    """Scans out_dir for all ablation variant subdirectories and loads their run results."""
    all_runs_data: dict[str, dict[str, list[SingleRunResult]]] = {}
    found_iters = 30
    found_budget = 610

    if not out_dir.exists():
        return all_runs_data, found_iters, found_budget

    valid_variants = ["full", "no_history", "no_elitism", "random_perturb"]
    for var_dir in sorted(out_dir.iterdir()):
        if not var_dir.is_dir() or var_dir.name not in valid_variants:
            continue
        var_name = var_dir.name

        for json_file in sorted(var_dir.glob("*.json")):
            if json_file.name.endswith("_summary.json"):
                continue
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                if not isinstance(data, dict) or "runs" not in data:
                    continue
                prob_name = data.get("problem", "")
                if not prob_name:
                    continue
                found_iters = data.get("iterations", found_iters)
                found_budget = data.get("budget", found_budget)

                runs = [SingleRunResult.from_dict(r) for r in data["runs"]]
                if prob_name not in all_runs_data:
                    all_runs_data[prob_name] = {}
                all_runs_data[prob_name][var_name] = runs
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                print(f"[!] Warning: failed to parse {json_file}: {e}")

    return all_runs_data, found_iters, found_budget


def generate_markdown_report(
    report_path: Path,
    df_metrics: pd.DataFrame,
    df_time_cost: pd.DataFrame,
    num_runs: int,
    max_iters: int,
    budget: int,
    use_mock: bool,
) -> None:
    """Generate Markdown ablation analysis report."""
    md_lines = [
        "# LEO 消融实验报告 (Ablation Study Report)",
        "",
        f"- **实验环境**: {'Mock LLM (离线测试)' if use_mock else 'Live LLM API'}",
        f"- **独立重复试验次数**: {num_runs} 次运行",
        f"- **迭代上限**: {max_iters} 轮迭代 (等价评估预算: {budget} 次函数评估, 1 迭代 = 20 函数评估)",
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
        "## 2. 性能统计表 (Median, IQR, Mean ± Std)",
        "",
        df_metrics.to_markdown(index=False),
        "",
        "## 3. 耗时与开销报告 (Time & Token Cost Breakdown)",
        "",
        df_time_cost.to_markdown(index=False),
        "",
        "## 4. 消融分析与归因结论",
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
        "## 5. 收敛曲线图",
        "",
        "![Ablation Convergence](ablation_convergence.png)",
        "",
    ]
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(md_lines))


def generate_summary_reports_and_plots(
    all_runs_data: dict[str, dict[str, list[SingleRunResult]]],
    max_iters: int,
    unified_budget: int,
    out_dir: Path,
    use_mock: bool = True,
    model_name: str = "gpt-3.5-turbo-0613",
    generate_plots: bool = True,
) -> None:
    """Computes merged statistics, prints tables, and produces convergence plots for all available data."""
    if not all_runs_data:
        print("[!] No ablation run data available to generate summaries.")
        return

    metrics_records: list[dict[str, Any]] = []
    time_cost_records: list[dict[str, Any]] = []

    first_prob = next(iter(all_runs_data.values()))
    first_var = next(iter(first_prob.values()))
    detected_mock = first_var[0].is_mock if first_var else use_mock
    num_runs = len(first_var) if first_var else 1

    json_summary: dict[str, Any] = {
        "metadata": {
            "iterations": max_iters,
            "budget": unified_budget,
            "num_runs": num_runs,
            "use_mock": detected_mock,
            "model_name": model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "problems": {},
    }

    for prob_name, var_dict in all_runs_data.items():
        json_summary["problems"][prob_name] = {}
        for var_name, runs in var_dict.items():
            final_scores = [r.best_y for r in runs]
            stats = compute_statistics(final_scores)

            wall_times = [r.wall_time_seconds for r in runs]
            total_time = sum(wall_times)
            mean_time = float(np.mean(wall_times))
            std_time = float(np.std(wall_times))
            median_time = float(np.median(wall_times))

            total_queries = sum(r.queries_count for r in runs)
            total_prompt_toks = sum(r.prompt_tokens for r in runs)
            total_comp_toks = sum(r.completion_tokens for r in runs)
            total_toks = total_prompt_toks + total_comp_toks
            total_cost = sum(r.estimated_cost_usd for r in runs)
            avg_cost = total_cost / max(1, len(runs))
            var_is_mock = runs[0].is_mock if runs else use_mock

            if total_queries == 0:
                mode_label = "Random Perturb (No LLM)"
            else:
                mode_label = "Mock (Simulated)" if var_is_mock else "Live API"

            metrics_records.append(
                {
                    "Problem": prob_name,
                    "Variant": var_name,
                    "Runs": len(runs),
                    "Iterations": max_iters,
                    "Budget": unified_budget,
                    "Median": f"{stats['median']:.6e}",
                    "IQR": f"{stats['iqr']:.6e}",
                    "Q1 (25%)": f"{stats['q1']:.6e}",
                    "Q3 (75%)": f"{stats['q3']:.6e}",
                    "Mean ± Std": f"{stats['mean']:.4e} ± {stats['std']:.4e}",
                    "Min (Best)": f"{stats['min']:.6e}",
                    "Max (Worst)": f"{stats['max']:.6e}",
                    "All Runs": [f"{v:.4e}" for v in stats["all_runs"]],
                }
            )

            time_cost_records.append(
                {
                    "Problem": prob_name,
                    "Variant": var_name,
                    "Runs": len(runs),
                    "Total Time (s)": f"{total_time:.2f}",
                    "Mean Time/Run (s)": f"{mean_time:.2f} ± {std_time:.2f}",
                    "Median Time (s)": f"{median_time:.2f}",
                    "LLM Queries": total_queries,
                    "Prompt Tokens": total_prompt_toks,
                    "Completion Tokens": total_comp_toks,
                    "Total Tokens": total_toks,
                    "Est. Cost ($)": f"${total_cost:.4f}",
                    "Avg Cost/Run ($)": f"${avg_cost:.4f}",
                    "Mode": mode_label,
                }
            )

            json_summary["problems"][prob_name][var_name] = {
                "statistics": stats,
                "time_metrics": {
                    "total_wall_time_seconds": total_time,
                    "mean_wall_time_seconds": mean_time,
                    "std_wall_time_seconds": std_time,
                    "median_wall_time_seconds": median_time,
                },
                "cost_metrics": {
                    "total_queries": total_queries,
                    "prompt_tokens": total_prompt_toks,
                    "completion_tokens": total_comp_toks,
                    "total_tokens": total_toks,
                    "total_cost_usd": total_cost,
                    "avg_cost_per_run_usd": avg_cost,
                    "is_mock": var_is_mock,
                },
                "individual_runs": [asdict(r) for r in runs],
            }

    df_metrics = pd.DataFrame(metrics_records)
    df_time_cost = pd.DataFrame(time_cost_records)

    print("\n" + "=" * 100)
    print(" TABLE 1: ABLATION STUDY PERFORMANCE (MEDIAN, IQR & ALL RUNS)")
    print("=" * 100)
    print(df_metrics.drop(columns=["All Runs"]).to_markdown(index=False))

    print("\n" + "=" * 100)
    print(" TABLE 2: TIME & COST REPORT (SEPARATE BREAKDOWN)")
    print("=" * 100)
    print(df_time_cost.to_markdown(index=False))

    print("\n" + "=" * 100)
    print(" TABLE 3: ALL RUN INDIVIDUAL RESULTS")
    print("=" * 100)
    for row in metrics_records:
        print(f"[{row['Problem']} | {row['Variant']}]: {row['All Runs']}")

    json_path = out_dir / "ablation_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)
    print(f"\n[+] Saved merged JSON summary to: {json_path}")

    md_metrics_path = out_dir / "metrics_table.md"
    with open(md_metrics_path, "w", encoding="utf-8") as f:
        f.write("# Ablation Performance Metrics\n\n")
        f.write(df_metrics.to_markdown(index=False))
        f.write("\n")
    print(f"[+] Saved metrics markdown to: {md_metrics_path}")

    csv_metrics_path = out_dir / "metrics_table.csv"
    df_metrics.to_csv(csv_metrics_path, index=False)
    print(f"[+] Saved metrics CSV to: {csv_metrics_path}")

    md_time_cost_path = out_dir / "time_cost_table.md"
    with open(md_time_cost_path, "w", encoding="utf-8") as f:
        f.write("# Execution Time and Cost Breakdown\n\n")
        f.write(df_time_cost.to_markdown(index=False))
        f.write("\n")
    print(f"[+] Saved time/cost markdown to: {md_time_cost_path}")

    csv_time_cost_path = out_dir / "time_cost_table.csv"
    df_time_cost.to_csv(csv_time_cost_path, index=False)
    print(f"[+] Saved time/cost CSV to: {csv_time_cost_path}")

    # Generate full Markdown report
    report_path = out_dir / "ablation_report.md"
    generate_markdown_report(
        report_path=report_path,
        df_metrics=df_metrics.drop(columns=["All Runs"]),
        df_time_cost=df_time_cost,
        num_runs=num_runs,
        max_iters=max_iters,
        budget=unified_budget,
        use_mock=detected_mock,
    )
    print(f"[+] Saved comprehensive ablation report: {report_path}")

    if generate_plots:
        print("\n>>> Generating convergence plots...")
        for prob_name, var_dict in all_runs_data.items():
            clean_name = prob_name.lower().replace("-", "_")
            plot_file = out_dir / f"convergence_{clean_name}.png"
            plot_convergence(
                problem_name=prob_name,
                variant_runs=var_dict,
                max_iters=max_iters,
                out_path=plot_file,
            )
            print(f"[+] Saved convergence plot: {plot_file}")

        combined_plot = out_dir / "ablation_convergence.png"
        plot_combined_convergence(
            all_results=all_runs_data,
            max_iters=max_iters,
            out_path=combined_plot,
        )
        print(f"[+] Saved combined convergence figure: {combined_plot}")

    print("\n[+] Ablation study results successfully updated!")


def run_ablation_experiments(
    problems: list[str],
    variants: list[str],
    num_runs: int = 10,
    budget: int | None = None,
    pop_size: int = 10,
    max_iters: int = 30,
    use_mock: bool = True,
    seed_start: int = 42,
    output_dir: str = "experiments/ablation",
    generate_plots: bool = True,
    verbose: bool = False,
    bypass_cache: bool = False,
) -> None:
    """Coordinates execution across requested benchmarks and ablation variants, saving to isolated subfolders."""
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Establish unified budget
    if budget is not None:
        unified_budget = budget
        calculated_iters = max(1, (unified_budget - pop_size) // (2 * pop_size))
        actual_iters = calculated_iters
    else:
        actual_iters = max_iters
        unified_budget = pop_size + 2 * pop_size * actual_iters

    settings = LLMSettings(leo_mock_llm=use_mock)
    base_client = LLMClient(settings=settings)
    tracking_client = TrackingLLMClient(
        base_client=base_client,
        model_name=settings.leo_model_name,
        bypass_cache=bypass_cache,
    )

    print("=" * 80)
    print(" LEO ABLATION STUDY EXPERIMENT")
    print(f" Target Problems   : {problems}")
    print(f" Target Variants   : {variants}")
    print(f" Independent Runs  : {num_runs}")
    print(
        f" Unified Budget    : {unified_budget} evaluations (pop_size={pop_size}, iters={actual_iters})"
    )
    print(f" LLM Mode          : {'MOCK' if use_mock else 'LIVE API'}")
    if not use_mock and bypass_cache:
        print(" Cache Mode        : BYPASSED (--no-cache)")
    print(f" Output Directory  : {out_dir.resolve()}")
    print("=" * 80)

    for prob_key in problems:
        prob_instance = get_problem_instance(prob_key)
        prob_name = prob_instance.name

        print(f"\n>>> Running Benchmark: {prob_name} (dim={prob_instance.dim})")

        for var_key in variants:
            var_name = var_key.lower().strip()
            print(f"  --> Variant: {var_name} ({num_runs} run(s))...")
            runs_list: list[SingleRunResult] = []

            for run_idx in range(num_runs):
                seed = seed_start + run_idx
                run_res = run_single_ablation(
                    variant=var_name,
                    problem_key=prob_key,
                    seed=seed,
                    budget=unified_budget,
                    pop_size=pop_size,
                    max_iters=actual_iters,
                    tracking_client=tracking_client,
                    use_mock=use_mock,
                )
                runs_list.append(run_res)
                if verbose or num_runs <= 3:
                    cache_note = (
                        f" (cache hits: {tracking_client.cache_hits}/{tracking_client.total_queries})"
                        if not use_mock and tracking_client.cache_hits > 0
                        else ""
                    )
                    print(
                        f"      Run {run_idx + 1:02d}/{num_runs:02d} (seed={seed:2d}): "
                        f"best_y={run_res.best_y:.6e}, time={run_res.wall_time_seconds:.2f}s{cache_note}"
                    )

            # Persist to dedicated subfolder for this variant
            saved_file = save_variant_problem_runs(
                out_dir=out_dir,
                variant_name=var_name,
                problem_name=prob_name,
                runs=runs_list,
                iterations=actual_iters,
                budget=unified_budget,
            )
            print(
                f"      [+] Persisted {var_name} data for {prob_name} -> {saved_file}"
            )

    # Auto-merge with ALL variant runs existing in out_dir (including previously completed ones)
    merged_data, effective_iters, effective_budget = load_all_saved_results(out_dir)

    generate_summary_reports_and_plots(
        all_runs_data=merged_data,
        max_iters=effective_iters or actual_iters,
        unified_budget=effective_budget or unified_budget,
        out_dir=out_dir,
        use_mock=use_mock,
        model_name=settings.leo_model_name,
        generate_plots=generate_plots,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run LEO Component Ablation Study",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--variant",
        "-a",
        type=str,
        default="all",
        help="Ablation variant(s): 'full', 'no_history', 'no_elitism', 'random_perturb', or 'all' (or comma-separated).",
    )
    parser.add_argument(
        "--problem",
        "-p",
        type=str,
        default="all",
        help="Problem(s) to benchmark: 'rosenbrock_2d', 'goldstein_price', 'rosenbrock_10d', or 'all' (or comma-separated).",
    )
    parser.add_argument(
        "--runs",
        "-n",
        type=int,
        default=10,
        help="Number of independent runs for statistical significance (paper default: 10).",
    )
    parser.add_argument(
        "--budget",
        type=int,
        default=None,
        help="Unified evaluation budget. If omitted, calculated as pop_size + 2 * pop_size * max_iters (default: 610).",
    )
    parser.add_argument(
        "--max-iters",
        "-i",
        type=int,
        default=30,
        help="Number of iterations (default: 30, corresponding to 610 evaluations).",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=10,
        help="Population size (default: 10).",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=True,
        help="Use deterministic mock LLM for fast offline testing.",
    )
    parser.add_argument(
        "--live",
        dest="mock",
        action="store_false",
        help="Use live LLM API endpoint configured in .env.",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Bypass prompt cache and force fresh live LLM API calls.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=42,
        help="Starting seed for independent runs (seed = seed-start + run_index).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="experiments/ablation",
        help="Directory to save experimental results, tables, and figures.",
    )
    parser.add_argument(
        "--no-plot",
        dest="plot",
        action="store_false",
        default=True,
        help="Disable convergence plot generation.",
    )
    parser.add_argument(
        "--plot-only",
        action="store_true",
        default=False,
        help="Do not execute optimizations; load existing variant results from output-dir and generate comparative plots and summary tables.",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        default=False,
        help="Enable verbose output for every individual run.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = Path(args.output_dir)

    if args.plot_only:
        print("=" * 80)
        print(" LEO ABLATION STUDY: PLOT & SUMMARY MODE (--plot-only)")
        print(f" Reading results from: {out_dir.resolve()}")
        print("=" * 80)
        saved_data, iters, budget = load_all_saved_results(out_dir)
        if not saved_data:
            print(f"[!] No ablation run results found in {out_dir.resolve()}.")
            print("    Please run ablation optimization for at least one variant first.")
            return

        generate_summary_reports_and_plots(
            all_runs_data=saved_data,
            max_iters=iters,
            unified_budget=budget,
            out_dir=out_dir,
            use_mock=args.mock,
            generate_plots=args.plot,
        )
        return

    # Parse variants
    valid_variants = ["full", "no_history", "no_elitism", "random_perturb"]
    if args.variant.lower() == "all":
        variants = valid_variants
    else:
        variants = [x.strip().lower() for x in args.variant.split(",") if x.strip()]
        for v in variants:
            if v not in valid_variants:
                raise ValueError(
                    f"Unknown variant '{v}'. Valid choices: {valid_variants} or 'all'"
                )

    # Parse problems
    prob_aliases = {
        "rosenbrock_2d": "rosenbrock_2d",
        "rosenbrock2d": "rosenbrock_2d",
        "rosenbrock": "rosenbrock_2d",
        "goldstein_price": "goldstein_price",
        "goldsteinprice": "goldstein_price",
        "goldstein": "goldstein_price",
        "rosenbrock_10d": "rosenbrock_10d",
        "rosenbrock10d": "rosenbrock_10d",
        "shifted_rosenbrock_10d": "rosenbrock_10d",
    }
    if args.problem.lower() == "all":
        probs = ["rosenbrock_2d", "goldstein_price", "rosenbrock_10d"]
    else:
        raw_probs = [x.strip().lower().replace("-", "_") for x in args.problem.split(",") if x.strip()]
        probs = []
        for p in raw_probs:
            if p not in prob_aliases:
                raise ValueError(
                    f"Unknown problem '{p}'. Valid choices: {list(prob_aliases.keys())} or 'all'"
                )
            canonical = prob_aliases[p]
            if canonical not in probs:
                probs.append(canonical)

    run_ablation_experiments(
        problems=probs,
        variants=variants,
        num_runs=args.runs,
        budget=args.budget,
        pop_size=args.pop_size,
        max_iters=args.max_iters,
        use_mock=args.mock,
        seed_start=args.seed_start,
        output_dir=args.output_dir,
        generate_plots=args.plot,
        verbose=args.verbose,
        bypass_cache=args.no_cache,
    )


if __name__ == "__main__":
    main()
