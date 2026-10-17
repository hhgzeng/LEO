"""Benchmark Comparison Experiment Runner: LEO vs LEO-Rnd vs CMA-ES.

Evaluates 3 benchmark functions:
1. 2D Rosenbrock
2. Goldstein-Price
3. 10D Rosenbrock (ShiftedRosenbrockND, dim=10)

Under a unified evaluation budget across multiple independent runs.
Reports:
- Median, IQR (Interquartile Range: Q3 - Q1), Q1, Q3, Mean ± Std, and all run results.
- Convergence curves with Function Evaluations on the X-axis (median + IQR bands).
- Separate dedicated reports for execution time and LLM token/financial costs.
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

from leo.baselines.classical import CMAESOptimizer
from leo.baselines.leo_rnd import LEORndOptimizer
from leo.core.algorithm import LEOAlgorithm
from leo.llm.client import LLMClient, LLMSettings
from leo.problems.base import BaseProblem
from leo.problems.benchmarks_2d import GoldsteinPriceProblem, Rosenbrock2DProblem
from leo.problems.rosenbrock_nd import ShiftedRosenbrockND


class TrackingLLMClient:
    """Thread-safe LLM client wrapper that records token usage, queries, and cost."""

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
        """Returns (prompt_price_per_1m, completion_price_per_1m) in USD."""
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
        prompt_price, comp_price = self.get_pricing()
        return (prompt_toks / 1e6) * prompt_price + (comp_toks / 1e6) * comp_price

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

        # Check cache if live
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
    """Result of a single optimization run, stored by iteration."""

    algorithm: str
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
    history_best_y: list[float]  # Length: iterations + 1 (best-so-far objective at iteration 0..max_iters)
    iteration_checkpoints: list[int] = field(default_factory=list)

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> SingleRunResult:
        hist = d.get("history_best_y", [])
        iters = d.get("iterations", 30)
        # Resample legacy evaluation-level history (e.g. 610 checkpoints) to iteration checkpoints
        if len(hist) > iters + 1:
            pop_size = 10
            target_cps = [pop_size + 20 * k for k in range(iters + 1)]
            hist = [hist[min(len(hist) - 1, cp - 1)] for cp in target_cps]
        cps = d.get("iteration_checkpoints", list(range(len(hist))))
        return cls(
            algorithm=d["algorithm"],
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
    """Factory for the 3 benchmark problems."""
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


def run_single_optimization(
    algorithm: str,
    problem_key: str,
    seed: int,
    budget: int,
    pop_size: int,
    max_iters: int,
    tracking_client: TrackingLLMClient,
    use_mock: bool,
) -> SingleRunResult:
    """Executes one independent run of an algorithm on a benchmark problem."""
    problem = get_problem_instance(problem_key)
    algo_name = algorithm.lower().strip()

    # Reset tracker stats before run
    tracking_client.reset_stats()
    t0 = time.perf_counter()

    if algo_name == "leo":
        optimizer = LEOAlgorithm(
            problem=problem,
            pop_size=pop_size,
            max_iters=max_iters,
            llm_client=tracking_client,  # type: ignore
            seed=seed,
            verbose=False,
        )
        res = optimizer.optimize()
        wall_time = time.perf_counter() - t0
        if not use_mock and tracking_client.cache_hits > 0:
            wall_time += sum(tracking_client.cached_latencies) / 2.0
        best_y = float(res.best_y)
        best_x = [float(v) for v in res.best_x]
        raw_history = [float(v) for v in res.history_exploit_best_y]
        total_evals = res.total_evaluations
        if len(raw_history) < max_iters + 1:
            raw_history = raw_history + [best_y] * (max_iters + 1 - len(raw_history))
        else:
            raw_history = raw_history[: max_iters + 1]
        iter_checkpoints = list(range(len(raw_history)))

    elif algo_name in ["leo-rnd", "leornd", "leo_rnd"]:
        optimizer = LEORndOptimizer(
            problem=problem,
            pop_size=pop_size,
            max_iters=max_iters,
            llm_client=tracking_client,  # type: ignore
            seed=seed,
        )
        res = optimizer.optimize()
        wall_time = time.perf_counter() - t0
        if not use_mock and tracking_client.cache_hits > 0:
            wall_time += sum(tracking_client.cached_latencies) / 2.0
        best_y = float(res.best_y)
        best_x = [float(v) for v in res.best_x]
        raw_history = [float(v) for v in res.history_exploit_best_y]
        total_evals = res.total_evaluations
        if len(raw_history) < max_iters + 1:
            raw_history = raw_history + [best_y] * (max_iters + 1 - len(raw_history))
        else:
            raw_history = raw_history[: max_iters + 1]
        iter_checkpoints = list(range(len(raw_history)))

    elif algo_name in ["cma-es", "cmaes", "cma"]:
        optimizer = CMAESOptimizer(
            problem=problem,
            max_evals=budget,
            seed=seed,
        )
        res = optimizer.optimize()
        wall_time = time.perf_counter() - t0
        best_y = float(res.best_y)
        best_x = [float(v) for v in res.best_x]
        raw_eval_history = [float(v) for v in res.history_best_y]
        total_evals = res.evaluations
        # Sample by iterations: Iteration 0 = pop_size evals, Iteration k = pop_size + 20*k evals (20 evals per iteration)
        target_checkpoints = [pop_size + 20 * k for k in range(max_iters + 1)]
        raw_history = []
        for cp in target_checkpoints:
            idx = min(len(raw_eval_history) - 1, cp - 1)
            raw_history.append(raw_eval_history[idx])
        iter_checkpoints = list(range(len(raw_history)))

    else:
        raise ValueError(
            f"Unknown algorithm '{algorithm}'. Choices: leo, leo-rnd, cma-es"
        )

    queries = tracking_client.total_queries
    p_toks = tracking_client.prompt_tokens
    c_toks = tracking_client.completion_tokens
    cost = tracking_client.calculate_cost(p_toks, c_toks)

    return SingleRunResult(
        algorithm=algo_name,
        problem=problem.name,
        seed=seed,
        best_y=best_y,
        best_x=best_x,
        total_evaluations=total_evals,
        iterations=max_iters,
        wall_time_seconds=wall_time,
        queries_count=queries,
        prompt_tokens=p_toks,
        completion_tokens=c_toks,
        total_tokens=p_toks + c_toks,
        estimated_cost_usd=cost,
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


def plot_convergence(
    problem_name: str,
    algo_runs: dict[str, list[SingleRunResult]],
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

    styles = {
        "leo": {"label": "LEO", "color": "#d62728", "linestyle": "-", "marker": "o"},
        "leo-rnd": {
            "label": "LEO-Rnd",
            "color": "#1f77b4",
            "linestyle": "--",
            "marker": "s",
        },
        "cma-es": {
            "label": "CMA-ES",
            "color": "#2ca02c",
            "linestyle": "-.",
            "marker": "^",
        },
    }

    x_iters = np.arange(0, max_iters + 1)

    for algo, runs in algo_runs.items():
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

        style = styles.get(
            algo, {"label": algo, "color": "gray", "linestyle": "-", "marker": None}
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
        f"Convergence Comparison on {problem_name}", fontsize=13, fontweight="bold"
    )
    ax.set_xlabel("Iterations (迭代次数)", fontsize=11)
    ax.set_ylabel("Best-so-far Objective Value ($f_{min}$)", fontsize=11)
    ax.set_yscale("log")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True, fontsize=10)

    # Secondary X-axis for equivalent function evaluations (1 iteration = 20 evaluations)
    secax = ax.secondary_xaxis(
        "top", functions=(lambda x: x * 20, lambda x: x / 20)
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

    styles = {
        "leo": {"label": "LEO", "color": "#d62728", "linestyle": "-", "marker": "o"},
        "leo-rnd": {
            "label": "LEO-Rnd",
            "color": "#1f77b4",
            "linestyle": "--",
            "marker": "s",
        },
        "cma-es": {
            "label": "CMA-ES",
            "color": "#2ca02c",
            "linestyle": "-.",
            "marker": "^",
        },
    }

    x_iters = np.arange(0, max_iters + 1)
    markevery = max(1, max_iters // 8)

    for idx, (prob_name, algo_dict) in enumerate(all_results.items()):
        ax = axes[idx]
        for algo, runs in algo_dict.items():
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

            style = styles.get(
                algo,
                {"label": algo, "color": "gray", "linestyle": "-", "marker": None},
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


def save_algorithm_problem_runs(
    out_dir: Path,
    algo_name: str,
    problem_name: str,
    runs: list[SingleRunResult],
    iterations: int,
    budget: int,
) -> Path:
    """Saves runs of a single algorithm on a specific problem into a dedicated algorithm subfolder."""
    algo_dir = out_dir / algo_name
    algo_dir.mkdir(parents=True, exist_ok=True)
    clean_prob = problem_name.lower().replace("-", "_")
    file_path = algo_dir / f"{algo_name}_{clean_prob}.json"
    data = {
        "algorithm": algo_name,
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
    """Scans out_dir for all algorithm subdirectories and loads their run results."""
    all_runs_data: dict[str, dict[str, list[SingleRunResult]]] = {}
    found_iters = 30
    found_budget = 610

    if not out_dir.exists():
        return all_runs_data, found_iters, found_budget

    for algo_dir in sorted(out_dir.iterdir()):
        if not algo_dir.is_dir() or algo_dir.name in [
            "archive",
            "__pycache__",
            "figures",
        ]:
            continue
        algo_name = algo_dir.name

        for json_file in sorted(algo_dir.glob("*.json")):
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
                all_runs_data[prob_name][algo_name] = runs
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                print(f"[!] Warning: failed to parse {json_file}: {e}")

    return all_runs_data, found_iters, found_budget


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
        print("[!] No benchmark run data available to generate summaries.")
        return

    metrics_records: list[dict[str, Any]] = []
    time_cost_records: list[dict[str, Any]] = []

    # Detect sample LLM run for metadata (prioritize LLM algorithms over classical baselines)
    llm_runs = [
        r
        for p_dict in all_runs_data.values()
        for a_name, r_list in p_dict.items()
        if a_name.lower() not in ["cma-es", "cmaes", "cma"]
        for r in r_list
    ]
    first_prob = next(iter(all_runs_data.values()))
    first_algo = next(iter(first_prob.values()))
    detected_mock = (
        llm_runs[0].is_mock
        if llm_runs
        else (first_algo[0].is_mock if first_algo else use_mock)
    )
    num_runs = len(first_algo) if first_algo else 1

    json_summary: dict[str, Any] = {
        "metadata": {
            "budget": unified_budget,
            "num_runs": num_runs,
            "use_mock": detected_mock,
            "model_name": model_name,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "problems": {},
    }

    for prob_name, algo_dict in all_runs_data.items():
        json_summary["problems"][prob_name] = {}
        for algo_name, runs in algo_dict.items():
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
            algo_is_mock = runs[0].is_mock if runs else use_mock

            if algo_name.lower() in ["cma-es", "cmaes", "cma"] or total_queries == 0:
                mode_label = "Classical (CPU)"
            else:
                mode_label = "Mock (Simulated)" if algo_is_mock else "Live API"

            metrics_records.append(
                {
                    "Problem": prob_name,
                    "Algorithm": algo_name.upper(),
                    "Runs": len(runs),
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
                    "Algorithm": algo_name.upper(),
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

            json_summary["problems"][prob_name][algo_name] = {
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
                    "is_mock": algo_is_mock,
                },
                "individual_runs": [asdict(r) for r in runs],
            }

    df_metrics = pd.DataFrame(metrics_records)
    df_time_cost = pd.DataFrame(time_cost_records)

    print("\n" + "=" * 100)
    print(" TABLE 1: OPTIMIZATION PERFORMANCE (MEDIAN, IQR & ALL RUNS)")
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
        print(f"[{row['Problem']} | {row['Algorithm']}]: {row['All Runs']}")

    json_path = out_dir / "comparison_summary.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(json_summary, f, indent=2)
    print(f"\n[+] Saved merged JSON summary to: {json_path}")

    md_metrics_path = out_dir / "metrics_table.md"
    with open(md_metrics_path, "w", encoding="utf-8") as f:
        f.write("# Optimization Performance Metrics\n\n")
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

    if generate_plots:
        print("\n>>> Generating convergence plots...")
        for prob_name, algo_dict in all_runs_data.items():
            clean_name = prob_name.lower().replace("-", "_")
            plot_file = out_dir / f"convergence_{clean_name}.png"
            plot_convergence(
                problem_name=prob_name,
                algo_runs=algo_dict,
                max_iters=max_iters,
                out_path=plot_file,
            )
            print(f"[+] Saved convergence plot: {plot_file}")

        combined_plot = out_dir / "convergence_all_benchmarks.png"
        plot_combined_convergence(
            all_results=all_runs_data,
            max_iters=max_iters,
            out_path=combined_plot,
        )
        print(f"[+] Saved combined convergence figure: {combined_plot}")

    print("\n[+] Benchmark comparison results successfully updated!")


def run_experiments(
    problems: list[str],
    algorithms: list[str],
    num_runs: int = 10,
    budget: int | None = None,
    pop_size: int = 10,
    max_iters: int = 30,
    use_mock: bool = True,
    seed_start: int = 0,
    output_dir: str = "experiments/comparison",
    generate_plots: bool = True,
    verbose: bool = False,
    bypass_cache: bool = False,
) -> None:
    """Coordinates execution across requested benchmarks and algorithms, saving to isolated subfolders."""
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
    print(" LEO BENCHMARK COMPARISON EXPERIMENT")
    print(f" Target Problems   : {problems}")
    print(f" Target Algorithms : {algorithms}")
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

        for algo_key in algorithms:
            algo_name = algo_key.lower().strip()
            print(f"  --> Algorithm: {algo_name.upper()} ({num_runs} run(s))...")
            runs_list: list[SingleRunResult] = []

            for run_idx in range(num_runs):
                seed = seed_start + run_idx
                run_res = run_single_optimization(
                    algorithm=algo_name,
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

            # Persist to dedicated subfolder for this algorithm
            saved_file = save_algorithm_problem_runs(
                out_dir=out_dir,
                algo_name=algo_name,
                problem_name=prob_name,
                runs=runs_list,
                iterations=actual_iters,
                budget=unified_budget,
            )
            print(
                f"      [+] Persisted {algo_name.upper()} data for {prob_name} -> {saved_file}"
            )

    # Auto-merge with ALL algorithm runs existing in out_dir (including previously completed ones)
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
        description="Run Benchmark Optimization Experiments: LEO vs LEO-Rnd vs CMA-ES",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--algorithm",
        "-a",
        type=str,
        default="all",
        help="Algorithm(s) to run: 'leo', 'leo-rnd', 'cma-es', or 'all' (or comma-separated).",
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
        help="Number of iterations for LEO & LEO-Rnd (default: 30, corresponding to 610 evaluations).",
    )
    parser.add_argument(
        "--pop-size",
        type=int,
        default=10,
        help="Population size for LEO & LEO-Rnd (default: 10).",
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
        help="Use live LLM API (OpenAI or compatible endpoint configured in .env).",
    )
    parser.add_argument(
        "--no-cache",
        action="store_true",
        default=False,
        help="Bypass SQLite prompt cache and force fresh live LLM API calls.",
    )
    parser.add_argument(
        "--seed-start",
        type=int,
        default=0,
        help="Starting seed for independent runs (seed = seed-start + run_index).",
    )
    parser.add_argument(
        "--output-dir",
        "-o",
        type=str,
        default="experiments/comparison",
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
        help="Do not execute optimizations; load existing algorithm results from output-dir and generate comparative plots and summary tables.",
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
        print(" LEO BENCHMARK COMPARISON: PLOT & SUMMARY MODE (--plot-only)")
        print(f" Reading results from: {out_dir.resolve()}")
        print("=" * 80)
        saved_data, iters, budget = load_all_saved_results(out_dir)
        if not saved_data:
            print(f"[!] No algorithm run results found in {out_dir.resolve()}.")
            print("    Please run optimization for at least one algorithm first.")
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

    # Parse algorithms
    valid_algos = ["leo", "leo-rnd", "cma-es"]
    if args.algorithm.lower() == "all":
        algos = valid_algos
    else:
        algos = [x.strip().lower() for x in args.algorithm.split(",") if x.strip()]
        for a in algos:
            if a not in valid_algos:
                raise ValueError(
                    f"Unknown algorithm '{a}'. Valid choices: {valid_algos} or 'all'"
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

    run_experiments(
        problems=probs,
        algorithms=algos,
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
