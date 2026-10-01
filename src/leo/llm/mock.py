"""Deterministic Mock LLM generator for offline testing and fast CI/CD execution.

Simulates the behavior of explore, exploit, and modular prompts without requiring
external network connectivity or LLM API billing.
"""

from __future__ import annotations

import re

import numpy as np

from leo.llm.parser import FLOAT_REGEX


class MockLLM:
    """Mock LLM responding deterministically to LEO prompts."""

    def __init__(self, seed: int = 42) -> None:
        self.rng = np.random.default_rng(seed)

    def generate(self, prompt: str) -> str:
        """Parse prompt intent and return mock CSV candidate matrix or numerical list."""
        # 1. Detect LEO-modular prompt
        if "intelligent assistant who can understand great technical details" in prompt:
            return self._mock_modular(prompt)

        # 2. Extract shape requirements: rows and cols
        row_match = re.search(r"exactly\s+(\d+)\s+new", prompt)
        rows = int(row_match.group(1)) if row_match else 10

        col_match = re.search(r"(\d+)\s+columns", prompt)
        if not col_match:
            col_match = re.search(r"N=(\d+)\s+variables", prompt)
        cols = int(col_match.group(1)) if col_match else 2

        # 3. Detect bounds in between min and max
        bound_match = re.search(
            r"in between\s+([-+]?\d*\.?\d+)\s+and\s+([-+]?\d*\.?\d+)", prompt
        )
        if bound_match:
            min_val = float(bound_match.group(1))
            max_val = float(bound_match.group(2))
        else:
            min_val, max_val = -2.0, 2.0

        # 4. Check prompt type: Explore vs Exploit
        is_exploit = "exploit close by" in prompt or "Current best candidate" in prompt
        is_rnd = "LEO-Rnd" in prompt or "within all variables, in between" in prompt

        if is_rnd:
            candidates = self.rng.uniform(min_val, max_val, size=(rows, cols))
        elif is_exploit:
            # Extract current best point from prompt if available
            best_match = re.search(r"are:\s*\[(.*?)(?:\]|\n)", prompt, flags=re.DOTALL)
            best_coords = None
            if best_match:
                content = best_match.group(1)
                nums = [float(m.group(0)) for m in FLOAT_REGEX.finditer(content)]
                if len(nums) >= cols + 1:
                    # Skip loss (last element)
                    best_coords = np.array(nums[:cols], dtype=np.float64)

            if best_coords is None:
                best_coords = np.zeros(cols)

            # Local perturbation around best point
            span = max_val - min_val
            sigma = 0.08 * span
            noise = self.rng.normal(0, sigma, size=(rows, cols))
            candidates = np.clip(best_coords + noise, min_val, max_val)
        else:
            # Explore: Scattered points across the domain
            candidates = self.rng.uniform(min_val, max_val, size=(rows, cols))

        # Format as CSV string
        lines = []
        for row in candidates:
            lines.append(", ".join(f"{val:.6f}" for val in row))
        return "\n".join(lines)

    def _mock_modular(self, prompt: str) -> str:
        """Generate 2 pairs of values for LEO-modular NSGA-II."""
        is_exploit = "exploit closeby" in prompt
        # Extract parent values if possible
        # Format: Candidate 1: a=0.2536, b=0.7346 ...
        cand_matches = re.findall(r"a=([0-9.]+),\s*b=([0-9.]+)", prompt)
        if len(cand_matches) >= 2:
            p1 = np.array([float(cand_matches[0][0]), float(cand_matches[0][1])])
            p2 = np.array([float(cand_matches[1][0]), float(cand_matches[1][1])])
        else:
            p1 = np.array([0.5, 0.5])
            p2 = np.array([0.5, 0.5])

        if is_exploit:
            # Offspring 1 near p1, Offspring 2 near midpoint/p2
            o1 = p1 + self.rng.normal(0, 0.05, size=2)
            mid = 0.5 * (p1 + p2)
            o2 = mid + self.rng.normal(0, 0.05, size=2)
        else:
            # Explore: broader mutation/crossover
            diff = p2 - p1
            o1 = p1 + self.rng.uniform(-0.3, 0.3, size=2) * (np.abs(diff) + 0.1)
            o2 = self.rng.uniform(0.0, 1.0, size=2)

        o1 = np.clip(o1, 0.0001, 0.9999)
        o2 = np.clip(o2, 0.0001, 0.9999)
        return f"{o1[0]:.4f}, {o1[1]:.4f}, {o2[0]:.4f}, {o2[1]:.4f}"
