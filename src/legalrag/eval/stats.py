"""Uncertainty for per-query metrics.

* ``bootstrap_ci``: percentile bootstrap over queries for a mean.
* ``paired_bootstrap``: confidence interval for the difference of two systems on the
  same queries.
* ``paired_permutation``: two-sided sign-flip randomisation test for that difference.
* ``holm``: Holm-Bonferroni adjustment for a family of comparisons.
* ``wilson``: interval for a proportion, used for small generation samples.

The resampling unit is the query. Queries drawn from the same citing court or the
same cited passage are not independent, so intervals are somewhat too narrow; the
benchmark limits this by sampling one query per citing opinion.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import numpy as np


def bootstrap_ci(
    values: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """``(mean, low, high)``."""
    x = np.asarray(values, dtype=np.float64)
    if x.size == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    low, high = np.quantile(means, [alpha / 2, 1 - alpha / 2])
    return float(x.mean()), float(low), float(high)


def paired_bootstrap(
    a: Sequence[float], b: Sequence[float], n_boot: int = 2000, alpha: float = 0.05, seed: int = 0
) -> tuple[float, float, float]:
    """Mean of ``a - b`` with its confidence interval."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return bootstrap_ci(d, n_boot, alpha, seed)


def paired_permutation(
    a: Sequence[float], b: Sequence[float], n_perm: int = 10000, seed: int = 0
) -> float:
    """Two-sided p-value for mean(a) == mean(b) under random sign flips."""
    d = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    if d.size == 0 or not np.any(d):
        return 1.0
    rng = np.random.default_rng(seed)
    observed = abs(d.mean())
    exceed = 0
    block = 500
    for start in range(0, n_perm, block):
        m = min(block, n_perm - start)
        signs = rng.choice((-1.0, 1.0), size=(m, d.size))
        exceed += int(np.sum(np.abs((signs * d).mean(axis=1)) >= observed - 1e-12))
    return (exceed + 1) / (n_perm + 1)


def holm(p_values: Sequence[float]) -> list[float]:
    """Holm-Bonferroni adjusted p-values, in the input order."""
    m = len(p_values)
    order = sorted(range(m), key=lambda i: p_values[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p_values[i]))
        adjusted[i] = running
    return adjusted


def wilson(successes: int, n: int, z: float = 1.959964) -> tuple[float, float, float]:
    """``(proportion, low, high)`` Wilson score interval."""
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    p = successes / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return p, max(0.0, centre - half), min(1.0, centre + half)


def auroc(scores: Sequence[float], labels: Sequence[int]) -> float:
    """Area under the ROC curve via the rank-sum statistic (ties get mean rank)."""
    s = np.asarray(scores, dtype=np.float64)
    y = np.asarray(labels, dtype=bool)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(s.size)
    sorted_s = s[order]
    i = 0
    while i < s.size:
        j = i
        while j + 1 < s.size and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def summarise(per_query: dict[str, list[float]], seed: int = 0) -> dict[str, dict[str, float]]:
    """Mean and 95% bootstrap interval for each metric."""
    out = {}
    for name, values in per_query.items():
        mean, low, high = bootstrap_ci(values, seed=seed)
        out[name] = {"mean": round(mean, 4), "ci_low": round(low, 4), "ci_high": round(high, 4)}
    return out
