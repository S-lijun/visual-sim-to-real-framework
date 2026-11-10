"""Kernel Inception Distance on saved 2048-d embeddings.

The value is the unbiased polynomial MMD from Binkowski et al., kernel
``(x·y / d + 1)^3``. Lower means the two embedding distributions are closer.
Uncertainty is the standard deviation across random image subsets. This is the
sampling uncertainty of KID itself; trajectory-level intervals belong to the
later metric stage.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .config import ExtractConfig


def load_embeddings(folder: Path) -> np.ndarray:
    files = sorted(folder.glob("episode_*.npz"))
    if not files:
        raise FileNotFoundError(f"No appearance features in {folder}")
    parts = [np.load(path)["embedding"] for path in files]
    return np.concatenate(parts, axis=0).astype(np.float32)


def polynomial_mmd(x: np.ndarray, y: np.ndarray, degree: int = 3) -> float:
    gamma = 1.0 / x.shape[1]
    k_xx = (gamma * (x @ x.T) + 1.0) ** degree
    k_yy = (gamma * (y @ y.T) + 1.0) ** degree
    k_xy = (gamma * (x @ y.T) + 1.0) ** degree
    n = x.shape[0]
    m = y.shape[0]
    sum_xx = (k_xx.sum() - np.trace(k_xx)) / (n * (n - 1))
    sum_yy = (k_yy.sum() - np.trace(k_yy)) / (m * (m - 1))
    return float(sum_xx + sum_yy - 2.0 * k_xy.mean())


def kid_with_uncertainty(
    sim: np.ndarray,
    real: np.ndarray,
    config: ExtractConfig,
) -> dict:
    subset = min(config.kid_subset_size, sim.shape[0], real.shape[0])
    if subset < 2:
        raise ValueError("KID needs at least 2 embeddings on each side")
    rng = np.random.default_rng(config.kid_seed)
    scores = np.empty(config.kid_repeats, dtype=np.float64)
    for index in range(config.kid_repeats):
        sim_idx = rng.choice(sim.shape[0], size=subset, replace=False)
        real_idx = rng.choice(real.shape[0], size=subset, replace=False)
        scores[index] = polynomial_mmd(sim[sim_idx], real[real_idx], degree=config.kid_degree)
    return {
        "metric": "kid_unbiased_polynomial_mmd",
        "lower_is_closer": True,
        "kernel": "(dot(x, y) / d + 1) ^ 3",
        "subset_size": subset,
        "repeats": config.kid_repeats,
        "seed": config.kid_seed,
        "n_sim": int(sim.shape[0]),
        "n_real": int(real.shape[0]),
        "embedding_dim": int(sim.shape[1]),
        "mean": float(scores.mean()),
        "std": float(scores.std(ddof=1)),
        "scores": scores.tolist(),
    }


def write_kid(out_dir: Path, config: ExtractConfig) -> dict:
    sim = load_embeddings(out_dir / "sim" / "appearance")
    real = load_embeddings(out_dir / "real" / "appearance")
    payload = kid_with_uncertainty(sim, real, config)
    (out_dir / "kid.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload
