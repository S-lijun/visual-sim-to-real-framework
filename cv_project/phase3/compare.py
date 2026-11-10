"""Sim-vs-real gap profile with trajectory resampling.

The point estimate uses every trajectory once. The confidence interval resamples
whole trajectories with replacement, so neighboring frames are not treated as
independent observations. Real-vs-real and sim-vs-sim use repeated
non-overlapping splits of the trajectories in that domain.
"""

from __future__ import annotations

import numpy as np
import torch

from .distances import circular_wasserstein, class_counts, jensen_shannon, wasserstein_1d
from .features import DomainData

KID_SUBSET = 1000
KID_POINT_REPEATS = 100
KID_BOOT_REPEATS = 10
KID_DEGREE = 3


def _pool(chunks: list[np.ndarray], indices: np.ndarray) -> np.ndarray:
    parts = [chunks[int(i)] for i in indices if chunks[int(i)].size]
    if not parts:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(parts, axis=0)


def _pool_labels(chunks: list[np.ndarray], indices: np.ndarray, support: np.ndarray) -> np.ndarray:
    total = class_counts(np.zeros((0,), dtype=np.int16), support)
    for index in indices:
        total += class_counts(chunks[int(index)], support)
    return total


def _pool_hist(chunks: list[np.ndarray], indices: np.ndarray) -> np.ndarray:
    total = np.zeros(chunks[0].shape[1], dtype=np.float64)
    for index in indices:
        hist = chunks[int(index)]
        if hist.size:
            total += hist.sum(axis=0)
    return total


def _kid_once(x: torch.Tensor, y: torch.Tensor, generator: torch.Generator) -> float:
    subset = min(KID_SUBSET, int(x.shape[0]), int(y.shape[0]))
    if subset < 2:
        return float("nan")
    x_idx = torch.randperm(x.shape[0], generator=generator, device=x.device)[:subset]
    y_idx = torch.randperm(y.shape[0], generator=generator, device=y.device)[:subset]
    xs = x[x_idx]
    ys = y[y_idx]
    gamma = 1.0 / xs.shape[1]
    k_xx = (gamma * (xs @ xs.T) + 1.0).pow(KID_DEGREE)
    k_yy = (gamma * (ys @ ys.T) + 1.0).pow(KID_DEGREE)
    k_xy = (gamma * (xs @ ys.T) + 1.0).pow(KID_DEGREE)
    n = subset
    sum_xx = (k_xx.sum() - torch.trace(k_xx)) / (n * (n - 1))
    sum_yy = (k_yy.sum() - torch.trace(k_yy)) / (n * (n - 1))
    return float((sum_xx + sum_yy - 2.0 * k_xy.mean()).item())


def kid_score(x: torch.Tensor, y: torch.Tensor, repeats: int, seed: int) -> float:
    generator = torch.Generator(device=x.device)
    generator.manual_seed(seed)
    scores = [_kid_once(x, y, generator) for _ in range(repeats)]
    return float(np.nanmean(scores))


def _stack_embeddings(chunks: list[np.ndarray], indices: np.ndarray, device: torch.device) -> torch.Tensor:
    parts = [torch.from_numpy(chunks[int(i)]) for i in indices]
    return torch.cat(parts, dim=0).to(device)


def scalar_distances(left: DomainData, right: DomainData, indices_left: np.ndarray, indices_right: np.ndarray) -> dict[str, float]:
    out = {
        "semantic.count": wasserstein_1d(_pool(left.count, indices_left), _pool(right.count, indices_right)),
        "semantic.confidence": wasserstein_1d(_pool(left.conf, indices_left), _pool(right.conf, indices_right)),
        "semantic.box_area": wasserstein_1d(_pool(left.area, indices_left), _pool(right.area, indices_right)),
    }
    for key, chunks_left in left.scalars.items():
        chunks_right = right.scalars[key]
        dimension = "geometry" if key.startswith("disparity") or key.startswith("grad") else "temporal"
        out[f"{dimension}.{key}"] = wasserstein_1d(_pool(chunks_left, indices_left), _pool(chunks_right, indices_right))
    return out


def categorical_distances(
    left: DomainData,
    right: DomainData,
    indices_left: np.ndarray,
    indices_right: np.ndarray,
    support: np.ndarray,
) -> dict[str, float]:
    left_counts = _pool_labels(left.cls, indices_left, support)
    right_counts = _pool_labels(right.cls, indices_right, support)
    if left_counts.sum() == 0 or right_counts.sum() == 0:
        js = float("nan")
    else:
        js = jensen_shannon(left_counts, right_counts)
    return {
        "semantic.class_js": js,
        "temporal.direction": circular_wasserstein(
            _pool_hist(left.dir_hist, indices_left),
            _pool_hist(right.dir_hist, indices_right),
        ),
    }


def all_distances(
    left: DomainData,
    right: DomainData,
    indices_left: np.ndarray,
    indices_right: np.ndarray,
    support: np.ndarray,
    left_embed: torch.Tensor | None = None,
    right_embed: torch.Tensor | None = None,
    kid_repeats: int = KID_BOOT_REPEATS,
    kid_seed: int = 0,
) -> dict[str, float]:
    values = scalar_distances(left, right, indices_left, indices_right)
    values.update(categorical_distances(left, right, indices_left, indices_right, support))
    if left_embed is not None and right_embed is not None:
        values["appearance.kid"] = kid_score(left_embed, right_embed, kid_repeats, kid_seed)
    return values


def sample_counts(domain: DomainData, indices: np.ndarray) -> dict[str, int]:
    return {
        "episodes": int(indices.size),
        "frames": int(sum(domain.count[int(i)].size for i in indices)),
        "detections": int(sum(domain.conf[int(i)].size for i in indices)),
        "flow_pairs": int(sum(domain.dir_hist[int(i)].shape[0] for i in indices)),
        "embeddings": int(sum(domain.embedding[int(i)].shape[0] for i in indices)),
    }
