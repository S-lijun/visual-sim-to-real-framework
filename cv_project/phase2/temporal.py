"""Farneback flow inside a single trajectory.

Parameters match the existing notebook. The first frame has no predecessor.
Nothing is computed across episode boundaries.

Direction is stored as a magnitude-weighted histogram on ``[-pi, pi)`` plus
the magnitude-weighted mean cosine and sine, so later comparisons can treat
angle as circular.
"""

from __future__ import annotations

import cv2
import numpy as np

from .config import ExtractConfig


def flow_features(rgb: np.ndarray, config: ExtractConfig) -> dict[str, np.ndarray]:
    gray = np.empty(rgb.shape[:3], dtype=np.uint8)
    for index in range(rgb.shape[0]):
        gray[index] = cv2.cvtColor(rgb[index], cv2.COLOR_RGB2GRAY)
    n_pairs = rgb.shape[0] - 1
    mag_mean = np.full(n_pairs, np.nan, dtype=np.float32)
    mag_std = np.full(n_pairs, np.nan, dtype=np.float32)
    mag_p95 = np.full(n_pairs, np.nan, dtype=np.float32)
    motion_frac = np.full(n_pairs, np.nan, dtype=np.float32)
    dir_cos = np.full(n_pairs, np.nan, dtype=np.float32)
    dir_sin = np.full(n_pairs, np.nan, dtype=np.float32)
    hist = np.zeros((n_pairs, config.direction_bins), dtype=np.float32)
    edges = np.linspace(-np.pi, np.pi, config.direction_bins + 1)

    for index in range(n_pairs):
        flow = cv2.calcOpticalFlowFarneback(
            gray[index],
            gray[index + 1],
            None,
            pyr_scale=config.flow_pyr_scale,
            levels=config.flow_levels,
            winsize=config.flow_winsize,
            iterations=config.flow_iterations,
            poly_n=config.flow_poly_n,
            poly_sigma=config.flow_poly_sigma,
            flags=0,
        )
        fx = flow[..., 0]
        fy = flow[..., 1]
        mag = np.sqrt(fx * fx + fy * fy)
        mag_mean[index] = float(mag.mean())
        mag_std[index] = float(mag.std())
        mag_p95[index] = float(np.percentile(mag, 95))
        motion_frac[index] = float(np.mean(mag > config.flow_motion_px))
        angle = np.arctan2(fy, fx).reshape(-1)
        weight = mag.reshape(-1)
        weight_sum = float(weight.sum()) + 1e-6
        dir_cos[index] = float(np.sum(np.cos(angle) * weight) / weight_sum)
        dir_sin[index] = float(np.sum(np.sin(angle) * weight) / weight_sum)
        hist[index] = np.histogram(angle, bins=edges, weights=weight)[0].astype(np.float32)

    return {
        "pair_src": np.arange(n_pairs, dtype=np.int32),
        "mag_mean": mag_mean,
        "mag_std": mag_std,
        "mag_p95": mag_p95,
        "motion_frac": motion_frac,
        "dir_cos": dir_cos,
        "dir_sin": dir_sin,
        "dir_hist": hist,
    }
