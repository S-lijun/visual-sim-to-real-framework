"""Line plots of distance against single-factor intensity.

Example
-------
python -m phase5.plot_ablation
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

ROOT = Path(__file__).resolve().parents[1]
METRICS = ROOT / "results" / "phase5" / "metrics.csv"
FIG = ROOT / "results" / "phase5" / "figures"

PANELS = (
    ("appearance.kid", "KID (Appearance)"),
    ("semantic.count", "Object count (Semantic)"),
    ("semantic.confidence", "Confidence (Semantic)"),
    ("semantic.box_area", "Box area (Semantic)"),
    ("semantic.class_js", "Class JS, bits (Semantic)"),
    ("geometry.disparity_mean", "Disparity mean (Geometry)"),
    ("geometry.disparity_std", "Disparity std (Geometry)"),
    ("geometry.disparity_p05", "Disparity p05 (Geometry)"),
    ("geometry.disparity_p50", "Disparity p50 (Geometry)"),
    ("geometry.disparity_p95", "Disparity p95 (Geometry)"),
    ("geometry.grad_mean", "Gradient mean (Geometry)"),
    ("geometry.grad_p95", "Gradient p95 (Geometry)"),
    ("temporal.mag_mean", "Flow magnitude mean (Temporal)"),
    ("temporal.mag_std", "Flow magnitude std (Temporal)"),
    ("temporal.mag_p95", "Flow magnitude p95 (Temporal)"),
    ("temporal.motion_frac", "Motion fraction (Temporal)"),
    ("temporal.direction", "Direction, rad (Temporal)"),
)
FACTORS = (
    ("noise", "Gaussian noise σ", "Noise σ (pixel levels)"),
    ("blur", "Gaussian blur σ", "Blur σ (pixels)"),
    ("brightness", "Brightness shift", "Added level"),
    ("color", "Color temperature", "Signed shift (positive raises R and lowers B)"),
)

plt.rcParams.update(
    {
        "font.size": 9,
        "axes.titlesize": 8,
        "axes.labelsize": 9,
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _series(rows: list[dict[str, str]], factor: str, metric: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    chosen = [row for row in rows if row["factor"] == factor and row["metric"] == metric]
    chosen.sort(key=lambda row: float(row["intensity"]))
    intensity = np.asarray([float(row["intensity"]) for row in chosen], dtype=np.float64)
    point = np.asarray([float(row["point"]) for row in chosen], dtype=np.float64)
    low = np.asarray([float(row["ci95_low"]) for row in chosen], dtype=np.float64)
    high = np.asarray([float(row["ci95_high"]) for row in chosen], dtype=np.float64)
    return intensity, point, low, high


def main() -> None:
    rows = _read(METRICS)
    baseline = {}
    for row in rows:
        baseline.setdefault(row["metric"], float(row["baseline"]))
    FIG.mkdir(parents=True, exist_ok=True)
    for factor, title, xlabel in FACTORS:
        fig, axes = plt.subplots(6, 3, figsize=(11.2, 13.6), sharex=True)
        for ax, (metric, label) in zip(axes.ravel(), PANELS):
            intensity, point, low, high = _series(rows, factor, metric)
            x = np.concatenate([[0.0], intensity])
            y = np.concatenate([[baseline[metric]], point])
            order = np.argsort(x)
            x, y = x[order], y[order]
            ax.plot(x, y, color="#4C78A8", linewidth=1.4)
            if np.isfinite(low).all():
                ax.errorbar(intensity, point, yerr=np.vstack([point - low, high - point]), fmt="o", color="#4C78A8", markersize=3.5, capsize=2, linewidth=1.0)
            else:
                ax.plot(intensity, point, marker="o", color="#4C78A8")
            ax.plot([0.0], [baseline[metric]], marker="o", color="#333333", markersize=3.5)
            ax.axvline(0.0, color="#bbbbbb", linewidth=0.8)
            ax.set_title(label)
        for ax in axes.ravel()[len(PANELS) :]:
            ax.axis("off")
        for ax in axes[-1, :2]:
            ax.set_xlabel(xlabel)
        axes[-2, -1].tick_params(labelbottom=True)
        axes[-2, -1].set_xlabel(xlabel)
        fig.suptitle(f"{title}. Intensity 0 is the original simulation.", fontsize=12)
        fig.tight_layout()
        fig.savefig(FIG / f"{factor}.png")
        plt.close(fig)

    metrics = [metric for metric, _label in PANELS]
    labels = [label for _metric, label in PANELS]
    columns = []
    for factor, _title, _xlabel in FACTORS:
        seen = [row for row in rows if row["factor"] == factor and row["metric"] == metrics[0]]
        seen.sort(key=lambda row: float(row["intensity"]))
        columns.extend((row["condition_id"], factor) for row in seen)
    delta = np.zeros((len(metrics), len(columns)), dtype=np.float64)
    for i, metric in enumerate(metrics):
        for j, (condition_id, _factor) in enumerate(columns):
            match = next(row for row in rows if row["condition_id"] == condition_id and row["metric"] == metric)
            delta[i, j] = float(match["delta"])
    fig, ax = plt.subplots(figsize=(12.8, 7.2))
    scale = np.nanmax(np.abs(delta), axis=1, keepdims=True)
    scale[scale == 0] = 1.0
    image = ax.imshow(delta / scale, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    ax.set_yticks(np.arange(len(labels)), labels)
    ax.set_xticks(np.arange(len(columns)), [condition for condition, _factor in columns], rotation=45, ha="right")
    ax.set_title("Direction of each change, scaled within its own metric")
    fig.colorbar(image, ax=ax, fraction=0.03, pad=0.02).set_label("Change divided by the largest absolute change in that row")
    fig.tight_layout()
    fig.savefig(FIG / "delta_heatmap.png")
    plt.close(fig)
    print("wrote", FIG)


if __name__ == "__main__":
    main()
