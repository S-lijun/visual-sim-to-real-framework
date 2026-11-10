"""Figures for the Phase 4 baseline. Distances are read from Phase 3; features are not recomputed.

Example
-------
python -m phase4.plot_baseline
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3.features import load_domain

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "features"
METRICS = ROOT / "results" / "phase3" / "metrics.csv"
OUT = ROOT / "results" / "phase4"
FIG = OUT / "figures"

SIM_COLOR = "#F58518"
REAL_COLOR = "#4C78A8"
PCA_PER_DOMAIN = 4000
PCA_SEED = 4

plt.rcParams.update(
    {
        "font.size": 10,
        "axes.titlesize": 11,
        "axes.labelsize": 10,
        "legend.fontsize": 9,
        "figure.dpi": 160,
        "savefig.dpi": 160,
        "axes.spines.top": False,
        "axes.spines.right": False,
    }
)


def _pool(chunks: list[np.ndarray]) -> np.ndarray:
    parts = [chunk for chunk in chunks if chunk.size]
    if not parts:
        return np.zeros((0,), dtype=np.float64)
    return np.concatenate(parts, axis=0)


def _finite(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    return values[np.isfinite(values)]


def _density(ax, sim: np.ndarray, real: np.ndarray, title: str, xlabel: str, bins: np.ndarray) -> None:
    ax.hist(real, bins=bins, density=True, histtype="stepfilled", alpha=0.55, color=REAL_COLOR, label="Real")
    ax.hist(sim, bins=bins, density=True, histtype="stepfilled", alpha=0.45, color=SIM_COLOR, label="Sim")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel("Density")


def _central_bins(sim: np.ndarray, real: np.ndarray, n_bins: int = 48) -> np.ndarray:
    lo = min(np.percentile(sim, 1), np.percentile(real, 1))
    hi = max(np.percentile(sim, 99), np.percentile(real, 99))
    if hi <= lo:
        hi = lo + 1.0
    return np.linspace(lo, hi, n_bins)


def _class_names() -> dict[int, str]:
    names: dict[int, str] = {}
    for line in (FEATURES / "yolo_classes.tsv").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        class_id, name = line.split("\t", 1)
        names[int(class_id)] = name
    return names


def _load_metrics() -> list[dict[str, str]]:
    with METRICS.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def plot_semantic(sim, real) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(11.2, 3.4))
    count_labels = ["0", "1", "2", "3+"]

    def _count_share(values: np.ndarray) -> np.ndarray:
        clipped = np.minimum(values.astype(np.int64), 3)
        share = np.bincount(clipped, minlength=4).astype(np.float64)
        return share / share.sum()

    x = np.arange(4)
    width = 0.38
    axes[0].bar(x - width / 2, _count_share(_finite(np.concatenate(real.count))), width, color=REAL_COLOR, label="Real")
    axes[0].bar(x + width / 2, _count_share(_finite(np.concatenate(sim.count))), width, color=SIM_COLOR, label="Sim")
    axes[0].set_xticks(x, count_labels)
    axes[0].set_title("Boxes per frame")
    axes[0].set_xlabel("Count")
    axes[0].set_ylabel("Fraction of frames")

    conf_bins = np.linspace(0.25, 1.0, 31)
    _density(axes[1], _finite(_pool(sim.conf)), _finite(_pool(real.conf)), "Detection confidence", "Confidence", conf_bins)
    area_bins = np.linspace(0.0, 1.0, 41)
    _density(
        axes[2],
        np.clip(_finite(_pool(sim.area)), 0.0, 1.0),
        np.clip(_finite(_pool(real.area)), 0.0, 1.0),
        "Normalized box area",
        "Fraction of the image",
        area_bins,
    )
    axes[0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "semantic_distributions.png")
    plt.close(fig)


def plot_classes(sim, real) -> None:
    names = _class_names()
    sim_labels = _pool(sim.cls).astype(np.int64)
    real_labels = _pool(real.cls).astype(np.int64)
    support = sorted(set(sim_labels.tolist()) | set(real_labels.tolist()))
    sim_share = []
    real_share = []
    for class_id in support:
        sim_share.append(float(np.mean(sim_labels == class_id)) if sim_labels.size else 0.0)
        real_share.append(float(np.mean(real_labels == class_id)) if real_labels.size else 0.0)
    rank = np.argsort(-(np.maximum(sim_share, real_share)))[:10]
    labels = [names.get(support[i], str(support[i])) for i in rank][::-1]
    sim_top = np.asarray(sim_share)[rank][::-1]
    real_top = np.asarray(real_share)[rank][::-1]
    y = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    height = 0.38
    ax.barh(y - height / 2, real_top, height, color=REAL_COLOR, label="Real")
    ax.barh(y + height / 2, sim_top, height, color=SIM_COLOR, label="Sim")
    ax.set_yticks(y, labels)
    ax.set_xlabel("Share of detections")
    ax.set_title("Most common detector labels")
    ax.legend(frameon=False, loc="center right")
    fig.tight_layout()
    fig.savefig(FIG / "class_frequencies.png")
    plt.close(fig)


def plot_geometry(sim, real) -> None:
    keys = (
        ("disparity_mean", "Disparity mean", "MiDaS disparity"),
        ("disparity_std", "Disparity std", "MiDaS disparity"),
        ("disparity_p50", "Disparity median", "MiDaS disparity"),
        ("disparity_p95", "Disparity p95", "MiDaS disparity"),
        ("grad_mean", "Gradient mean", "Disparity per pixel"),
        ("grad_p95", "Gradient p95", "Disparity per pixel"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(11.2, 6.2))
    for ax, (key, title, xlabel) in zip(axes.ravel(), keys):
        sim_values = _finite(_pool(sim.scalars[key]))
        real_values = _finite(_pool(real.scalars[key]))
        _density(ax, sim_values, real_values, title, xlabel, _central_bins(sim_values, real_values))
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "geometry_distributions.png")
    plt.close(fig)


def plot_temporal(sim, real) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.2))
    scalar_keys = (
        ("mag_mean", "Flow magnitude mean", "Pixels per frame"),
        ("mag_p95", "Flow magnitude p95", "Pixels per frame"),
        ("motion_frac", "Motion fraction", "Fraction of pixels above 1 px"),
    )
    for ax, (key, title, xlabel) in zip(axes.ravel(), scalar_keys):
        sim_values = _finite(_pool(sim.scalars[key]))
        real_values = _finite(_pool(real.scalars[key]))
        _density(ax, sim_values, real_values, title, xlabel, _central_bins(sim_values, real_values))
    sim_hist = _pool(sim.dir_hist).sum(axis=0)
    real_hist = _pool(real.dir_hist).sum(axis=0)
    width = 2.0 * np.pi / sim_hist.size
    # Place the branch cut on a low-density bin so one circular mode is not drawn as two peaks.
    shift = int(np.argmin(sim_hist + real_hist))
    sim_hist = np.roll(sim_hist, -shift)
    real_hist = np.roll(real_hist, -shift)
    start = -np.pi + shift * width
    centers = start + (np.arange(sim_hist.size) + 0.5) * width
    axes[1, 1].plot(centers, real_hist / real_hist.sum(), color=REAL_COLOR, label="Real")
    axes[1, 1].plot(centers, sim_hist / sim_hist.sum(), color=SIM_COLOR, label="Sim")
    axes[1, 1].set_title("Flow direction")
    axes[1, 1].set_xlabel("Angle, branch cut at a low-density bin")
    axes[1, 1].set_ylabel("Magnitude-weighted share")

    def _pi_label(angle: float) -> str:
        units = ((angle + np.pi) % (2 * np.pi) - np.pi) / np.pi
        if abs(units) < 0.04:
            return "0"
        if abs(abs(units) - 1.0) < 0.04:
            return "±π"
        text = f"{units:.2f}".rstrip("0").rstrip(".")
        return f"{text}π"

    tick_pos = start + np.array([0.0, 0.5, 1.0, 1.5, 2.0]) * np.pi
    axes[1, 1].set_xticks(tick_pos, [_pi_label(float(pos)) for pos in tick_pos])
    axes[0, 0].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "temporal_distributions.png")
    plt.close(fig)


def plot_pca(sim, real) -> dict[str, float]:
    rng = np.random.default_rng(PCA_SEED)
    sim_emb = _pool(sim.embedding)
    real_emb = _pool(real.embedding)
    sim_idx = rng.choice(sim_emb.shape[0], size=PCA_PER_DOMAIN, replace=False)
    real_idx = rng.choice(real_emb.shape[0], size=PCA_PER_DOMAIN, replace=False)
    sim_sample = sim_emb[sim_idx].astype(np.float64)
    real_sample = real_emb[real_idx].astype(np.float64)
    combined = np.vstack([sim_sample, real_sample])
    centered = combined - combined.mean(axis=0, keepdims=True)
    _, singular, vt = np.linalg.svd(centered, full_matrices=False)
    variance = singular**2
    explained = variance / variance.sum()
    projection = centered @ vt[:2].T
    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    ax.scatter(projection[PCA_PER_DOMAIN:, 0], projection[PCA_PER_DOMAIN:, 1], s=6, alpha=0.35, c=REAL_COLOR, label="Real", linewidths=0)
    ax.scatter(projection[:PCA_PER_DOMAIN, 0], projection[:PCA_PER_DOMAIN, 1], s=6, alpha=0.35, c=SIM_COLOR, label="Sim", linewidths=0)
    ax.set_title("Inception embeddings, first two principal components")
    ax.set_xlabel(f"PC1 ({100 * explained[0]:.1f}% of subsample variance)")
    ax.set_ylabel(f"PC2 ({100 * explained[1]:.1f}% of subsample variance)")
    ax.legend(frameon=False, markerscale=2)
    fig.tight_layout()
    fig.savefig(FIG / "appearance_pca.png")
    plt.close(fig)
    return {
        "per_domain": PCA_PER_DOMAIN,
        "seed": PCA_SEED,
        "pc1": float(explained[0]),
        "pc2": float(explained[1]),
        "pc1_pc2": float(explained[0] + explained[1]),
    }


def plot_ratios(rows: list[dict[str, str]]) -> list[dict[str, float | str]]:
    order = [
        "appearance.kid",
        "semantic.count",
        "semantic.confidence",
        "semantic.box_area",
        "semantic.class_js",
        "geometry.disparity_mean",
        "geometry.disparity_std",
        "geometry.disparity_p05",
        "geometry.disparity_p50",
        "geometry.disparity_p95",
        "geometry.grad_mean",
        "geometry.grad_p95",
        "temporal.mag_mean",
        "temporal.mag_std",
        "temporal.mag_p95",
        "temporal.motion_frac",
        "temporal.direction",
    ]
    labels = {
        "appearance.kid": "Appearance KID",
        "semantic.count": "Box count",
        "semantic.confidence": "Confidence",
        "semantic.box_area": "Box area",
        "semantic.class_js": "Class JS",
        "geometry.disparity_mean": "Disparity mean",
        "geometry.disparity_std": "Disparity std",
        "geometry.disparity_p05": "Disparity p05",
        "geometry.disparity_p50": "Disparity p50",
        "geometry.disparity_p95": "Disparity p95",
        "geometry.grad_mean": "Gradient mean",
        "geometry.grad_p95": "Gradient p95",
        "temporal.mag_mean": "Flow mean",
        "temporal.mag_std": "Flow std",
        "temporal.mag_p95": "Flow p95",
        "temporal.motion_frac": "Motion fraction",
        "temporal.direction": "Direction",
    }
    by_name = {row["metric"]: row for row in rows}
    records = []
    for name in order:
        point = float(by_name[name]["point"])
        reference = float(by_name[name]["real_vs_real_mean"])
        records.append(
            {
                "metric": name,
                "label": labels[name],
                "point": point,
                "real_vs_real_mean": reference,
                "ratio_to_real_vs_real": point / reference,
            }
        )
    fig, ax = plt.subplots(figsize=(7.4, 6.2))
    y = np.arange(len(records))
    ratios = [float(record["ratio_to_real_vs_real"]) for record in records]
    ax.barh(y, ratios, color="#4C78A8")
    ax.axvline(1.0, color="#333333", linewidth=1.0)
    ax.set_yticks(y, [str(record["label"]) for record in records])
    ax.invert_yaxis()
    ax.set_xscale("log")
    ax.set_xlabel("Sim-vs-real distance / real-vs-real mean")
    ax.set_title("Each distance relative to its own real-data spread")
    fig.tight_layout()
    fig.savefig(FIG / "gap_vs_real_variability.png")
    plt.close(fig)
    return records


def main() -> None:
    FIG.mkdir(parents=True, exist_ok=True)
    print("loading features")
    sim = load_domain(FEATURES, "sim")
    real = load_domain(FEATURES, "real")
    print("semantic")
    plot_semantic(sim, real)
    plot_classes(sim, real)
    print("geometry")
    plot_geometry(sim, real)
    print("temporal")
    plot_temporal(sim, real)
    print("pca")
    pca = plot_pca(sim, real)
    ratios = plot_ratios(_load_metrics())
    payload = {"pca": pca, "ratios": ratios, "display_note": "Scalar histograms except count, confidence, and box area show the 1st to 99th percentile window."}
    (OUT / "figure_meta.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (OUT / "ratios.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(ratios[0]))
        writer.writeheader()
        writer.writerows(ratios)
    print(json.dumps(pca))
    for record in ratios:
        print(f"{record['metric']}\t{record['ratio_to_real_vs_real']:.4f}")
    print("wrote", FIG)


if __name__ == "__main__":
    main()
