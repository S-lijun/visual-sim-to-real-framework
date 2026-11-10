"""Build the Phase 3 sim-to-real distribution-gap profile.

Example
-------
python -m phase3.run
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase3.compare import all_distances, sample_counts
from phase3.distances import circular_wasserstein
from phase3.features import DomainData, class_support, load_domain

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "features"
OUT = ROOT / "results" / "phase3"

UNITS = {
    "appearance.kid": "unbiased polynomial MMD",
    "semantic.count": "boxes per frame",
    "semantic.confidence": "confidence",
    "semantic.box_area": "fraction of image",
    "semantic.class_js": "bits",
    "temporal.direction": "radians",
    "temporal.mag_mean": "pixels per frame",
    "temporal.mag_std": "pixels per frame",
    "temporal.mag_p95": "pixels per frame",
    "temporal.motion_frac": "fraction of pixels",
}
for _name in ("disparity_mean", "disparity_std", "disparity_p05", "disparity_p50", "disparity_p95"):
    UNITS[f"geometry.{_name}"] = "MiDaS disparity (larger is closer)"
UNITS["geometry.grad_mean"] = "disparity per pixel"
UNITS["geometry.grad_p95"] = "disparity per pixel"


def _dimension(name: str) -> str:
    return name.split(".", 1)[0]


def _identity(n: int) -> np.ndarray:
    return np.arange(n, dtype=np.int64)


def _split_indices(n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    order = rng.permutation(n)
    cut = n // 2
    return np.sort(order[:cut]), np.sort(order[cut:])


def _summarize(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "p2_5": float("nan"), "p97_5": float("nan"), "n": 0}
    return {
        "mean": float(finite.mean()),
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "p2_5": float(np.percentile(finite, 2.5)),
        "p97_5": float(np.percentile(finite, 97.5)),
        "n": int(finite.size),
    }


def _reference(
    domain: DomainData,
    support: np.ndarray,
    n_splits: int,
    seed: int,
    device: torch.device,
) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    names: list[str] | None = None
    rows: list[list[float]] = []
    for split in range(n_splits):
        left_idx, right_idx = _split_indices(len(domain), rng)
        left_embed = torch.from_numpy(np.concatenate([domain.embedding[int(i)] for i in left_idx], axis=0)).to(device)
        right_embed = torch.from_numpy(np.concatenate([domain.embedding[int(i)] for i in right_idx], axis=0)).to(device)
        values = all_distances(
            domain,
            domain,
            left_idx,
            right_idx,
            support,
            left_embed,
            right_embed,
            kid_repeats=10,
            kid_seed=10_000 + split,
        )
        if names is None:
            names = list(values)
        rows.append([values[name] for name in names])
        if split % 10 == 0:
            print(f"  {domain.name} split {split + 1}/{n_splits}")
    matrix = np.asarray(rows, dtype=np.float64)
    assert names is not None
    return {name: {**_summarize(matrix[:, i]), "values": matrix[:, i].tolist()} for i, name in enumerate(names)}


def _bootstrap(
    sim: DomainData,
    real: DomainData,
    support: np.ndarray,
    n_boot: int,
    seed: int,
    device: torch.device,
) -> dict[str, dict]:
    rng = np.random.default_rng(seed)
    names: list[str] | None = None
    rows: list[list[float]] = []
    for draw in range(n_boot):
        sim_idx = rng.integers(0, len(sim), size=len(sim))
        real_idx = rng.integers(0, len(real), size=len(real))
        sim_embed = torch.from_numpy(np.concatenate([sim.embedding[int(i)] for i in sim_idx], axis=0)).to(device)
        real_embed = torch.from_numpy(np.concatenate([real.embedding[int(i)] for i in real_idx], axis=0)).to(device)
        values = all_distances(
            sim,
            real,
            sim_idx,
            real_idx,
            support,
            sim_embed,
            real_embed,
            kid_repeats=10,
            kid_seed=20_000 + draw,
        )
        if names is None:
            names = list(values)
        rows.append([values[name] for name in names])
        if draw % 20 == 0:
            print(f"  bootstrap {draw + 1}/{n_boot}")
    matrix = np.asarray(rows, dtype=np.float64)
    assert names is not None
    return {name: _summarize(matrix[:, i]) for i, name in enumerate(names)}


def _check_circular() -> None:
    adjacent = circular_wasserstein(np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0, 0.0]))
    opposite = circular_wasserstein(np.array([1.0, 0.0, 0.0, 0.0]), np.array([0.0, 0.0, 1.0, 0.0]))
    if not np.isclose(adjacent, np.pi / 2) or not np.isclose(opposite, np.pi):
        raise RuntimeError(f"Circular W1 self-check failed: {adjacent}, {opposite}")


def build_profile(features: Path, out_dir: Path, n_boot: int, n_splits: int) -> Path:
    _check_circular()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    sim = load_domain(features, "sim")
    real = load_domain(features, "real")
    support = class_support(sim, real)
    real_support = class_support(real)
    sim_support = class_support(sim)
    print(f"loaded sim {len(sim)} episodes, real {len(real)} episodes, {support.size} classes")

    sim_idx = _identity(len(sim))
    real_idx = _identity(len(real))
    point = all_distances(sim, real, sim_idx, real_idx, support)
    published_kid = json.loads((features / "kid.json").read_text(encoding="utf-8"))
    n_sim_frames = int(sum(x.shape[0] for x in sim.embedding))
    n_real_frames = int(sum(x.shape[0] for x in real.embedding))
    if published_kid["n_sim"] != n_sim_frames or published_kid["n_real"] != n_real_frames:
        raise RuntimeError("features/kid.json does not match the loaded embeddings")
    point["appearance.kid"] = float(published_kid["mean"])
    print("point KID", point["appearance.kid"])

    print("trajectory bootstrap")
    boot = _bootstrap(sim, real, support, n_boot, seed=1, device=device)
    print("real-vs-real")
    real_ref = _reference(real, real_support, n_splits, seed=2, device=device)
    print("sim-vs-sim")
    sim_ref = _reference(sim, sim_support, n_splits, seed=3, device=device)

    sim_n = sample_counts(sim, sim_idx)
    real_n = sample_counts(real, real_idx)
    rows = []
    for name, estimate in point.items():
        boot_stats = boot[name]
        real_stats = real_ref[name]
        sim_stats = sim_ref[name]
        rows.append(
            {
                "metric": name,
                "dimension": _dimension(name),
                "unit": UNITS[name],
                "lower_is_closer": True,
                "point": estimate,
                "ci95_low": boot_stats["p2_5"],
                "ci95_high": boot_stats["p97_5"],
                "bootstrap_std": boot_stats["std"],
                "n_bootstrap": boot_stats["n"],
                "real_vs_real_mean": real_stats["mean"],
                "real_vs_real_std": real_stats["std"],
                "real_vs_real_p2_5": real_stats["p2_5"],
                "real_vs_real_p97_5": real_stats["p97_5"],
                "sim_vs_sim_mean": sim_stats["mean"],
                "sim_vs_sim_std": sim_stats["std"],
                "n_sim_episodes": len(sim),
                "n_real_episodes": len(real),
                "n_sim_frames": sim_n["frames"],
                "n_real_frames": real_n["frames"],
                "n_sim_detections": sim_n["detections"],
                "n_real_detections": real_n["detections"],
                "n_sim_flow_pairs": sim_n["flow_pairs"],
                "n_real_flow_pairs": real_n["flow_pairs"],
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "lower_is_closer": True,
        "kid": {
            "kernel": "(dot(x, y) / 2048 + 1) ^ 3",
            "point_source": "features/kid.json",
            "point_subsets": published_kid["repeats"],
            "point_seed": published_kid["seed"],
            "point_subset_std": published_kid["std"],
            "bootstrap_subsets_per_draw": 10,
            "subset_size": published_kid["subset_size"],
            "note": "point_subset_std is image-subset noise. ci95 is the trajectory bootstrap.",
        },
        "bootstrap": {
            "method": "resample trajectories with replacement, same count as the original domain",
            "draws": n_boot,
            "seed": 1,
            "interval": "percentile 2.5 to 97.5",
        },
        "within_domain": {
            "method": "repeated non-overlapping random halves",
            "splits": n_splits,
            "real_seed": 2,
            "sim_seed": 3,
            "class_support": "classes observed inside that domain",
        },
        "sim_vs_real_class_support": support.astype(int).tolist(),
        "sample_counts": {"sim": sim_n, "real": real_n},
        "metrics": rows,
    }
    json_path = out_dir / "metrics.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    csv_path = out_dir / "metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", json_path)
    return json_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Compute the Phase 3 distribution-gap profile.")
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--bootstrap", type=int, default=200)
    parser.add_argument("--splits", type=int, default=50)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    build_profile(args.features, args.out, args.bootstrap, args.splits)


if __name__ == "__main__":
    main()
