"""Distances from each single-factor simulation condition to the original real set.

The baseline uses the same simulation episodes as the perturbations, read from the
original Phase 2 features. Real features are not recomputed.

Example
-------
python -m phase5.evaluate
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

from phase3.compare import all_distances
from phase3.features import DomainData, load_domain
from phase3.run import UNITS
from phase5.perturb import CONDITIONS

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ROOT / "features"
PHASE5 = FEATURES / "phase5"
OUT = ROOT / "results" / "phase5"


def _mmd(x: torch.Tensor, y: torch.Tensor) -> float:
    gamma = 1.0 / x.shape[1]
    k_xx = (gamma * (x @ x.T) + 1.0).pow(3)
    k_yy = (gamma * (y @ y.T) + 1.0).pow(3)
    k_xy = (gamma * (x @ y.T) + 1.0).pow(3)
    n = x.shape[0]
    sum_xx = (k_xx.sum() - torch.trace(k_xx)) / (n * (n - 1))
    sum_yy = (k_yy.sum() - torch.trace(k_yy)) / (n * (n - 1))
    return float((sum_xx + sum_yy - 2.0 * k_xy.mean()).item())


def kid_point(sim: np.ndarray, real: np.ndarray, device: torch.device, repeats: int = 100, subset: int = 1000, seed: int = 0) -> float:
    sim_t = torch.from_numpy(np.ascontiguousarray(sim)).to(device)
    real_t = torch.from_numpy(np.ascontiguousarray(real)).to(device)
    rng = np.random.default_rng(seed)
    width = min(subset, int(sim_t.shape[0]), int(real_t.shape[0]))
    scores = np.empty(repeats, dtype=np.float64)
    for index in range(repeats):
        sim_idx = torch.as_tensor(rng.choice(sim_t.shape[0], width, replace=False), device=device)
        real_idx = torch.as_tensor(rng.choice(real_t.shape[0], width, replace=False), device=device)
        scores[index] = _mmd(sim_t[sim_idx], real_t[real_idx])
    return float(scores.mean())


def _concat(chunks: list[np.ndarray], indices: np.ndarray) -> np.ndarray:
    return np.concatenate([chunks[int(i)] for i in indices], axis=0)


def _subset(domain: DomainData, episode_ids: set[int]) -> DomainData:
    keep = [index for index, episode in enumerate(domain.episodes) if episode in episode_ids]
    if len(keep) != len(episode_ids):
        found = {domain.episodes[index] for index in keep}
        raise RuntimeError(f"Missing original episodes: {sorted(episode_ids.difference(found))}")

    def take(chunks: list[np.ndarray]) -> list[np.ndarray]:
        return [chunks[index] for index in keep]

    return DomainData(
        name=domain.name,
        episodes=[domain.episodes[index] for index in keep],
        embedding=take(domain.embedding),
        count=take(domain.count),
        conf=take(domain.conf),
        area=take(domain.area),
        cls=take(domain.cls),
        scalars={key: take(chunks) for key, chunks in domain.scalars.items()},
        dir_hist=take(domain.dir_hist),
    )


def _support() -> np.ndarray:
    payload = json.loads((ROOT / "results" / "phase3" / "metrics.json").read_text(encoding="utf-8"))
    return np.asarray(payload["sim_vs_real_class_support"], dtype=np.int16)


def _summarize(values: np.ndarray) -> dict[str, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return {"std": float("nan"), "p2_5": float("nan"), "p97_5": float("nan"), "n": 0}
    return {
        "std": float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        "p2_5": float(np.percentile(finite, 2.5)),
        "p97_5": float(np.percentile(finite, 97.5)),
        "n": int(finite.size),
    }


def _bootstrap(sim, real, support, n_boot: int, device: torch.device) -> dict[str, dict]:
    rng = np.random.default_rng(1)
    names: list[str] | None = None
    rows: list[list[float]] = []
    for draw in range(n_boot):
        sim_idx = rng.integers(0, len(sim), size=len(sim))
        real_idx = rng.integers(0, len(real), size=len(real))
        sim_embed = torch.from_numpy(_concat(sim.embedding, sim_idx)).to(device)
        real_embed = torch.from_numpy(_concat(real.embedding, real_idx)).to(device)
        values = all_distances(sim, real, sim_idx, real_idx, support, sim_embed, real_embed, kid_repeats=10, kid_seed=20_000 + draw)
        if names is None:
            names = list(values)
        rows.append([values[name] for name in names])
    matrix = np.asarray(rows, dtype=np.float64)
    assert names is not None
    return {name: _summarize(matrix[:, i]) for i, name in enumerate(names)}


def evaluate(n_boot: int) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    real = load_domain(FEATURES, "real")
    support = _support()
    probe = load_domain(PHASE5 / CONDITIONS[0].condition_id, "sim")
    original = _subset(load_domain(FEATURES, "sim"), set(probe.episodes))
    print(f"baseline episodes {original.episodes[0]}-{original.episodes[-1]} ({len(original)}), real episodes {len(real)}", flush=True)
    real_idx = np.arange(len(real))
    real_embed = np.concatenate(real.embedding, axis=0)
    baseline = all_distances(original, real, np.arange(len(original)), real_idx, support)
    baseline["appearance.kid"] = kid_point(np.concatenate(original.embedding, axis=0), real_embed, device)
    print("baseline KID", baseline["appearance.kid"], flush=True)
    del original
    rows: list[dict] = []

    for condition in CONDITIONS:
        folder = PHASE5 / condition.condition_id
        print("evaluate", condition.condition_id, flush=True)
        sim = load_domain(folder, "sim")
        sim_idx = np.arange(len(sim))
        point = all_distances(sim, real, sim_idx, real_idx, support)
        point["appearance.kid"] = kid_point(np.concatenate(sim.embedding, axis=0), real_embed, device)
        print("  KID", point["appearance.kid"], flush=True)
        boot = _bootstrap(sim, real, support, n_boot, device) if n_boot else {}
        for name, estimate in point.items():
            interval = boot.get(name, {"p2_5": float("nan"), "p97_5": float("nan"), "std": float("nan"), "n": 0})
            base = baseline[name]
            rows.append(
                {
                    "condition_id": condition.condition_id,
                    "factor": condition.factor,
                    "level": condition.level,
                    "intensity": condition.intensity,
                    "metric": name,
                    "unit": UNITS[name],
                    "point": estimate,
                    "baseline": base,
                    "delta": estimate - base,
                    "ci95_low": interval["p2_5"],
                    "ci95_high": interval["p97_5"],
                    "bootstrap_std": interval["std"],
                    "n_bootstrap": interval["n"],
                    "n_episodes": len(sim),
                }
            )

    OUT.mkdir(parents=True, exist_ok=True)
    payload = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rule": "one factor per condition; real images unchanged",
        "class_support": "fixed to the Phase 3 sim-vs-real union",
        "sim_episodes": probe.episodes,
        "n_real_episodes": len(real),
        "baseline_source": "original Phase 2 features on the same simulation episodes; all real episodes",
        "kid_point": "100 subsets of 1000, numpy seed 0, same kernel as Phase 2",
        "bootstrap": {"draws": n_boot, "seed": 1, "inner_kid_subsets": 10},
        "rows": rows,
    }
    (OUT / "metrics.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (OUT / "metrics.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", OUT / "metrics.csv")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Measure sim-to-real distances for each perturbation.")
    parser.add_argument("--bootstrap", type=int, default=200)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    evaluate(args.bootstrap)


if __name__ == "__main__":
    main()
