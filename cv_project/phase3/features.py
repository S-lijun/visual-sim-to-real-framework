"""Load cached Phase 2 features, one list entry per trajectory."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

MODULES = ("appearance", "semantic", "geometry", "temporal")
SCALAR_KEYS = {
    "geometry": (
        "disparity_mean",
        "disparity_std",
        "disparity_p05",
        "disparity_p50",
        "disparity_p95",
        "grad_mean",
        "grad_p95",
    ),
    "temporal": ("mag_mean", "mag_std", "mag_p95", "motion_frac"),
}


@dataclass
class DomainData:
    name: str
    episodes: list[int]
    embedding: list[np.ndarray]
    count: list[np.ndarray]
    conf: list[np.ndarray]
    area: list[np.ndarray]
    cls: list[np.ndarray]
    scalars: dict[str, list[np.ndarray]]
    dir_hist: list[np.ndarray]

    def __len__(self) -> int:
        return len(self.episodes)


def _episode_id(path: Path) -> int:
    return int(path.stem.split("_")[1])


def load_domain(features_root: Path, domain: str) -> DomainData:
    base = features_root / domain
    appearance = sorted((base / "appearance").glob("episode_*.npz"))
    if len(appearance) == 0:
        raise FileNotFoundError(base / "appearance")
    episodes = [_episode_id(path) for path in appearance]
    embedding: list[np.ndarray] = []
    count: list[np.ndarray] = []
    conf: list[np.ndarray] = []
    area: list[np.ndarray] = []
    cls: list[np.ndarray] = []
    scalars: dict[str, list[np.ndarray]] = {key: [] for keys in SCALAR_KEYS.values() for key in keys}
    dir_hist: list[np.ndarray] = []

    for path in appearance:
        episode = _episode_id(path)
        with np.load(path) as data:
            embedding.append(np.asarray(data["embedding"], dtype=np.float32))
        sem_path = base / "semantic" / f"episode_{episode:04d}.npz"
        geo_path = base / "geometry" / f"episode_{episode:04d}.npz"
        flow_path = base / "temporal" / f"episode_{episode:04d}.npz"
        with np.load(sem_path) as data:
            count.append(np.asarray(data["count"], dtype=np.float64))
            conf.append(np.asarray(data["det_conf"], dtype=np.float64))
            area.append(np.asarray(data["det_norm_area"], dtype=np.float64))
            cls.append(np.asarray(data["det_cls"], dtype=np.int16))
        with np.load(geo_path) as data:
            for key in SCALAR_KEYS["geometry"]:
                scalars[key].append(np.asarray(data[key], dtype=np.float64))
        with np.load(flow_path) as data:
            for key in SCALAR_KEYS["temporal"]:
                scalars[key].append(np.asarray(data[key], dtype=np.float64))
            dir_hist.append(np.asarray(data["dir_hist"], dtype=np.float64))

    return DomainData(
        name=domain,
        episodes=episodes,
        embedding=embedding,
        count=count,
        conf=conf,
        area=area,
        cls=cls,
        scalars=scalars,
        dir_hist=dir_hist,
    )


def class_support(*domains: DomainData) -> np.ndarray:
    labels: set[int] = set()
    for domain in domains:
        for episode in domain.cls:
            labels.update(int(value) for value in episode.tolist())
    return np.asarray(sorted(labels), dtype=np.int16)
