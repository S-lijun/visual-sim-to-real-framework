"""Shared settings for Phase 2 feature extraction.

The same values are applied to simulation and real RGB. Resolved copies are
written next to the saved features.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
FEATURES = ROOT / "features"


@dataclass
class ExtractConfig:
    real_dir: Path = DATA / "real_obses_15fps" / "obses_15fps"
    sim_dir: Path = DATA / "sim_obses_15fps" / "obses_15fps"
    out_dir: Path = FEATURES
    yolo_weights: Path = ROOT / "yolov8n.pt"
    yolo_conf: float = 0.25
    yolo_batch: int = 16
    midas_model: str = "DPT_Large"
    midas_batch: int = 8
    inception_batch: int = 32
    image_hw: tuple[int, int] = (480, 640)
    declared_fps: float = 15.0
    # Farneback parameters copied from the existing notebook.
    flow_pyr_scale: float = 0.5
    flow_levels: int = 3
    flow_winsize: int = 15
    flow_iterations: int = 3
    flow_poly_n: int = 5
    flow_poly_sigma: float = 1.2
    flow_motion_px: float = 1.0
    direction_bins: int = 36
    kid_subset_size: int = 1000
    kid_repeats: int = 100
    kid_degree: int = 3
    kid_seed: int = 0
    seed: int = 0

    def to_dict(self) -> dict:
        raw = asdict(self)
        return {key: str(value) if isinstance(value, Path) else value for key, value in raw.items()}


def domain_dir(config: ExtractConfig, domain: str) -> Path:
    if domain == "real":
        return config.real_dir
    if domain == "sim":
        return config.sim_dir
    raise ValueError(f"Unknown domain: {domain}")
