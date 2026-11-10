"""Extract features for each single-factor simulation perturbation.

Real images are not read or written. Original simulation files are only read.
An existing episode file is skipped, so the command can be resumed.

Example
-------
python -m phase5.extract --limit-episodes 1
python -m phase5.extract
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase2.appearance import AppearanceEncoder, rgb_stats
from phase2.config import ExtractConfig, domain_dir
from phase2.dataset import list_episodes, load_rgb
from phase2.geometry import MidasEncoder, disparity_features
from phase2.semantic import YoloEncoder
from phase2.temporal import flow_features
from phase5.perturb import CONDITIONS, Condition, apply_condition

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "features" / "phase5"
MODULES = ("appearance", "semantic", "geometry", "temporal")


def _episode_path(condition_dir: Path, module: str, episode: int) -> Path:
    return condition_dir / "sim" / module / f"episode_{episode:04d}.npz"


def _save(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _missing(condition_dir: Path, episode: int) -> list[str]:
    return [module for module in MODULES if not _episode_path(condition_dir, module, episode).is_file()]


def _write_condition(condition_dir: Path, rgb: np.ndarray, episode: int, modules: list[str], models: dict) -> None:
    if "appearance" in modules:
        embedding = models["appearance"].embed(rgb)
        mean, std = rgb_stats(rgb)
        _save(
            _episode_path(condition_dir, "appearance", episode),
            embedding=embedding,
            rgb_mean=mean,
            rgb_std=std,
            frame_index=np.arange(rgb.shape[0], dtype=np.int32),
        )
    if "semantic" in modules:
        _save(_episode_path(condition_dir, "semantic", episode), **models["semantic"].detect(rgb))
    if "geometry" in modules:
        disparity = models["geometry"].disparity(rgb)
        _save(
            _episode_path(condition_dir, "geometry", episode),
            frame_index=np.arange(rgb.shape[0], dtype=np.int32),
            **disparity_features(disparity),
        )
    if "temporal" in modules:
        if rgb.shape[0] < 2:
            config = models["config"]
            _save(
                _episode_path(condition_dir, "temporal", episode),
                pair_src=np.zeros((0,), dtype=np.int32),
                mag_mean=np.zeros((0,), dtype=np.float32),
                mag_std=np.zeros((0,), dtype=np.float32),
                mag_p95=np.zeros((0,), dtype=np.float32),
                motion_frac=np.zeros((0,), dtype=np.float32),
                dir_cos=np.zeros((0,), dtype=np.float32),
                dir_sin=np.zeros((0,), dtype=np.float32),
                dir_hist=np.zeros((0, config.direction_bins), dtype=np.float32),
            )
        else:
            _save(_episode_path(condition_dir, "temporal", episode), **flow_features(rgb, models["config"]))


def extract(conditions: list[Condition], limit: int | None) -> None:
    config = ExtractConfig()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    yolo_device = "0" if device.type == "cuda" else "cpu"
    models = {
        "config": config,
        "appearance": AppearanceEncoder(config, device),
        "semantic": YoloEncoder(config, yolo_device),
        "geometry": MidasEncoder(config, device),
    }
    episodes = list_episodes(domain_dir(config, "sim"), "sim")
    if limit is not None:
        episodes = episodes[:limit]
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "rule": "each condition changes one factor only",
        "real_images": "not modified and not read",
        "conditions": [condition.as_dict() for condition in conditions],
    }
    (OUT / "conditions.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    for ref in episodes:
        pending = [(condition, _missing(OUT / condition.condition_id, ref.episode)) for condition in conditions]
        pending = [(condition, modules) for condition, modules in pending if modules]
        if not pending:
            print(f"skip episode {ref.episode}")
            continue
        rgb = load_rgb(ref)
        if rgb.shape[1:3] != config.image_hw:
            raise ValueError(f"{ref.path} resolution {rgb.shape[1:3]} != {config.image_hw}")
        print(f"episode {ref.episode}: {rgb.shape[0]} frames, {len(pending)} conditions", flush=True)
        for condition, modules in pending:
            transformed = apply_condition(rgb, condition, ref.episode)
            started = time.perf_counter()
            print(f"  {condition.condition_id} {modules}", flush=True)
            _write_condition(OUT / condition.condition_id, transformed, ref.episode, modules, models)
            print(f"    finished in {time.perf_counter() - started:.1f}s", flush=True)
            del transformed
    print("done")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract single-factor perturbation features.")
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--conditions", nargs="*", default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    selected = CONDITIONS
    if args.conditions:
        wanted = set(args.conditions)
        selected = tuple(condition for condition in CONDITIONS if condition.condition_id in wanted)
        missing = wanted.difference(condition.condition_id for condition in selected)
        if missing:
            raise SystemExit(f"Unknown conditions: {sorted(missing)}")
    extract(list(selected), args.limit_episodes)


if __name__ == "__main__":
    main()
