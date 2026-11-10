"""Extract appearance, semantic, geometry, and temporal features.

Example
-------
python -m phase2.extract --limit-episodes 1
python -m phase2.extract
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase2.appearance import AppearanceEncoder, rgb_stats
from phase2.config import ExtractConfig, domain_dir
from phase2.dataset import list_episodes, load_rgb
from phase2.geometry import VALUE_MEANING, MidasEncoder, disparity_features
from phase2.kid import write_kid
from phase2.semantic import YoloEncoder, class_table
from phase2.temporal import flow_features

MODULES = ("appearance", "semantic", "geometry", "temporal")


def _episode_path(out_dir: Path, domain: str, module: str, episode: int) -> Path:
    return out_dir / domain / module / f"episode_{episode:04d}.npz"


def _save(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.replace(path)


def _done(out_dir: Path, domain: str, episode: int, modules: tuple[str, ...]) -> bool:
    return all(_episode_path(out_dir, domain, module, episode).is_file() for module in modules)


def extract_features(config: ExtractConfig, domains: list[str], modules: list[str], limit: int | None) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)
    selected = tuple(modules)
    appearance = AppearanceEncoder(config, device) if "appearance" in selected else None
    semantic = None
    geometry = None
    config.out_dir.mkdir(parents=True, exist_ok=True)
    if "semantic" in selected:
        yolo_device = "0" if device.type == "cuda" else "cpu"
        semantic = YoloEncoder(config, yolo_device)
        class_table(semantic.names, config.out_dir / "yolo_classes.tsv")
    if "geometry" in selected:
        geometry = MidasEncoder(config, device)

    counts = {domain: {"episodes": 0, "frames": 0} for domain in domains}
    for domain in domains:
        episodes = list_episodes(domain_dir(config, domain), domain)
        if limit is not None:
            episodes = episodes[:limit]
        print(f"[{domain}] {len(episodes)} episodes")
        for ref in episodes:
            if _done(config.out_dir, domain, ref.episode, selected):
                print(f"  skip episode {ref.episode}")
                continue
            rgb = load_rgb(ref)
            if rgb.shape[1:3] != config.image_hw:
                raise ValueError(f"{ref.path} resolution {rgb.shape[1:3]} != {config.image_hw}")
            print(f"  episode {ref.episode}: {rgb.shape[0]} frames")
            if appearance is not None and not _episode_path(config.out_dir, domain, "appearance", ref.episode).is_file():
                embedding = appearance.embed(rgb)
                mean, std = rgb_stats(rgb)
                _save(
                    _episode_path(config.out_dir, domain, "appearance", ref.episode),
                    embedding=embedding,
                    rgb_mean=mean,
                    rgb_std=std,
                    frame_index=np.arange(rgb.shape[0], dtype=np.int32),
                )
            if semantic is not None and not _episode_path(config.out_dir, domain, "semantic", ref.episode).is_file():
                detected = semantic.detect(rgb)
                _save(_episode_path(config.out_dir, domain, "semantic", ref.episode), **detected)
            if geometry is not None and not _episode_path(config.out_dir, domain, "geometry", ref.episode).is_file():
                disparity = geometry.disparity(rgb)
                _save(
                    _episode_path(config.out_dir, domain, "geometry", ref.episode),
                    frame_index=np.arange(rgb.shape[0], dtype=np.int32),
                    **disparity_features(disparity),
                )
            if "temporal" in selected and not _episode_path(config.out_dir, domain, "temporal", ref.episode).is_file():
                if rgb.shape[0] < 2:
                    _save(
                        _episode_path(config.out_dir, domain, "temporal", ref.episode),
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
                    _save(
                        _episode_path(config.out_dir, domain, "temporal", ref.episode),
                        **flow_features(rgb, config),
                    )
            counts[domain]["episodes"] += 1
            counts[domain]["frames"] += int(rgb.shape[0])

    meta = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "device": str(device),
        "modules": list(selected),
        "limit_episodes": limit,
        "counts_this_run": counts,
        "geometry_value_meaning": VALUE_MEANING,
        "geometry_normalization": "none_raw_midas_output",
        "temporal_scope": "consecutive frames inside one episode",
        "semantic_empty_policy": "count is 0; confidence and boxes are omitted",
        "appearance_preprocess": "bilinear 299x299, ImageNet mean/std, no center crop",
        "config": config.to_dict(),
        "torch": torch.__version__,
    }
    config.out_dir.mkdir(parents=True, exist_ok=True)
    (config.out_dir / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    if limit is None and "appearance" in selected and set(domains) >= {"sim", "real"}:
        payload = write_kid(config.out_dir, config)
        print(f"KID mean {payload['mean']:.6f} std {payload['std']:.6f} (lower is closer)")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Extract sim/real visual features.")
    parser.add_argument("--domains", nargs="+", default=["sim", "real"], choices=["sim", "real"])
    parser.add_argument("--modules", nargs="+", default=list(MODULES), choices=list(MODULES))
    parser.add_argument("--limit-episodes", type=int, default=None)
    parser.add_argument("--out", type=Path, default=None)
    return parser


def main(argv: list[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    config = ExtractConfig()
    if args.out is not None:
        config.out_dir = args.out
    extract_features(config, args.domains, args.modules, args.limit_episodes)


if __name__ == "__main__":
    main()
