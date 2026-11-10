"""Load RGB episodes. One ``episode_*.pth`` file is one trajectory."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

_EP_RE = re.compile(r"episode_(\d+)\.pth$")


@dataclass(frozen=True)
class EpisodeRef:
    domain: str
    episode: int
    path: Path


def list_episodes(folder: Path, domain: str) -> list[EpisodeRef]:
    files = [path for path in folder.glob("episode_*.pth") if path.is_file()]

    def episode_id(path: Path) -> int:
        match = _EP_RE.search(path.name)
        if match is None:
            raise ValueError(f"Unrecognized episode name: {path.name}")
        return int(match.group(1))

    return [EpisodeRef(domain, episode_id(path), path) for path in sorted(files, key=episode_id)]


def load_rgb(episode: EpisodeRef) -> np.ndarray:
    """Return uint8 RGB with shape (T, H, W, 3)."""
    tensor = torch.load(episode.path, map_location="cpu", weights_only=False)
    if not torch.is_tensor(tensor):
        raise TypeError(f"{episode.path} is {type(tensor).__name__}, expected a tensor")
    if tensor.ndim != 4 or tensor.shape[-1] != 3 or tensor.dtype != torch.uint8:
        raise ValueError(f"{episode.path} has shape {tuple(tensor.shape)} dtype {tensor.dtype}")
    return tensor.numpy()
