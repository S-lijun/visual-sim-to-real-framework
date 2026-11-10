"""Save one labeled contact sheet of every single-factor level.

Example
-------
python -m phase5.preview
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

if __package__ is None:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from phase2.config import ExtractConfig, domain_dir
from phase2.dataset import list_episodes, load_rgb
from phase5.perturb import CONDITIONS, apply_condition

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results" / "phase5" / "preview_episode0.png"


def main() -> None:
    config = ExtractConfig()
    episode = list_episodes(domain_dir(config, "sim"), "sim")[0]
    rgb = load_rgb(episode)
    frame_index = 20 if rgb.shape[0] > 20 else 0
    original = rgb[frame_index]
    panels = [("original", original)]
    for condition in CONDITIONS:
        transformed = apply_condition(rgb[frame_index : frame_index + 1], condition, episode.episode)
        panels.append((f"{condition.factor} {condition.level}", transformed[0]))

    columns = 5
    rows = int(np.ceil(len(panels) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(columns * 2.5, rows * 2.3))
    for ax, (title, image) in zip(axes.ravel(), panels):
        ax.imshow(image)
        ax.set_title(title, fontsize=8)
        ax.axis("off")
        mean = image.astype(np.float32).mean(axis=(0, 1))
        print(f"{title}\tR {mean[0]:.1f}\tG {mean[1]:.1f}\tB {mean[2]:.1f}")
    for ax in axes.ravel()[len(panels) :]:
        ax.axis("off")
    fig.suptitle(f"Simulation episode {episode.episode}, frame {frame_index}. One factor per panel.", fontsize=11)
    fig.tight_layout()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, dpi=140)
    plt.close(fig)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
