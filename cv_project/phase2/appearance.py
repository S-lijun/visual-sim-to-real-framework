"""InceptionV3 embeddings and per-frame RGB statistics.

Preprocessing is the same for both domains: bilinear resize to 299x299, then
ImageNet mean/std from ``Inception_V3_Weights.DEFAULT``. The 2048-d vector is
the pool feature used later for KID. Frames are not center-cropped, so the
full field of view stays in the embedding.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torchvision.models import Inception_V3_Weights, inception_v3

from .config import ExtractConfig

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class AppearanceEncoder:
    name = "inception_v3"
    weights = "Inception_V3_Weights.DEFAULT"
    embedding_dim = 2048

    def __init__(self, config: ExtractConfig, device: torch.device):
        weights = Inception_V3_Weights.DEFAULT
        model = inception_v3(weights=weights, transform_input=False)
        model.fc = torch.nn.Identity()
        model.eval()
        self.model = model.to(device)
        self.device = device
        self.batch = config.inception_batch
        self.mean = torch.tensor(IMAGENET_MEAN, device=device).view(1, 3, 1, 1)
        self.std = torch.tensor(IMAGENET_STD, device=device).view(1, 3, 1, 1)

    @torch.inference_mode()
    def embed(self, rgb: np.ndarray) -> np.ndarray:
        outputs = []
        for start in range(0, rgb.shape[0], self.batch):
            batch = torch.from_numpy(rgb[start : start + self.batch]).to(self.device)
            batch = batch.permute(0, 3, 1, 2).float().div(255.0)
            batch = F.interpolate(batch, size=(299, 299), mode="bilinear", align_corners=False)
            batch = (batch - self.mean) / self.std
            feat = self.model(batch)
            if isinstance(feat, (tuple, list)):
                feat = feat[0]
            outputs.append(feat.detach().float().cpu().numpy())
        return np.concatenate(outputs, axis=0).astype(np.float32)


def rgb_stats(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    flat = rgb.astype(np.float32).reshape(rgb.shape[0], -1, 3)
    return flat.mean(axis=1).astype(np.float32), flat.std(axis=1).astype(np.float32)
