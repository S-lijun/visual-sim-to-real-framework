"""MiDaS relative-depth statistics.

DPT models from intel-isl/MiDaS emit disparity-like values: a larger raw value
means closer, not a larger metric distance. Statistics are computed on that raw
map. Frames are not shifted or scaled to zero mean and unit variance, so mean
and spread remain comparable across domains.

The transform is the hub ``dpt_transform`` used by the existing notebook. Input
here is already RGB, so it is not converted from BGR again.
"""

from __future__ import annotations

import cv2
import numpy as np
import torch

from .config import ExtractConfig

VALUE_MEANING = "midas_dpt_disparity_larger_is_closer"


class MidasEncoder:
    def __init__(self, config: ExtractConfig, device: torch.device):
        model = torch.hub.load("intel-isl/MiDaS", config.midas_model, trust_repo=True)
        transforms = torch.hub.load("intel-isl/MiDaS", "transforms", trust_repo=True)
        model.eval()
        self.model = model.to(device)
        self.transform = transforms.dpt_transform
        self.device = device
        self.batch = config.midas_batch
        self.model_type = config.midas_model

    @torch.inference_mode()
    def disparity(self, rgb: np.ndarray) -> np.ndarray:
        maps = []
        height, width = rgb.shape[1:3]
        for start in range(0, rgb.shape[0], self.batch):
            chunk = rgb[start : start + self.batch]
            batches = [self.transform(frame).to(self.device) for frame in chunk]
            batch = torch.cat(batches, dim=0)
            pred = self.model(batch)
            if pred.ndim == 3:
                pred = pred.unsqueeze(1)
            pred = torch.nn.functional.interpolate(
                pred,
                size=(height, width),
                mode="bicubic",
                align_corners=False,
            ).squeeze(1)
            maps.append(pred.detach().float().cpu().numpy())
        return np.concatenate(maps, axis=0).astype(np.float32)


def disparity_features(disparity: np.ndarray) -> dict[str, np.ndarray]:
    flat = disparity.reshape(disparity.shape[0], -1)
    percentiles = np.percentile(flat, [5, 25, 50, 75, 95], axis=1).astype(np.float32)
    grad_mean = np.empty(disparity.shape[0], dtype=np.float32)
    grad_p95 = np.empty(disparity.shape[0], dtype=np.float32)
    for index, frame in enumerate(disparity):
        gx = cv2.Sobel(frame, cv2.CV_32F, 1, 0, ksize=3)
        gy = cv2.Sobel(frame, cv2.CV_32F, 0, 1, ksize=3)
        mag = np.sqrt(gx * gx + gy * gy)
        grad_mean[index] = float(mag.mean())
        grad_p95[index] = float(np.percentile(mag, 95))
    return {
        "disparity_mean": flat.mean(axis=1).astype(np.float32),
        "disparity_std": flat.std(axis=1).astype(np.float32),
        "disparity_p05": percentiles[0],
        "disparity_p25": percentiles[1],
        "disparity_p50": percentiles[2],
        "disparity_p75": percentiles[3],
        "disparity_p95": percentiles[4],
        "grad_mean": grad_mean,
        "grad_p95": grad_p95,
    }
