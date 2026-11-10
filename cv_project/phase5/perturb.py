"""One-factor transforms for simulation RGB.

Each condition changes exactly one factor. Noise, blur, brightness, and color
are never applied together. Real images are not handled here.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

NOISE_SEED = 5


@dataclass(frozen=True)
class Condition:
    condition_id: str
    factor: str
    level: str
    intensity: float
    params: dict[str, float]

    def as_dict(self) -> dict:
        return {
            "condition_id": self.condition_id,
            "factor": self.factor,
            "level": self.level,
            "intensity": self.intensity,
            "params": self.params,
        }


def _kernel_size(sigma: float) -> int:
    return int(2 * np.ceil(3.0 * sigma) + 1)


CONDITIONS: tuple[Condition, ...] = (
    Condition("noise_s05", "noise", "5", 5.0, {"sigma": 5.0}),
    Condition("noise_s15", "noise", "15", 15.0, {"sigma": 15.0}),
    Condition("noise_s30", "noise", "30", 30.0, {"sigma": 30.0}),
    Condition("blur_s0p6", "blur", "0.6", 0.6, {"sigma": 0.6, "kernel": _kernel_size(0.6)}),
    Condition("blur_s1p2", "blur", "1.2", 1.2, {"sigma": 1.2, "kernel": _kernel_size(1.2)}),
    Condition("blur_s2p4", "blur", "2.4", 2.4, {"sigma": 2.4, "kernel": _kernel_size(2.4)}),
    Condition("bright_m30", "brightness", "-30", -30.0, {"delta": -30.0}),
    Condition("bright_m15", "brightness", "-15", -15.0, {"delta": -15.0}),
    Condition("bright_p15", "brightness", "+15", 15.0, {"delta": 15.0}),
    Condition("bright_p30", "brightness", "+30", 30.0, {"delta": 30.0}),
    Condition("color_m32", "color", "-32", -32.0, {"delta": -32.0}),
    Condition("color_m16", "color", "-16", -16.0, {"delta": -16.0}),
    Condition("color_m08", "color", "-8", -8.0, {"delta": -8.0}),
    Condition("color_p08", "color", "+8", 8.0, {"delta": 8.0}),
    Condition("color_p16", "color", "+16", 16.0, {"delta": 16.0}),
    Condition("color_p32", "color", "+32", 32.0, {"delta": 32.0}),
)


def _noise(rgb: np.ndarray, sigma: float, episode_id: int) -> np.ndarray:
    rng = np.random.default_rng(np.random.SeedSequence([NOISE_SEED, int(episode_id)]))
    noise = rng.standard_normal(rgb.shape).astype(np.float32) * np.float32(sigma)
    shifted = rgb.astype(np.float32) + noise
    return np.clip(np.rint(shifted), 0, 255).astype(np.uint8)


def _blur(rgb: np.ndarray, sigma: float) -> np.ndarray:
    kernel = _kernel_size(sigma)
    out = np.empty_like(rgb)
    for index in range(rgb.shape[0]):
        out[index] = cv2.GaussianBlur(rgb[index], (kernel, kernel), sigmaX=sigma, sigmaY=sigma)
    return out


def _brightness(rgb: np.ndarray, delta: float) -> np.ndarray:
    shifted = rgb.astype(np.int16) + int(delta)
    return np.clip(shifted, 0, 255).astype(np.uint8)


def _color(rgb: np.ndarray, delta: float) -> np.ndarray:
    """Signed color temperature. Positive delta raises R and lowers B."""
    shifted = rgb.astype(np.int16).copy()
    step = int(delta)
    shifted[..., 0] = np.clip(shifted[..., 0] + step, 0, 255)
    shifted[..., 2] = np.clip(shifted[..., 2] - step, 0, 255)
    return shifted.astype(np.uint8)


def apply_condition(rgb: np.ndarray, condition: Condition, episode_id: int) -> np.ndarray:
    if rgb.dtype != np.uint8 or rgb.ndim != 4 or rgb.shape[-1] != 3:
        raise ValueError(f"Expected uint8 RGB (T, H, W, 3), got {rgb.dtype} {rgb.shape}")
    if condition.factor == "noise":
        return _noise(rgb, condition.params["sigma"], episode_id)
    if condition.factor == "blur":
        return _blur(rgb, condition.params["sigma"])
    if condition.factor == "brightness":
        return _brightness(rgb, condition.params["delta"])
    if condition.factor == "color":
        return _color(rgb, condition.params["delta"])
    raise ValueError(f"Unknown factor: {condition.factor}")


def by_id(condition_id: str) -> Condition:
    for condition in CONDITIONS:
        if condition.condition_id == condition_id:
            return condition
    raise KeyError(condition_id)
