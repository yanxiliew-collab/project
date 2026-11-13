"""Contrast and brightness enhancement helpers."""

import cv2
import numpy as np


def gamma_correct(image: np.ndarray, gamma: float) -> np.ndarray:
    """Apply power-law correction, preserving the input shape and uint8 type."""

    if gamma <= 0:
        raise ValueError("gamma must be positive")
    values = np.asarray(image, dtype=np.float32) / 255.0
    corrected = np.power(np.clip(values, 0.0, 1.0), 1.0 / gamma) * 255.0
    return np.clip(corrected, 0, 255).astype(np.uint8)


def clahe_enhance(
    image: np.ndarray, clip_limit: float = 2.0, tile_grid: tuple[int, int] = (8, 8)
) -> np.ndarray:
    """Apply CLAHE only to the L channel in CIELAB color space."""

    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError("clahe_enhance expects a BGR color image")
    if clip_limit <= 0 or tile_grid[0] <= 0 or tile_grid[1] <= 0:
        raise ValueError("CLAHE parameters must be positive")
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def auto_gamma(image: np.ndarray, target_mean: float = 0.5) -> np.ndarray:
    """Choose gamma so the mean luminance approaches ``target_mean``."""

    if not 0 < target_mean < 1:
        raise ValueError("target_mean must be in (0, 1)")
    if image.size == 0:
        return image.copy()
    if image.ndim == 3:
        luminance = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        luminance = image
    mean = float(np.mean(luminance.astype(np.float32) / 255.0))
    if mean <= 0:
        return image.copy()
    gamma = np.log(mean) / np.log(target_mean)
    return gamma_correct(image, max(gamma, 1e-3))
