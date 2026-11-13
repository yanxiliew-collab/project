"""Synthetic image degradations for preprocessing experiments."""

from __future__ import annotations

from typing import Optional

import cv2
import numpy as np

from common.types import TrackFrame


def _odd_kernel(size: int) -> int:
    size = max(3, int(size))
    return size if size % 2 else size + 1


def _motion_kernel(length: int, angle: float) -> np.ndarray:
    kernel = np.zeros((length, length), dtype=np.float32)
    center = (length - 1) / 2.0
    line = cv2.getRotationMatrix2D((center, center), angle, 1.0)
    line[0, 2] += center - line[0, 0] * center - line[0, 1] * center
    line[1, 2] += center - line[1, 0] * center - line[1, 1] * center
    cv2.line(
        kernel,
        (0, int(round(center))),
        (length - 1, int(round(center))),
        1.0,
        1,
    )
    kernel = cv2.warpAffine(kernel, line, (length, length))
    total = float(kernel.sum())
    return kernel / total if total else np.eye(length, dtype=np.float32)[length // 2]


def degrade(
    img: np.ndarray,
    *,
    gamma: float = 1.0,
    brightness_scale: float = 1.0,
    gaussian_sigma: float = 0.0,
    motion_length: int = 0,
    motion_angle: float = 0.0,
    noise_std: float = 0.0,
    downsample_scale: float = 1.0,
    seed: Optional[int] = None,
) -> np.ndarray:
    """Apply composable synthetic degradations while preserving shape and dtype.

    ``gamma`` uses the standard power-law convention: values greater than one
    darken the image. ``downsample_scale`` must be in ``(0, 1]``.
    """

    if img.ndim not in (2, 3):
        raise ValueError("img must be a 2-D grayscale or 3-D color image")
    if gamma <= 0:
        raise ValueError("gamma must be positive")
    if brightness_scale < 0:
        raise ValueError("brightness_scale must be non-negative")
    if gaussian_sigma < 0 or motion_length < 0 or noise_std < 0:
        raise ValueError("degradation strengths must be non-negative")
    if not 0 < downsample_scale <= 1:
        raise ValueError("downsample_scale must be in (0, 1]")

    rng = np.random.default_rng(seed)
    result = img.astype(np.float32, copy=True)

    if gamma != 1.0:
        result = np.power(np.clip(result / 255.0, 0.0, 1.0), gamma) * 255.0
    if brightness_scale != 1.0:
        result *= brightness_scale
    if gaussian_sigma > 0:
        kernel = _odd_kernel(np.ceil(gaussian_sigma * 6).astype(int))
        result = cv2.GaussianBlur(result, (kernel, kernel), gaussian_sigma)
    if motion_length > 1:
        length = _odd_kernel(motion_length)
        result = cv2.filter2D(result, -1, _motion_kernel(length, motion_angle))
    if noise_std > 0:
        result += rng.normal(0.0, noise_std, result.shape).astype(np.float32)
    if downsample_scale < 1.0:
        height, width = result.shape[:2]
        small_size = (
            max(1, int(round(width * downsample_scale))),
            max(1, int(round(height * downsample_scale))),
        )
        result = cv2.resize(result, small_size, interpolation=cv2.INTER_AREA)
        result = cv2.resize(result, (width, height), interpolation=cv2.INTER_LINEAR)

    return np.clip(result, 0, 255).astype(np.uint8)


def make_track(
    img: np.ndarray, n: int, seed: Optional[int] = None
) -> tuple[list[TrackFrame], list[np.ndarray]]:
    """Create transformed, independently noisy frames and their affine matrices.

    Each returned matrix maps coordinates in ``img`` to coordinates in the
    corresponding ``TrackFrame.full_frame``.
    """

    if n < 0:
        raise ValueError("n must be non-negative")
    if img.ndim not in (2, 3):
        raise ValueError("img must be a 2-D grayscale or 3-D color image")

    rng = np.random.default_rng(seed)
    height, width = img.shape[:2]
    center = (width / 2.0, height / 2.0)
    frames: list[TrackFrame] = []
    transforms: list[np.ndarray] = []
    timestamp = 0.0

    for frame_idx in range(n):
        angle = float(rng.uniform(-3.0, 3.0))
        scale = float(rng.uniform(0.97, 1.03))
        matrix = cv2.getRotationMatrix2D(center, angle, scale)
        matrix[:, 2] += rng.uniform(-2.0, 2.0, size=2)
        transformed = cv2.warpAffine(
            img,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REFLECT,
        )
        transformed = degrade(
            transformed,
            gaussian_sigma=0.35,
            noise_std=2.0,
            seed=int(rng.integers(0, 2**32 - 1)),
        )
        timestamp += float(rng.uniform(0.08, 0.20))
        frames.append(
            TrackFrame(
                frame_idx=frame_idx,
                timestamp=timestamp,
                full_frame=transformed,
                bbox=(0, 0, width, height),
            )
        )
        transforms.append(matrix.astype(np.float32))

    return frames, transforms
