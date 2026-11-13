"""ECC alignment and multi-frame image fusion."""

from time import perf_counter
from typing import Sequence

import cv2
import numpy as np

from common.types import PreprocessResult, TrackFrame
from preprocess.config import PreprocessConfig
from preprocess.core.best_frame import _roi, sharpness_score


def _expanded_bbox(
    bbox: tuple[int, int, int, int], shape: tuple[int, ...], fraction: float = 0.2
) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    x1, y1, x2, y2 = bbox
    expand_x = (x2 - x1) * fraction
    expand_y = (y2 - y1) * fraction
    return (
        max(0, int(np.floor(x1 - expand_x))),
        max(0, int(np.floor(y1 - expand_y))),
        min(width, int(np.ceil(x2 + expand_x))),
        min(height, int(np.ceil(y2 + expand_y))),
    )


def _gray(image: np.ndarray) -> np.ndarray:
    if image.ndim == 3:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return image.astype(np.float32) / 255.0


def _alignment_mask(
    bbox: tuple[int, int, int, int], shape: tuple[int, ...]
) -> np.ndarray:
    x1, y1, x2, y2 = _expanded_bbox(bbox, shape)
    mask = np.zeros(shape[:2], dtype=np.uint8)
    mask[y1:y2, x1:x2] = 255
    return mask


def _align(
    reference: np.ndarray, candidate: np.ndarray, mask: np.ndarray
) -> tuple[np.ndarray, float]:
    warp = np.eye(3, dtype=np.float32)
    criteria = (
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        100,
        1e-5,
    )
    correlation, warp = cv2.findTransformECC(
        _gray(reference),
        _gray(candidate),
        warp,
        cv2.MOTION_HOMOGRAPHY,
        criteria,
        mask,
        5,
    )
    aligned = cv2.warpPerspective(
        candidate,
        warp,
        (reference.shape[1], reference.shape[0]),
        flags=cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP,
        borderMode=cv2.BORDER_REFLECT,
    )
    return aligned, float(correlation)


def fuse(
    track_frames: Sequence[TrackFrame], cfg: PreprocessConfig
) -> PreprocessResult:
    """Align selected frames to the clearest frame and fuse their ROIs."""

    started = perf_counter()
    if not track_frames:
        return PreprocessResult(
            image=None,
            diagnostics={"mode": "fusion", "empty_input": True, "accepted": []},
        )

    scored = sorted(
        ((sharpness_score(_roi(frame)), index, frame) for index, frame in enumerate(track_frames)),
        reverse=True,
        key=lambda item: item[0],
    )
    _, reference_index, reference = scored[0]
    reference_image = reference.full_frame
    max_frames = max(1, int(cfg.max_frames))
    selected = scored[:max_frames]
    mask = _alignment_mask(reference.bbox, reference_image.shape)
    accepted_images = [_roi(reference)]
    accepted = [{"frame_idx": reference.frame_idx, "correlation": 1.0}]
    rejected: list[dict[str, object]] = []

    for _, index, frame in selected:
        if index == reference_index:
            continue
        if (perf_counter() - started) * 1000.0 >= cfg.time_budget_ms:
            rejected.append({"frame_idx": frame.frame_idx, "reason": "time_budget"})
            continue
        try:
            aligned, correlation = _align(
                reference_image, frame.full_frame, mask
            )
        except cv2.error as exc:
            rejected.append(
                {"frame_idx": frame.frame_idx, "reason": f"ECC failed: {exc}"}
            )
            continue
        if correlation < cfg.ecc_min_correlation:
            rejected.append(
                {
                    "frame_idx": frame.frame_idx,
                    "reason": "low_correlation",
                    "correlation": correlation,
                }
            )
            continue
        x1, y1, x2, y2 = reference.bbox
        accepted_images.append(aligned[y1:y2, x1:x2])
        accepted.append({"frame_idx": frame.frame_idx, "correlation": correlation})

    if len(accepted_images) == 1 and len(track_frames) > 1:
        return PreprocessResult(
            image=None,
            diagnostics={
                "mode": "fusion",
                "fallback_best_frame": True,
                "reference_frame_idx": reference.frame_idx,
                "accepted": accepted,
                "rejected": rejected,
                "processed_frames": len(accepted),
            },
        )

    stack = np.stack(accepted_images).astype(np.float32)
    if cfg.fusion_method == "mean":
        image = np.mean(stack, axis=0)
    elif cfg.fusion_method == "median":
        image = np.median(stack, axis=0)
    else:
        raise ValueError(f"unsupported fusion method: {cfg.fusion_method!r}")
    return PreprocessResult(
        image=np.clip(image, 0, 255).astype(np.uint8),
        diagnostics={
            "mode": "fusion",
            "reference_frame_idx": reference.frame_idx,
            "accepted": accepted,
            "rejected": rejected,
            "processed_frames": len(accepted),
        },
    )
