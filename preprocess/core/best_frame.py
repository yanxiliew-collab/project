"""Best-frame selection based on denoised image sharpness."""

from typing import Sequence

import cv2
import numpy as np

from common.types import PreprocessResult, TrackFrame


def _roi(frame: TrackFrame) -> np.ndarray:
    x1, y1, x2, y2 = frame.bbox
    return frame.full_frame[y1:y2, x1:x2]


def sharpness_score(image: np.ndarray) -> float:
    """Score detail after denoising using gradients and Laplacian energy."""

    if image.size == 0:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
    denoised = cv2.GaussianBlur(gray, (3, 3), 0)
    gradient_x = cv2.Sobel(denoised, cv2.CV_32F, 1, 0)
    gradient_y = cv2.Sobel(denoised, cv2.CV_32F, 0, 1)
    gradient_energy = float(np.mean(gradient_x * gradient_x + gradient_y * gradient_y))
    laplacian_energy = float(np.var(cv2.Laplacian(denoised, cv2.CV_32F)))
    return gradient_energy + 0.5 * laplacian_energy


def best_frame(track_frames: Sequence[TrackFrame]) -> PreprocessResult:
    """Return the ROI from the frame with the highest denoised sharpness."""

    if not track_frames:
        return PreprocessResult(
            image=None,
            diagnostics={"mode": "best_frame", "empty_input": True, "frame_scores": []},
        )

    scores = [sharpness_score(_roi(frame)) for frame in track_frames]
    selected = int(np.argmax(scores))
    frame = track_frames[selected]
    return PreprocessResult(
        image=_roi(frame).copy(),
        diagnostics={
            "mode": "best_frame",
            "frame_idx": frame.frame_idx,
            "frame_scores": [
                {"frame_idx": item.frame_idx, "score": score}
                for item, score in zip(track_frames, scores)
            ],
        },
    )
