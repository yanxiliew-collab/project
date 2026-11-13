import cv2
import numpy as np

from common.types import TrackFrame
from preprocess.config import PreprocessConfig
from preprocess.core.fusion import fuse
from preprocess.core.pipeline import preprocess


def _reference() -> np.ndarray:
    image = np.zeros((96, 128, 3), dtype=np.uint8)
    cv2.rectangle(image, (12, 12), (112, 80), (180, 120, 60), -1)
    cv2.circle(image, (76, 48), 16, (30, 220, 240), -1)
    cv2.putText(image, "OCR", (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    return image


def _track(image: np.ndarray, shifts: list[tuple[int, int]]) -> list[TrackFrame]:
    rng = np.random.default_rng(42)
    frames = []
    for index, (dx, dy) in enumerate(shifts):
        matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        frame = cv2.warpAffine(
            image, matrix, (image.shape[1], image.shape[0]), borderMode=cv2.BORDER_REFLECT
        )
        frame = cv2.GaussianBlur(frame, (5, 5), 1.0)
        frame = np.clip(
            frame.astype(np.int16) + rng.normal(0, 5, frame.shape), 0, 255
        ).astype(np.uint8)
        frames.append(TrackFrame(index, index * 0.1, frame, (8, 8, 116, 88)))
    return frames


def test_alignment_improves_fusion_psnr() -> None:
    reference = _reference()
    frames = _track(reference, [(0, 0), (3, -2), (-2, 2)])
    cfg = PreprocessConfig(mode="fusion", max_frames=3, time_budget_ms=500)

    result = fuse(frames, cfg)
    roi = reference[8:88, 8:116]
    direct_mean = np.mean(
        [frame.full_frame[8:88, 8:116] for frame in frames], axis=0
    ).astype(np.uint8)

    assert result.image is not None
    assert cv2.PSNR(result.image, roi) > cv2.PSNR(direct_mean, roi)
    assert cv2.PSNR(result.image, roi) > max(
        cv2.PSNR(frame.full_frame[8:88, 8:116], roi) for frame in frames
    )


def test_unrelated_frame_is_rejected() -> None:
    reference = _reference()
    frames = _track(reference, [(0, 0), (2, 1)])
    unrelated = np.zeros_like(reference)
    unrelated[:, :] = (10, 10, 10)
    frames.append(TrackFrame(99, 1.0, unrelated, (8, 8, 116, 88)))

    result = fuse(
        frames,
        PreprocessConfig(mode="fusion", max_frames=3, ecc_min_correlation=0.9),
    )

    rejected_ids = {item["frame_idx"] for item in result.diagnostics["rejected"]}
    assert 99 in rejected_ids


def test_pipeline_connects_fusion_mode() -> None:
    frames = _track(_reference(), [(0, 0), (2, -1)])

    result = preprocess(
        frames,
        PreprocessConfig(mode="fusion", max_frames=2, time_budget_ms=500),
    )

    assert result.image is not None
    assert result.image.dtype == np.uint8
    assert result.diagnostics["mode"] == "fusion"
