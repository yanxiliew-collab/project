import cv2
import numpy as np

from common.types import TrackFrame
from preprocess.config import PreprocessConfig
from preprocess.core import pipeline
from preprocess.core.pipeline import preprocess


def _frames() -> list[TrackFrame]:
    first = np.zeros((8, 10, 3), dtype=np.uint8)
    second = np.full((8, 10, 3), 127, dtype=np.uint8)
    return [
        TrackFrame(0, 0.0, first, (1, 2, 6, 7)),
        TrackFrame(1, 0.1, second, (2, 1, 8, 6)),
    ]


def test_passthrough_returns_last_frame_roi_shape_and_dtype() -> None:
    result = preprocess(_frames(), PreprocessConfig())

    assert result.image is not None
    assert result.image.shape == (5, 6, 3)
    assert result.image.dtype == np.uint8
    assert np.all(result.image == 127)
    assert result.diagnostics["mode"] == "passthrough"
    assert "elapsed_ms" in result.diagnostics


def test_empty_input_returns_empty_result() -> None:
    result = preprocess([], PreprocessConfig())

    assert result.image is None
    assert result.diagnostics["empty_input"] is True
    assert "elapsed_ms" in result.diagnostics


def test_exception_falls_back_to_passthrough(monkeypatch) -> None:
    frames = _frames()

    def fail(_image, _clip_limit, _tile_grid):
        raise ValueError("synthetic enhancement failure")

    monkeypatch.setattr(pipeline, "clahe_enhance", fail)
    result = preprocess(frames, PreprocessConfig(mode="enhance"))

    assert result.image is not None
    assert result.image.shape == (5, 6, 3)
    assert result.image.dtype == np.uint8
    assert result.diagnostics["fallback"] is True
    assert "ValueError: synthetic enhancement failure" in result.diagnostics["error"]
    assert "elapsed_ms" in result.diagnostics


def test_best_frame_reports_scores_and_selects_sharper_frame() -> None:
    sharp = np.full((32, 32, 3), 100, dtype=np.uint8)
    cv2.rectangle(sharp, (4, 4), (27, 27), (255, 255, 255), 2)
    blurry = cv2.GaussianBlur(sharp, (9, 9), 3)
    frames = [
        TrackFrame(0, 0.0, blurry, (0, 0, 32, 32)),
        TrackFrame(1, 0.1, sharp, (0, 0, 32, 32)),
    ]

    result = preprocess(frames, PreprocessConfig(mode="best_frame"))

    assert result.diagnostics["frame_idx"] == 1
    assert len(result.diagnostics["frame_scores"]) == 2


def test_enhance_brightens_dark_image_without_large_hue_change() -> None:
    image = np.full((32, 32, 3), (30, 45, 60), dtype=np.uint8)
    before_hue = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 0].astype(np.int16)
    frames = [TrackFrame(0, 0.0, image, (0, 0, 32, 32))]

    result = preprocess(frames, PreprocessConfig(mode="enhance"))

    assert result.image is not None
    after_hsv = cv2.cvtColor(result.image, cv2.COLOR_BGR2HSV)
    assert float(result.image.mean()) > float(image.mean())
    assert float(np.abs(after_hsv[:, :, 0].astype(np.int16) - before_hue).mean()) < 5


def test_unimplemented_mode_falls_back_to_passthrough() -> None:
    result = preprocess(_frames(), PreprocessConfig(mode="invalid"))

    assert result.image is not None
    assert result.diagnostics["fallback"] is True
    assert "NotImplementedError" in result.diagnostics["error"]
