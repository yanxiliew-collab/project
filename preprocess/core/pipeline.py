"""Entry point for image preprocessing."""

from time import perf_counter
from typing import Sequence

from common.types import PreprocessResult, TrackFrame
from preprocess.config import PreprocessConfig
from preprocess.core.best_frame import best_frame
from preprocess.core.enhance import auto_gamma, clahe_enhance, gamma_correct
from preprocess.core.fusion import fuse


def _passthrough(track_frames: Sequence[TrackFrame]) -> PreprocessResult:
    if not track_frames:
        return PreprocessResult(
            image=None,
            diagnostics={"mode": "passthrough", "empty_input": True},
        )

    frame = track_frames[-1]
    x1, y1, x2, y2 = frame.bbox
    image = frame.full_frame[y1:y2, x1:x2].copy()
    return PreprocessResult(
        image=image,
        diagnostics={
            "mode": "passthrough",
            "frame_idx": frame.frame_idx,
            "bbox": frame.bbox,
        },
    )


def preprocess(
    track_frames: Sequence[TrackFrame], cfg: PreprocessConfig
) -> PreprocessResult:
    """Preprocess tracked frames, currently supporting passthrough only."""

    started = perf_counter()
    try:
        if cfg.mode == "passthrough":
            result = _passthrough(track_frames)
        elif cfg.mode == "best_frame":
            result = best_frame(track_frames)
        elif cfg.mode == "enhance":
            result = _passthrough(track_frames)
            if result.image is not None:
                enhanced = auto_gamma(result.image)
                enhanced = gamma_correct(enhanced, cfg.gamma)
                result.image = clahe_enhance(
                    enhanced, cfg.clahe_clip_limit, cfg.clahe_tile_grid
                )
                result.diagnostics["mode"] = "enhance"
        elif cfg.mode == "fusion":
            result = fuse(track_frames, cfg)
            if result.image is None and track_frames:
                result = best_frame(track_frames)
                result.diagnostics["fallback_best_frame"] = True
            elif result.image is not None:
                enhanced = auto_gamma(result.image)
                enhanced = gamma_correct(enhanced, cfg.gamma)
                result.image = clahe_enhance(
                    enhanced, cfg.clahe_clip_limit, cfg.clahe_tile_grid
                )
        else:
            raise NotImplementedError(
                f"preprocess mode {cfg.mode!r} is not implemented"
            )
    except Exception as exc:
        result = _passthrough(track_frames)
        result.diagnostics.update(
            {
                "fallback": True,
                "error": f"{type(exc).__name__}: {exc}",
            }
        )
    result.diagnostics["elapsed_ms"] = (perf_counter() - started) * 1000.0
    return result
