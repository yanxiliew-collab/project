"""Manual ROI selection followed by OpenCV single-object tracking."""

from __future__ import annotations

import logging
from collections import deque
from pathlib import Path
from typing import Any, Callable, Optional

import cv2
import numpy as np

from common.types import TrackFrame

LOGGER = logging.getLogger(__name__)
RoiSelector = Callable[[str, np.ndarray], tuple[int, int, int, int]]
TrackerFactory = Callable[[], Any]


def _tracker_factory() -> Any:
    for name in ("TrackerCSRT_create", "TrackerKCF_create"):
        factory = getattr(cv2, name, None)
        if factory is None and hasattr(cv2, "legacy"):
            factory = getattr(cv2.legacy, name, None)
        if factory is not None:
            return factory()
    raise RuntimeError("OpenCV CSRT/KCF tracker is unavailable")


def _clip_bbox(
    bbox: tuple[float, float, float, float], width: int, height: int
) -> tuple[int, int, int, int]:
    x, y, box_width, box_height = bbox
    x1 = max(0, min(width, int(round(x))))
    y1 = max(0, min(height, int(round(y))))
    x2 = max(0, min(width, int(round(x + box_width))))
    y2 = max(0, min(height, int(round(y + box_height))))
    return x1, y1, x2, y2


def _default_roi_selector(window_name: str, frame: np.ndarray) -> tuple[int, int, int, int]:
    return tuple(int(value) for value in cv2.selectROI(window_name, frame, False, False))


def manual_track(
    video_path: str | Path,
    max_frames_per_track: int = 8,
    fps_override: Optional[float] = None,
    *,
    window_name: str = "Select road sign",
    roi_selector: Optional[RoiSelector] = None,
    tracker_factory: Optional[TrackerFactory] = None,
) -> dict[int, list[TrackFrame]]:
    """Select a road-sign ROI and track it through the remainder of a video.

    A zero-sized selection skips the current frame and lets the user continue
    until the first clear frame is selected. If tracking fails, the track ends
    and the stopping frame is logged. As with ``iter_tracks``, tracks with
    fewer than two frames are omitted.
    """

    if max_frames_per_track < 1:
        raise ValueError("max_frames_per_track must be positive")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = float(fps_override or 0)
    if fps <= 0:
        capture.release()
        raise ValueError("video FPS unavailable; provide fps_override")

    select = roi_selector or _default_roi_selector
    create_tracker = tracker_factory or _tracker_factory
    history: deque[TrackFrame] = deque(maxlen=max_frames_per_track)
    frame_idx = 0
    selected = False
    tracker: Any = None
    selected_bbox: tuple[int, int, int, int] = (0, 0, 0, 0)

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if not selected:
                raw_bbox = select(window_name, frame)
                bbox = _clip_bbox(raw_bbox, frame.shape[1], frame.shape[0])
                if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                    frame_idx += 1
                    continue
                tracker = create_tracker()
                init_bbox = (
                    bbox[0],
                    bbox[1],
                    bbox[2] - bbox[0],
                    bbox[3] - bbox[1],
                )
                initialized = tracker.init(frame, init_bbox)
                if initialized is False:
                    LOGGER.warning("manual track initialization failed at frame %d", frame_idx)
                    break
                selected = True
                selected_bbox = bbox
            else:
                tracking_ok, tracked_bbox = tracker.update(frame)
                if not tracking_ok:
                    LOGGER.warning("manual track stopped at frame %d", frame_idx)
                    break
                selected_bbox = _clip_bbox(
                    tuple(float(value) for value in tracked_bbox),
                    frame.shape[1],
                    frame.shape[0],
                )
                if (
                    selected_bbox[2] <= selected_bbox[0]
                    or selected_bbox[3] <= selected_bbox[1]
                ):
                    LOGGER.warning("manual track stopped at frame %d: empty bbox", frame_idx)
                    break

            history.append(
                TrackFrame(
                    frame_idx=frame_idx,
                    timestamp=frame_idx / fps,
                    full_frame=frame.copy(),
                    bbox=selected_bbox,
                )
            )
            frame_idx += 1
    finally:
        capture.release()
        try:
            cv2.destroyWindow(window_name)
        except cv2.error:
            pass

    return {0: list(history)} if len(history) >= 2 else {}
