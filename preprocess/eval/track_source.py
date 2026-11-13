"""Video tracking source for real-world preprocessing evaluation."""

from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Iterable, Optional, Sequence

import cv2
import numpy as np

from common.types import TrackFrame

Detection = tuple[int, tuple[int, int, int, int], float, Optional[int]]
Detector = Callable[[np.ndarray], Any]


def _to_numpy(value: Any) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()
    return np.asarray(value)


def _default_detector(weights: str) -> Detector:
    from ultralytics import YOLO

    model = YOLO(weights)

    def detect(frame: np.ndarray) -> Any:
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        return results[0] if results else []

    return detect


def _iter_detections(raw: Any) -> Iterable[Detection]:
    boxes = getattr(raw, "boxes", None)
    if boxes is not None:
        coordinates = _to_numpy(getattr(boxes, "xyxy", []))
        ids = getattr(boxes, "id", None)
        if ids is None:
            return
        track_ids = _to_numpy(ids).reshape(-1)
        confidences = _to_numpy(getattr(boxes, "conf", np.ones(len(track_ids)))).reshape(-1)
        class_ids = getattr(boxes, "cls", None)
        classes = (
            _to_numpy(class_ids).reshape(-1)
            if class_ids is not None
            else np.full(len(track_ids), -1)
        )
        for index, track_id in enumerate(track_ids):
            yield (
                int(track_id),
                tuple(float(value) for value in coordinates[index]),
                float(confidences[index]),
                int(classes[index]) if classes[index] >= 0 else None,
            )
        return

    if isinstance(raw, dict):
        raw = [raw]
    for item in raw or []:
        if isinstance(item, dict):
            track_id = item.get("track_id", item.get("id"))
            bbox = item.get("bbox", item.get("xyxy"))
            confidence = item.get("conf", 1.0)
            class_id = item.get("class_id", item.get("cls"))
        else:
            track_id = getattr(item, "track_id", getattr(item, "id", None))
            bbox = getattr(item, "bbox", getattr(item, "xyxy", None))
            confidence = getattr(item, "conf", 1.0)
            class_id = getattr(item, "class_id", getattr(item, "cls", None))
        if track_id is None or bbox is None:
            continue
        yield (
            int(track_id),
            tuple(float(value) for value in bbox),
            float(confidence),
            None if class_id is None else int(class_id),
        )


def _clip_bbox(
    bbox: Sequence[float], width: int, height: int
) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = (int(round(value)) for value in bbox)
    return (
        max(0, min(width, x1)),
        max(0, min(height, y1)),
        max(0, min(width, x2)),
        max(0, min(height, y2)),
    )


def iter_tracks(
    video_path: str | Path,
    weights: str,
    classes: Optional[Sequence[int]] = None,
    conf: float = 0.3,
    max_frames_per_track: int = 8,
    drop_every: int = 0,
    fps_override: Optional[float] = None,
    detector: Optional[Detector] = None,
) -> dict[int, list[TrackFrame]]:
    """Track a video and retain only a bounded recent history per track."""

    if max_frames_per_track < 1:
        raise ValueError("max_frames_per_track must be positive")
    if drop_every < 0:
        raise ValueError("drop_every must be non-negative")
    if conf < 0:
        raise ValueError("conf must be non-negative")

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"unable to open video: {video_path}")
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if fps <= 0:
        fps = float(fps_override or 0)
    if fps <= 0:
        capture.release()
        raise ValueError("video FPS unavailable; provide fps_override")

    detect = detector or _default_detector(weights)
    allowed_classes = None if classes is None else set(classes)
    tracks: dict[int, deque[TrackFrame]] = defaultdict(
        lambda: deque(maxlen=max_frames_per_track)
    )
    frame_idx = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            current_idx = frame_idx
            frame_idx += 1
            if drop_every > 0 and frame_idx % drop_every == 0:
                continue
            for track_id, bbox, confidence, class_id in _iter_detections(detect(frame)):
                if confidence < conf:
                    continue
                if allowed_classes is not None and class_id not in allowed_classes:
                    continue
                height, width = frame.shape[:2]
                clipped = _clip_bbox(bbox, width, height)
                tracks[track_id].append(
                    TrackFrame(
                        frame_idx=current_idx,
                        timestamp=current_idx / fps,
                        full_frame=frame.copy(),
                        bbox=clipped,
                    )
                )
    finally:
        capture.release()

    return {
        track_id: list(history)
        for track_id, history in tracks.items()
        if len(history) >= 2
    }
