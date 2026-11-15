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


def _sign_candidates(frame: np.ndarray) -> list[dict[str, object]]:
    """Find likely sign regions for COCO models that do not contain signs."""

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    masks = [
        cv2.inRange(hsv, (0, 70, 50), (12, 255, 255)),
        cv2.inRange(hsv, (165, 70, 50), (179, 255, 255)),
        cv2.inRange(hsv, (90, 50, 45), (140, 255, 255)),
        cv2.inRange(hsv, (15, 70, 60), (45, 255, 255)),
    ]
    mask = masks[0]
    for candidate in masks[1:]:
        mask = cv2.bitwise_or(mask, candidate)
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    height, width = frame.shape[:2]
    minimum_area = max(20.0, height * width * 0.00015)
    detections: list[dict[str, object]] = []
    for contour in cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[
        0
    ]:
        area = cv2.contourArea(contour)
        if area < minimum_area:
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if box_width < 8 or box_height < 8:
            continue
        aspect = box_width / box_height
        if not 0.25 <= aspect <= 4.0:
            continue
        fill = area / float(box_width * box_height)
        if fill < 0.25:
            continue
        detections.append(
            {
                "bbox": (x, y, x + box_width, y + box_height),
                "conf": min(0.9, 0.35 + fill),
                "cls": None,
                "_score": area * fill,
            }
        )
    detections.sort(key=lambda item: float(item["_score"]), reverse=True)
    return [
        {key: value for key, value in item.items() if key != "_score"}
        for item in detections[:8]
    ]


def _default_detector(weights: str) -> Detector:
    from ultralytics import YOLO

    model = YOLO(weights)
    previous: list[tuple[int, tuple[int, int, int, int]]] = []
    next_id = -1

    def detect(frame: np.ndarray) -> Any:
        nonlocal next_id, previous
        results = model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",
            verbose=False,
        )
        detections: list[dict[str, object]] = []
        if results:
            for track_id, bbox, confidence, class_id in _iter_detections(results[0]):
                if class_id not in (9, 11):
                    continue
                detections.append(
                    {
                        "track_id": track_id,
                        "bbox": bbox,
                        "conf": confidence,
                        "cls": class_id,
                    }
                )

        # A stock COCO model commonly detects cars but has no generic road-sign
        # class. Keep its detections and add deterministic IDs for sign proposals.
        proposals = _sign_candidates(frame)
        current: list[tuple[int, tuple[int, int, int, int]]] = []
        for proposal in proposals:
            bbox = tuple(int(value) for value in proposal["bbox"])
            best_id: Optional[int] = None
            best_iou = 0.0
            for previous_id, previous_bbox in previous:
                overlap = _bbox_iou(bbox, previous_bbox)
                if overlap > best_iou:
                    best_iou = overlap
                    best_id = previous_id
            if best_id is not None and best_iou >= 0.2:
                track_id = best_id
            else:
                track_id = next_id
                next_id -= 1
            detections.append({**proposal, "track_id": track_id})
            current.append((track_id, bbox))
        previous = current
        return detections

    return detect


def _bbox_iou(
    first: Sequence[int], second: Sequence[int]
) -> float:
    ax1, ay1, ax2, ay2 = first
    bx1, by1, bx2, by2 = second
    intersection = max(0, min(ax2, bx2) - max(ax1, bx1)) * max(
        0, min(ay2, by2) - max(ay1, by1)
    )
    if intersection == 0:
        return 0.0
    first_area = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    second_area = max(0, bx2 - bx1) * max(0, by2 - by1)
    return intersection / float(first_area + second_area - intersection)


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
