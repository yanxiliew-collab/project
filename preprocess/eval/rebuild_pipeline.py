"""Rebuild road-sign extraction from a source MP4 without using prior outputs."""

from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from common.types import TrackFrame
from preprocess.config import PreprocessConfig
from preprocess.core.best_frame import sharpness_score
from preprocess.core.pipeline import preprocess
from preprocess.eval.run_dynamic_sign import (
    FRAME_END,
    FRAME_START,
    SEARCH_REGION,
    _edge_density,
    _sign_bbox,
)
from preprocess.eval.run_text_separation import _separation_metrics, _text_features


CANONICAL = (450, 300)
OUTPUT_DIR = Path("outputs/rebuild_road_sign")


def _metadata(video: Path) -> dict[str, object]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"unable to open video: {video}")
    result = {
        "path": str(video),
        "frame_count": int(capture.get(cv2.CAP_PROP_FRAME_COUNT)),
        "fps": float(capture.get(cv2.CAP_PROP_FPS)),
        "width": int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        "height": int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        "codec": int(capture.get(cv2.CAP_PROP_FOURCC)),
    }
    capture.release()
    result["duration_seconds"] = result["frame_count"] / max(result["fps"], 1.0)
    return result


def _dependencies(weights: Path) -> dict[str, object]:
    names = ["cv2", "numpy", "ultralytics", "torch", "onnxruntime", "realesrgan"]
    available = {name: importlib.util.find_spec(name) is not None for name in names}
    available["weights_exists"] = weights.exists()
    return available


def _yolo_probe(video: Path, weights: Path) -> dict[str, object]:
    result: dict[str, object] = {
        "attempted": False,
        "used_for_sign_track": False,
        "reason": "",
        "class_counts": {},
    }
    if not weights.exists():
        result["reason"] = "checkpoint missing; deterministic color fallback used"
        return result
    try:
        from ultralytics import YOLO

        model = YOLO(str(weights))
        capture = cv2.VideoCapture(str(video))
        indices = [0, FRAME_START, FRAME_END]
        counts: dict[str, int] = {}
        result["attempted"] = True
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            predictions = model.predict(frame, verbose=False)
            if not predictions:
                continue
            names = predictions[0].names
            classes = predictions[0].boxes.cls.cpu().numpy().astype(int) if predictions[0].boxes else []
            for class_id in classes:
                label = str(names.get(int(class_id), class_id))
                counts[label] = counts.get(label, 0) + 1
        capture.release()
        result["class_counts"] = counts
        result["reason"] = (
            "COCO checkpoint has no generic road-sign class; sign track remains deterministic color fallback"
        )
    except Exception as exc:
        result["reason"] = f"YOLO probe failed: {exc}; deterministic color fallback used"
    return result


def _order_points(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    ordered = np.zeros((4, 2), dtype=np.float32)
    sums = points.sum(axis=1)
    differences = np.diff(points, axis=1).reshape(-1)
    ordered[0] = points[np.argmin(sums)]
    ordered[2] = points[np.argmax(sums)]
    ordered[1] = points[np.argmin(differences)]
    ordered[3] = points[np.argmax(differences)]
    return ordered


def _rectify(image: np.ndarray, bbox: tuple[int, int, int, int]) -> tuple[np.ndarray, str]:
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([65, 80, 40], np.uint8), np.array([100, 255, 220], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        polygon = cv2.approxPolyDP(contour, 0.03 * perimeter, True)
        if len(polygon) == 4 and cv2.contourArea(polygon) > 0.25 * crop.shape[0] * crop.shape[1]:
            source = _order_points(polygon.reshape(4, 2))
            target = np.array([[0, 0], [CANONICAL[0] - 1, 0], [CANONICAL[0] - 1, CANONICAL[1] - 1], [0, CANONICAL[1] - 1]], dtype=np.float32)
            matrix = cv2.getPerspectiveTransform(source, target)
            return cv2.warpPerspective(crop, matrix, CANONICAL), "four_corner_homography"
    return cv2.resize(crop, CANONICAL, interpolation=cv2.INTER_AREA), "padded_bbox_resize"


def _quality(image: np.ndarray, previous: Optional[np.ndarray]) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flow = 0.0
    if previous is not None:
        a = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        vector = cv2.calcOpticalFlowFarneback(a, gray, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        flow = float(np.mean(np.linalg.norm(vector, axis=2)))
    focused, mask, stats = _text_features(image)
    separation = _separation_metrics(focused, mask)
    score = (
        0.45 * np.log1p(sharpness_score(image))
        + 0.25 * np.log1p(separation["foreground_background_contrast"])
        + 0.20 * np.log1p(separation["edge_density"] * 1000)
        - 0.10 * flow
    )
    return {
        "sharpness": float(sharpness_score(image)),
        "flow_from_previous": flow,
        "foreground_background_contrast": separation["foreground_background_contrast"],
        "foreground_ratio": stats["foreground_ratio"],
        "glare_ratio": stats["glare_ratio"],
        "selection_score": float(score),
    }


def _contact_sheet(items: list[tuple[str, np.ndarray]], path: Path) -> None:
    tiles = []
    for label, image in items:
        tile = cv2.resize(image, CANONICAL)
        cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(tile)
    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.zeros((rows * CANONICAL[1], columns * CANONICAL[0], 3), np.uint8)
    for index, tile in enumerate(tiles):
        y, x = divmod(index, columns)
        sheet[y * CANONICAL[1]:(y + 1) * CANONICAL[1], x * CANONICAL[0]:(x + 1) * CANONICAL[0]] = tile
    cv2.imwrite(str(path), sheet)


def run(video: str | Path, output: str | Path = OUTPUT_DIR, weights: str | Path = "yolo11n.pt") -> dict[str, object]:
    video_path = Path(video)
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    metadata = _metadata(video_path)
    dependency_report = _dependencies(Path(weights))
    yolo = _yolo_probe(video_path, Path(weights))

    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise ValueError(f"unable to open video: {video_path}")
    fps = float(metadata["fps"]) or 30.0
    canonical_frames: list[TrackFrame] = []
    records: list[dict[str, object]] = []
    previous = None
    try:
        frame_idx = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if FRAME_START <= frame_idx <= FRAME_END:
                bbox = _sign_bbox(image)
                if bbox is not None:
                    rectified, method = _rectify(image, bbox)
                    quality = _quality(rectified, previous)
                    canonical_frames.append(TrackFrame(frame_idx, frame_idx / fps, rectified, (0, 0, 450, 300)))
                    records.append({"frame_idx": frame_idx, "source_bbox": list(bbox), "rectification": method, **quality})
                    previous = rectified
            frame_idx += 1
    finally:
        capture.release()
    if len(canonical_frames) < 3:
        raise ValueError("fewer than three source sign frames were detected")

    best_index = max(range(len(records)), key=lambda index: records[index]["selection_score"])
    center_frame = records[best_index]["frame_idx"]
    candidates = [
        (frame, record)
        for frame, record in zip(canonical_frames, records)
        if abs(int(record["frame_idx"]) - int(center_frame)) <= 6
    ]
    selected_pairs = sorted(candidates, key=lambda pair: pair[1]["selection_score"], reverse=True)[:5]
    selected = [pair[0] for pair in selected_pairs]
    selected_ids = [frame.frame_idx for frame in selected]

    cfg = PreprocessConfig(mode="fusion", max_frames=len(selected), time_budget_ms=15000, ecc_min_correlation=0.75, fusion_method="median")
    original_result = preprocess(selected, cfg)
    text_frames = []
    masks = []
    for frame in selected:
        focused, mask, _ = _text_features(frame.full_frame)
        text_frames.append(TrackFrame(frame.frame_idx, frame.timestamp, focused, frame.bbox))
        masks.append(mask)
    text_result = preprocess(text_frames, cfg)
    foreground_frames = [
        TrackFrame(frame.frame_idx, frame.timestamp, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), frame.bbox)
        for frame, mask in zip(selected, masks)
    ]
    foreground_result = preprocess(foreground_frames, cfg)
    if (
        original_result.image is None
        or text_result.image is None
        or foreground_result.image is None
    ):
        raise ValueError("fusion returned no image")

    cv2.imwrite(str(output_path / "final_original_rectified.png"), original_result.image)
    cv2.imwrite(str(output_path / "final_text_focused.png"), text_result.image)
    cv2.imwrite(str(output_path / "final_foreground_mask.png"), foreground_result.image)
    focused_mask = _text_features(text_result.image)[1]
    items = [(f"f{frame.frame_idx}", frame.full_frame) for frame in selected]
    items.extend([
        ("original", original_result.image),
        ("text_focused", text_result.image),
        ("foreground", foreground_result.image),
        ("text_mask", cv2.cvtColor(focused_mask, cv2.COLOR_GRAY2BGR)),
    ])
    _contact_sheet(items, output_path / "rebuild_contact_sheet.png")

    report = {
        "video": str(video_path),
        "metadata": metadata,
        "dependencies": dependency_report,
        "yolo": yolo,
        "detector": {
            "primary": "HSV blue-green sign surface candidate",
            "fallback_reason": yolo["reason"],
            "frames_detected": len(records),
        },
        "selection": {
            "all_frame_count": len(records),
            "selected_frames": selected_ids,
            "window_center": center_frame,
            "criteria": "sign ROI sharpness + text separation + edge density - local optical flow",
            "rejected_frames": [int(record["frame_idx"]) for record in records if int(record["frame_idx"]) not in selected_ids],
        },
        "records": records,
        "outputs": {
            "final_original_rectified": {
                "path": "final_original_rectified.png",
                "metrics": _separation_metrics(original_result.image, _text_features(original_result.image)[1]),
                "diagnostics": original_result.diagnostics,
            },
            "final_text_focused": {
                "path": "final_text_focused.png",
                "metrics": _separation_metrics(text_result.image, focused_mask),
                "diagnostics": text_result.diagnostics,
            },
            "final_foreground_mask": {
                "path": "final_foreground_mask.png",
                "metrics": _separation_metrics(foreground_result.image, foreground_result.image[:, :, 0]),
                "diagnostics": foreground_result.diagnostics,
            },
        },
        "contact_sheet": "rebuild_contact_sheet.png",
        "limitations": [
            "The supplied COCO checkpoint is probed but not trusted for generic road-sign detection.",
            "HSV fallback is deterministic but not a trained sign detector and may require domain-specific training data.",
            "Foreground/text separation is an evaluation branch; it improves separability but does not restore missing strokes.",
            "No prior outputs are read as input; only data/road.mp4 is used.",
        ],
    }
    (output_path / "rebuild_manifest.json").write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    (output_path / "rebuild_report.md").write_text(
        "# From-scratch road-sign rebuild\n\n"
        f"- Source: `{video_path}`\n"
        f"- Frames detected: `{len(records)}`; selected: `{selected_ids}`\n"
        f"- Detector: HSV fallback; YOLO probe: `{yolo['reason']}`\n"
        "- Rectification: four-corner homography when a valid quadrilateral is found, otherwise padded bbox resize.\n"
        "- Fusion: sign-canonical ECC alignment followed by median fusion; no raw unaligned mean.\n"
        "- Text branch: R-channel illumination correction, CLAHE, glare exclusion, top-hat and adaptive threshold.\n"
        "- The text branch is visual/segmentation enhancement, not true detail recovery.\n",
        encoding="utf-8",
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default=str(OUTPUT_DIR))
    parser.add_argument("--weights", default="yolo11n.pt")
    args = parser.parse_args(argv)
    result = run(args.video, args.out, args.weights)
    print(json.dumps({"selected_frames": result["selection"]["selected_frames"], "output": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
