"""Track a road sign in the source video and run object-centric fusion."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from common.types import TrackFrame
from preprocess.config import PreprocessConfig
from preprocess.core.best_frame import sharpness_score
from preprocess.core.pipeline import preprocess


FRAME_START = 350
FRAME_END = 385
SEARCH_REGION = (2000, 900, 3700, 1800)
CANONICAL_SIZE = (450, 300)


def _sign_bbox(image: np.ndarray) -> tuple[int, int, int, int] | None:
    x0, y0, x2, y2 = SEARCH_REGION
    region = image[y0:y2, x0:x2]
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(
        hsv,
        np.array([65, 80, 40], dtype=np.uint8),
        np.array([100, 255, 210], dtype=np.uint8),
    )
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        area = cv2.contourArea(contour)
        if width >= 40 and height >= 40:
            candidates.append((area, x, y, width, height))
    if not candidates:
        return None
    _, x, y, width, height = max(candidates)
    pad_x = max(8, int(round(width * 0.12)))
    pad_y = max(8, int(round(height * 0.12)))
    return (
        max(0, x0 + x - pad_x),
        max(0, y0 + y - pad_y),
        min(image.shape[1], x0 + x + width + pad_x),
        min(image.shape[0], y0 + y + height + pad_y),
    )


def _canonical(image: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x1, y1, x2, y2 = bbox
    crop = image[y1:y2, x1:x2]
    return cv2.resize(crop, CANONICAL_SIZE, interpolation=cv2.INTER_AREA)


def _edge_density(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.Canny(gray, 50, 150) > 0))


def _metrics(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    red = image[:, :, 2]
    return {
        "sharpness_gray": sharpness_score(image),
        "laplacian_gray": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        "laplacian_red": float(cv2.Laplacian(red, cv2.CV_32F).var()),
        "edge_density_gray": _edge_density(image),
        "edge_density_red": float(np.mean(cv2.Canny(red, 50, 150) > 0)),
    }


def _contact_sheet(items: list[tuple[str, np.ndarray]], path: Path) -> None:
    tiles = []
    for label, image in items:
        tile = cv2.resize(image, CANONICAL_SIZE)
        cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        tiles.append(tile)
    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.zeros((rows * CANONICAL_SIZE[1], columns * CANONICAL_SIZE[0], 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y = index // columns
        x = index % columns
        sheet[
            y * CANONICAL_SIZE[1]:(y + 1) * CANONICAL_SIZE[1],
            x * CANONICAL_SIZE[0]:(x + 1) * CANONICAL_SIZE[0],
        ] = tile
    cv2.imwrite(str(path), sheet)


def run(video: str | Path, output: str | Path) -> dict[str, object]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"unable to open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames: list[TrackFrame] = []
    detections = []
    try:
        for frame_idx in range(FRAME_START, FRAME_END + 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, image = capture.read()
            if not ok:
                continue
            bbox = _sign_bbox(image)
            if bbox is None:
                detections.append({"frame_idx": frame_idx, "detected": False})
                continue
            canonical = _canonical(image, bbox)
            frames.append(TrackFrame(frame_idx, frame_idx / fps, canonical, (0, 0, 450, 300)))
            detections.append(
                {
                    "frame_idx": frame_idx,
                    "detected": True,
                    "source_bbox": list(bbox),
                    "score": sharpness_score(canonical),
                    "edge_density": _edge_density(canonical),
                }
            )
    finally:
        capture.release()
    if len(frames) < 3:
        raise ValueError(f"only {len(frames)} sign frames detected")

    ranked = sorted(frames, key=lambda frame: sharpness_score(frame.full_frame), reverse=True)
    selected = ranked[: min(5, len(ranked))]
    selected_ids = [frame.frame_idx for frame in selected]
    median = preprocess(
        selected,
        PreprocessConfig(
            mode="fusion",
            max_frames=len(selected),
            time_budget_ms=15000,
            ecc_min_correlation=0.75,
            fusion_method="median",
        ),
    )
    weighted = preprocess(
        selected,
        PreprocessConfig(
            mode="fusion",
            max_frames=len(selected),
            time_budget_ms=15000,
            ecc_min_correlation=0.75,
            fusion_method="weighted_mean",
        ),
    )
    if median.image is None or weighted.image is None:
        raise ValueError("dynamic sign fusion returned no image")

    median_path = output_path / "dynamic_sign_median.png"
    weighted_path = output_path / "dynamic_sign_weighted.png"
    red_path = output_path / "dynamic_sign_median_r_channel.png"
    final_path = output_path / "final_verified.png"
    cv2.imwrite(str(median_path), median.image)
    cv2.imwrite(str(weighted_path), weighted.image)
    cv2.imwrite(str(red_path), cv2.cvtColor(median.image[:, :, 2], cv2.COLOR_GRAY2BGR))
    cv2.imwrite(str(final_path), median.image)
    existing = cv2.imread(
        "outputs/road_signs/motion_aware/motion_aware_median.png"
    )
    comparison_items = [
        (f"f{frame.frame_idx}", frame.full_frame) for frame in selected
    ] + [
        ("dynamic_median", median.image),
        ("dynamic_weighted", weighted.image),
        ("dynamic_red", cv2.cvtColor(median.image[:, :, 2], cv2.COLOR_GRAY2BGR)),
    ]
    if existing is not None:
        comparison_items.append(("existing_median", existing))
    _contact_sheet(
        comparison_items,
        output_path / "dynamic_sign_contact_sheet.png",
    )
    manifest = {
        "video": str(video),
        "frame_range": [FRAME_START, FRAME_END],
        "search_region": list(SEARCH_REGION),
        "canonical_size": list(CANONICAL_SIZE),
        "detections": detections,
        "selected_frames": selected_ids,
        "selection": "top five sign-only sharpness scores after dynamic bbox crop",
        "outputs": {
            "dynamic_sign_median": {
                "path": median_path.name,
                "metrics": _metrics(median.image),
                "diagnostics": median.diagnostics,
            },
            "dynamic_sign_weighted": {
                "path": weighted_path.name,
                "metrics": _metrics(weighted.image),
                "diagnostics": weighted.diagnostics,
            },
            "dynamic_sign_median_r_channel": {
                "path": red_path.name,
                "metrics": _metrics(cv2.cvtColor(median.image[:, :, 2], cv2.COLOR_GRAY2BGR)),
                "interpretation": "controlled comparison only; not a universal grayscale replacement",
            },
        },
        "final_verified": final_path.name,
        "comparison": {
            "existing_motion_aware_median": (
                "outputs/road_signs/motion_aware/motion_aware_median.png"
                if existing is not None
                else "missing"
            ),
            "note": "Images use different canonical crops; visual comparison is supplemental, not a direct metric ranking.",
        },
        "contact_sheet": "dynamic_sign_contact_sheet.png",
        "limitations": [
            "The detector is a conservative HSV sign-surface candidate, not a trained road-sign detector.",
            "Axis-aligned dynamic crops do not recover details absent from the source frames.",
        ],
    }
    (output_path / "dynamic_sign_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )
    return manifest


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    manifest = run(args.video, args.out)
    print(json.dumps({"selected_frames": manifest["selected_frames"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
