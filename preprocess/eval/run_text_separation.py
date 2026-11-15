"""Text-focused sign preprocessing with glare suppression and foreground masks."""

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
from preprocess.eval.run_dynamic_sign import (
    CANONICAL_SIZE,
    FRAME_END,
    FRAME_START,
    SEARCH_REGION,
    _edge_density,
    _sign_bbox,
    _canonical,
)


def _glare_mask(image: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    saturation = hsv[:, :, 1]
    threshold = max(235, int(np.percentile(value, 99.3)))
    mask = ((value >= threshold) & (saturation <= 85)).astype(np.uint8) * 255
    return cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


def _text_features(image: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    glare = _glare_mask(image)
    red = image[:, :, 2]
    repaired = cv2.inpaint(red, glare, 3, cv2.INPAINT_TELEA)
    illumination = cv2.GaussianBlur(repaired, (0, 0), 21)
    corrected = cv2.divide(
        repaired.astype(np.float32),
        np.maximum(illumination.astype(np.float32), 1.0),
        scale=128.0,
    )
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(corrected)
    top_hat = cv2.morphologyEx(
        enhanced, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    )
    threshold = max(8, int(np.percentile(top_hat, 82)))
    local = cv2.adaptiveThreshold(
        enhanced,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31,
        -3,
    )
    foreground = ((top_hat >= threshold) & (local > 0) & (glare == 0)).astype(np.uint8) * 255
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    foreground = cv2.morphologyEx(foreground, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    # Keep a soft image for alignment/fusion; preserve the mask separately for evaluation.
    focused = np.where(foreground > 0, enhanced, (0.35 * enhanced).astype(np.uint8))
    focused_bgr = cv2.cvtColor(focused, cv2.COLOR_GRAY2BGR)
    stats = {
        "glare_ratio": float(np.mean(glare > 0)),
        "foreground_ratio": float(np.mean(foreground > 0)),
        "foreground_edge_density": float(np.mean(cv2.Canny(foreground, 50, 150) > 0)),
        "enhanced_local_contrast": float(enhanced.std()),
    }
    return focused_bgr, foreground, stats


def _separation_metrics(image: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    foreground = mask > 0
    background = ~foreground
    fg_mean = float(gray[foreground].mean()) if foreground.any() else 0.0
    bg_mean = float(gray[background].mean()) if background.any() else 0.0
    return {
        "sharpness": sharpness_score(image),
        "laplacian": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        "edge_density": _edge_density(image),
        "foreground_ratio": float(np.mean(foreground)),
        "foreground_background_contrast": abs(fg_mean - bg_mean),
        "foreground_mean": fg_mean,
        "background_mean": bg_mean,
    }


def _contact_sheet(items: list[tuple[str, np.ndarray]], path: Path) -> None:
    tiles = []
    for label, image in items:
        tile = cv2.resize(image, CANONICAL_SIZE)
        cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(tile)
    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.zeros((rows * CANONICAL_SIZE[1], columns * CANONICAL_SIZE[0], 3), np.uint8)
    for index, tile in enumerate(tiles):
        y, x = divmod(index, columns)
        sheet[y * CANONICAL_SIZE[1]:(y + 1) * CANONICAL_SIZE[1],
              x * CANONICAL_SIZE[0]:(x + 1) * CANONICAL_SIZE[0]] = tile
    cv2.imwrite(str(path), sheet)


def run(video: str | Path, output: str | Path) -> dict[str, object]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"unable to open video: {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
    enhanced_frames: list[TrackFrame] = []
    foreground_frames: list[TrackFrame] = []
    per_frame = []
    try:
        for frame_idx in range(FRAME_START, FRAME_END + 1):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, image = cap.read()
            if not ok:
                continue
            bbox = _sign_bbox(image)
            if bbox is None:
                continue
            canonical = _canonical(image, bbox)
            focused, foreground, stats = _text_features(canonical)
            enhanced_frames.append(TrackFrame(frame_idx, frame_idx / fps, focused, (0, 0, 450, 300)))
            foreground_frames.append(
                TrackFrame(
                    frame_idx,
                    frame_idx / fps,
                    cv2.cvtColor(foreground, cv2.COLOR_GRAY2BGR),
                    (0, 0, 450, 300),
                )
            )
            per_frame.append({"frame_idx": frame_idx, **stats})
    finally:
        cap.release()
    if len(enhanced_frames) < 3:
        raise ValueError("not enough dynamically detected sign frames")

    ranked = sorted(enhanced_frames, key=lambda frame: sharpness_score(frame.full_frame), reverse=True)
    selected_ids = [frame.frame_idx for frame in ranked[:5]]
    selected_enhanced = ranked[:5]
    by_id = {frame.frame_idx: frame for frame in foreground_frames}
    selected_foreground = [by_id[frame.frame_idx] for frame in selected_enhanced]
    cfg = PreprocessConfig(
        mode="fusion",
        max_frames=len(selected_enhanced),
        time_budget_ms=15000,
        ecc_min_correlation=0.75,
        fusion_method="median",
    )
    enhanced_result = preprocess(selected_enhanced, cfg)
    foreground_result = preprocess(selected_foreground, cfg)
    if enhanced_result.image is None or foreground_result.image is None:
        raise ValueError("text separation fusion returned no image")

    baseline = cv2.imread("outputs/road_signs/dynamic_sign/final_verified.png")
    mask_for_result = _text_features(enhanced_result.image)[1]
    outputs = {
        "text_enhanced_median": {
            "path": "text_enhanced_median.png",
            "metrics": _separation_metrics(enhanced_result.image, mask_for_result),
            "diagnostics": enhanced_result.diagnostics,
        },
        "foreground_mask_median": {
            "path": "foreground_mask_median.png",
            "metrics": _separation_metrics(foreground_result.image, foreground_result.image[:, :, 0]),
            "diagnostics": foreground_result.diagnostics,
        },
    }
    cv2.imwrite(str(output_path / "text_enhanced_median.png"), enhanced_result.image)
    cv2.imwrite(str(output_path / "foreground_mask_median.png"), foreground_result.image)
    if baseline is not None:
        baseline_mask = _text_features(cv2.resize(baseline, CANONICAL_SIZE))[1]
        outputs["existing_dynamic_final"] = {
            "path": "outputs/road_signs/dynamic_sign/final_verified.png",
            "metrics": _separation_metrics(cv2.resize(baseline, CANONICAL_SIZE), baseline_mask),
        }
    _contact_sheet(
        [
            ("enhanced", enhanced_result.image),
            ("foreground", foreground_result.image),
            ("mask", cv2.cvtColor(mask_for_result, cv2.COLOR_GRAY2BGR)),
        ] + ([("existing", cv2.resize(baseline, CANONICAL_SIZE))] if baseline is not None else []),
        output_path / "text_separation_contact_sheet.png",
    )
    manifest = {
        "video": str(video),
        "frame_range": [FRAME_START, FRAME_END],
        "search_region": list(SEARCH_REGION),
        "selected_frames": selected_ids,
        "method": {
            "color_channel": "BGR R channel",
            "glare": "HSV high-value low-saturation mask, Telea inpainting, excluded from foreground",
            "illumination": "single-scale division by Gaussian illumination estimate",
            "contrast": "CLAHE on corrected R channel",
            "text_features": "white top-hat plus adaptive Gaussian threshold",
            "fusion": "sign-only canonical crops, ECC alignment, median fusion",
        },
        "per_frame_quality": per_frame,
        "outputs": outputs,
        "contact_sheet": "text_separation_contact_sheet.png",
        "limitations": [
            "Foreground masks are a separation aid, not recovered ground-truth characters.",
            "Top-hat/adaptive thresholds can miss low-contrast or thick strokes and can retain foliage-like bright structures.",
            "The output improves separability/contrast; it does not restore text absent from source frames.",
        ],
    }
    (output_path / "text_separation_manifest.json").write_text(
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
