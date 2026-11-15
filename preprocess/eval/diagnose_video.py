"""Reproducible diagnosis of blur, motion, occlusion, and fusion artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np

from preprocess.core.fusion import _align, _alignment_mask
from preprocess.core.best_frame import sharpness_score


def _fft_high_frequency_ratio(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    spectrum = np.fft.fftshift(np.fft.fft2(gray - gray.mean()))
    magnitude = np.abs(spectrum)
    height, width = gray.shape
    yy, xx = np.ogrid[:height, :width]
    radius = np.sqrt(((yy - height / 2) / max(height / 2, 1)) ** 2 + ((xx - width / 2) / max(width / 2, 1)) ** 2)
    high = magnitude[radius > 0.35].sum()
    total = magnitude.sum()
    return float(high / max(total, 1e-6))


def _noise_snr(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    smooth = cv2.GaussianBlur(gray, (3, 3), 0)
    noise = gray - smooth
    return float(20.0 * np.log10(max(float(smooth.std()), 1e-6) / max(float(noise.std()), 1e-6)))


def _flow_metrics(previous: np.ndarray, current: np.ndarray) -> dict[str, float]:
    previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
    current_gray = cv2.cvtColor(current, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(
        previous_gray, current_gray, None, 0.5, 3, 21, 3, 5, 1.2, 0
    )
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    return {
        "mean_magnitude": float(np.mean(magnitude)),
        "p95_magnitude": float(np.percentile(magnitude, 95)),
        "direction_std_deg": float(np.std(np.degrees(angle))),
        "flow_inconsistency": float(np.std(magnitude)),
    }


def _contact_sheet(images: list[tuple[str, np.ndarray]], path: Path) -> None:
    tiles = []
    for label, image in images:
        tile = cv2.resize(image, (320, 220))
        cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)
        tiles.append(tile)
    columns = 4
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.zeros((rows * 220, columns * 320, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y = index // columns
        x = index % columns
        sheet[y * 220:(y + 1) * 220, x * 320:(x + 1) * 320] = tile
    cv2.imwrite(str(path), sheet)


def diagnose(
    video: str | Path,
    output: str | Path,
    frame_start: int = 350,
    frame_end: int = 385,
) -> dict[str, object]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    crop = (2500, 1050, 3150, 1600)
    sign_bbox = (100, 50, 550, 450)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"unable to open video: {video}")
    frames: dict[int, np.ndarray] = {}
    try:
        for index in range(frame_start, frame_end + 1):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, full = capture.read()
            if not ok:
                continue
            x1, y1, x2, y2 = crop
            frames[index] = full[y1:y2, x1:x2].copy()
    finally:
        capture.release()
    if not frames:
        raise ValueError("no diagnostic frames decoded")

    sx1, sy1, sx2, sy2 = sign_bbox
    rows: list[dict[str, object]] = []
    flow_rows: list[dict[str, object]] = []
    previous = None
    for index, image in frames.items():
        sign = image[sy1:sy2, sx1:sx2]
        gray = cv2.cvtColor(sign, cv2.COLOR_BGR2GRAY)
        laplacian = float(cv2.Laplacian(gray, cv2.CV_32F).var())
        contrast = float(gray.std())
        saturation = float(cv2.cvtColor(sign, cv2.COLOR_BGR2HSV)[..., 1].mean())
        row = {
            "frame_idx": index,
            "sharpness_laplacian": laplacian,
            "sharpness_composite": sharpness_score(sign),
            "fft_high_frequency_ratio": _fft_high_frequency_ratio(sign),
            "local_contrast": contrast,
            "noise_snr_db": _noise_snr(sign),
            "mean_brightness": float(gray.mean()),
            "saturation": saturation,
            "edge_density": float(np.mean(cv2.Canny(gray, 50, 150) > 0)),
        }
        rows.append(row)
        if previous is not None:
            metrics = _flow_metrics(previous, image)
            metrics["frame_idx"] = index
            flow_rows.append(metrics)
        previous = image

    reference_index = max(rows, key=lambda row: row["sharpness_composite"])["frame_idx"]
    reference = frames[reference_index]
    reference_sign = reference[sy1:sy2, sx1:sx2]
    alignment_mask = _alignment_mask(sign_bbox, reference.shape)
    alignment_rows = []
    for index, image in frames.items():
        if index == reference_index:
            continue
        try:
            aligned, correlation = _align(reference, image, alignment_mask)
            aligned_sign = aligned[sy1:sy2, sx1:sx2]
            residual = np.abs(
                reference_sign.astype(np.float32) - aligned_sign.astype(np.float32)
            ) / 255.0
            alignment_rows.append(
                {
                    "frame_idx": index,
                    "ecc_correlation": correlation,
                    "ecc_residual_mean": float(residual.mean()),
                    "ecc_residual_p95": float(np.percentile(residual, 95)),
                    "occlusion_ratio": float(np.mean(residual > 0.25)),
                }
            )
        except cv2.error as exc:
            alignment_rows.append({"frame_idx": index, "error": str(exc)})

    contact_images = []
    for index in (350, 360, 370, 380, 381, 382, 383, 384, 385):
        if index in frames:
            contact_images.append((f"f{index}", frames[index][sy1:sy2, sx1:sx2]))
    for name in (
        "baseline_best_frame.png",
        "motion_aware_median.png",
        "motion_aware_weighted.png",
        "baseline_raw_mean_not_for_use.png",
    ):
        image = cv2.imread(str(Path("outputs/road_signs/motion_aware") / name))
        if image is not None:
            contact_images.append((name[:16], image))
    _contact_sheet(contact_images, output_path / "diagnostic_contact_sheet.png")

    summary = {
        "video": str(video),
        "frame_range": [frame_start, frame_end],
        "crop_bbox": crop,
        "sign_bbox": sign_bbox,
        "frame_count": len(frames),
        "frame_metrics": rows,
        "flow_metrics": flow_rows,
        "alignment_metrics": alignment_rows,
        "reference_frame": reference_index,
        "diagnostic_contact_sheet": "diagnostic_contact_sheet.png",
        "root_causes": {
            "information_limit": "high_priority",
            "motion_blur": "high_priority",
            "defocus_or_low_detail": "medium_priority",
            "roi_and_alignment": "medium_priority",
            "occlusion": "low_to_medium_priority",
            "rolling_shutter": "not_proven",
            "compression": "low_priority",
        },
        "actions": {
            "use_best_or_motion_aware_fusion": "feasible_and_applied",
            "crop_tighter_before_ecc": "feasible_and_applied",
            "discard_low_quality_frames": "feasible_and_applied",
            "deep_deblur": "not_applied_without_validated_model",
            "super_resolution": "not_applied; cannot recover missing text and may hallucinate",
            "capture_higher_shutter_or_resolution": "required_for_true detail recovery",
        },
    }
    (output_path / "diagnosis.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    summary = diagnose(args.video, args.out)
    print(json.dumps({"reference_frame": summary["reference_frame"], "frames": summary["frame_count"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
