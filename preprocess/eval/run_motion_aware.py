"""Reproducible motion-aware processing for a road-sign video window."""

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


def _edge_density(image: np.ndarray) -> float:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return float(np.mean(cv2.Canny(gray, 50, 150) > 0))


def _read_frames(
    video: str | Path,
    frame_indices: Sequence[int],
    crop: tuple[int, int, int, int],
    sign_bbox: tuple[int, int, int, int],
) -> list[TrackFrame]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"unable to open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames: list[TrackFrame] = []
    x1, y1, x2, y2 = crop
    try:
        for frame_idx in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, full_frame = capture.read()
            if not ok:
                raise ValueError(f"unable to read frame {frame_idx}")
            image = full_frame[y1:y2, x1:x2].copy()
            frames.append(
                TrackFrame(
                    frame_idx=frame_idx,
                    timestamp=frame_idx / fps,
                    full_frame=image,
                    bbox=sign_bbox,
                )
            )
    finally:
        capture.release()
    return frames


def run(
    video: str | Path,
    output: str | Path,
    frame_indices: Sequence[int] = (381, 382, 383, 384, 385),
) -> dict[str, object]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    crop = (2500, 1050, 3150, 1600)
    sign_bbox = (100, 50, 550, 450)
    frames = _read_frames(video, frame_indices, crop, sign_bbox)

    baseline = preprocess(frames, PreprocessConfig(mode="best_frame"))
    median = preprocess(
        frames,
        PreprocessConfig(
            mode="fusion",
            max_frames=5,
            time_budget_ms=15000,
            ecc_min_correlation=0.75,
            fusion_method="median",
        ),
    )
    weighted = preprocess(
        frames,
        PreprocessConfig(
            mode="fusion",
            max_frames=5,
            time_budget_ms=15000,
            ecc_min_correlation=0.75,
            fusion_method="weighted_mean",
        ),
    )
    raw_mean = np.mean(
        np.stack([frame.full_frame[50:450, 100:550] for frame in frames]).astype(
            np.float32
        ),
        axis=0,
    ).clip(0, 255).astype(np.uint8)

    outputs: dict[str, dict[str, object]] = {}
    for name, image, diagnostics in (
        ("baseline_best_frame", baseline.image, baseline.diagnostics),
        ("motion_aware_median", median.image, median.diagnostics),
        ("motion_aware_weighted", weighted.image, weighted.diagnostics),
        ("baseline_raw_mean_not_for_use", raw_mean, {"method": "unaligned_mean"}),
    ):
        if image is None:
            continue
        path = output_path / f"{name}.png"
        cv2.imwrite(str(path), image)
        outputs[name] = {
            "path": path.name,
            "shape": list(image.shape),
            "sharpness": sharpness_score(image),
            "edge_density": _edge_density(image),
            "diagnostics": diagnostics,
        }

    preferred = max(
        ("motion_aware_median", "motion_aware_weighted"),
        key=lambda name: float(outputs[name]["sharpness"]),
    )
    final_path = output_path / "final_verified.png"
    cv2.imwrite(str(final_path), cv2.imread(str(output_path / outputs[preferred]["path"])))
    manifest = {
        "video": str(video),
        "frame_indices": list(frame_indices),
        "crop_bbox": crop,
        "sign_bbox": sign_bbox,
        "outputs": outputs,
        "preferred": preferred,
        "final_verified": final_path.name,
        "restoration": {
            "method": "gamma + CLAHE in existing preprocess",
            "deep_deblur": "not_available_and_not_applied",
            "super_resolution": "not_applied_to_avoid_text_hallucination",
        },
        "limitations": [
            "The raw mean is retained only as a baseline; it is not a final output.",
            "The sign ROI is supplied from the verified track and is not an automatic detector result.",
        ],
    }
    (output_path / "motion_aware_manifest.json").write_text(
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
    print(json.dumps({"preferred": manifest["preferred"], "final": manifest["final_verified"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
