"""Short-window motion-blur experiments without hallucinating restoration."""

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


def _flow_vector(first: np.ndarray, second: np.ndarray) -> tuple[float, float]:
    a = cv2.cvtColor(first, cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(second, cv2.COLOR_BGR2GRAY)
    flow = cv2.calcOpticalFlowFarneback(a, b, None, 0.5, 3, 21, 3, 5, 1.2, 0)
    magnitude, angle = cv2.cartToPolar(flow[..., 0], flow[..., 1])
    weights = magnitude / max(float(magnitude.sum()), 1e-6)
    dx = float(np.sum(flow[..., 0] * weights))
    dy = float(np.sum(flow[..., 1] * weights))
    return dx, dy


def _wiener_motion(image: np.ndarray, dx: float, dy: float, k: float = 0.01) -> np.ndarray:
    """Conservative frequency-domain inverse of a short linear motion PSF."""

    length = int(np.clip(np.hypot(dx, dy), 1.0, 12.0))
    angle = float(np.arctan2(dy, dx))
    psf = np.zeros(image.shape[:2], dtype=np.float32)
    center = (psf.shape[1] // 2, psf.shape[0] // 2)
    end = (
        int(round(center[0] + np.cos(angle) * length / 2)),
        int(round(center[1] + np.sin(angle) * length / 2)),
    )
    start = (
        int(round(center[0] - np.cos(angle) * length / 2)),
        int(round(center[1] - np.sin(angle) * length / 2)),
    )
    cv2.line(psf, start, end, 1.0, 1)
    psf /= max(float(psf.sum()), 1e-6)
    psf = np.roll(psf, (-center[1], -center[0]), axis=(0, 1))
    transfer = np.fft.fft2(psf)
    result = np.empty_like(image)
    for channel in range(image.shape[2]):
        spectrum = np.fft.fft2(image[..., channel].astype(np.float32))
        restored = np.fft.ifft2(
            np.conj(transfer) * spectrum / (np.abs(transfer) ** 2 + k)
        ).real
        result[..., channel] = np.clip(restored, 0, 255).astype(np.uint8)
    return result


def _metrics(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    text = gray[100:330, 80:430]
    return {
        "sharpness": sharpness_score(image),
        "text_laplacian": float(cv2.Laplacian(text, cv2.CV_32F).var()),
        "text_edge_density": float(np.mean(cv2.Canny(text, 50, 150) > 0)),
        "text_contrast": float(text.std()),
    }


def run(video: str | Path, output: str | Path) -> dict[str, object]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    capture = cv2.VideoCapture(str(video))
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames: dict[int, TrackFrame] = {}
    try:
        for index in range(345, 386):
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, full = capture.read()
            if ok:
                crop = full[1050:1600, 2500:3150].copy()
                frames[index] = TrackFrame(index, index / fps, crop, (100, 50, 550, 450))
    finally:
        capture.release()

    flow = [_flow_vector(frames[i].full_frame, frames[i + 1].full_frame) for i in range(345, 385)]
    mean_dx = float(np.mean([item[0] for item in flow]))
    mean_dy = float(np.mean([item[1] for item in flow]))
    experiments: list[dict[str, object]] = []
    windows = {
        "best_frame_3": [381, 383, 385],
        "median_3": [381, 383, 385],
        "weighted_3": [381, 383, 385],
        "median_5": [381, 382, 383, 384, 385],
        "weighted_5": [381, 382, 383, 384, 385],
        "median_7": [379, 380, 381, 382, 383, 384, 385],
    }
    for name, indices in windows.items():
        selected = [frames[index] for index in indices]
        if name.startswith("best"):
            result = preprocess(selected, PreprocessConfig(mode="best_frame"))
        else:
            method = "weighted_mean" if name.startswith("weighted") else "median"
            result = preprocess(
                selected,
                PreprocessConfig(
                    mode="fusion",
                    max_frames=len(selected),
                    time_budget_ms=15000,
                    ecc_min_correlation=0.75,
                    fusion_method=method,
                ),
            )
        if result.image is None:
            continue
        image = result.image
        out_name = f"{name}.png"
        cv2.imwrite(str(output_path / out_name), image)
        experiments.append(
            {
                "name": name,
                "indices": indices,
                "output": out_name,
                "metrics": _metrics(image),
                "diagnostics": result.diagnostics,
            }
        )

    baseline = next(item for item in experiments if item["name"] == "median_5")
    restored = _wiener_motion(
        cv2.imread(str(output_path / baseline["output"])), mean_dx, mean_dy
    )
    cv2.imwrite(str(output_path / "median_5_wiener.png"), restored)
    experiments.append(
        {
            "name": "median_5_wiener",
            "indices": baseline["indices"],
            "output": "median_5_wiener.png",
            "metrics": _metrics(restored),
            "restoration": {
                "method": "Wiener inverse with estimated linear motion PSF",
                "dx": mean_dx,
                "dy": mean_dy,
                "length_px": float(np.clip(np.hypot(mean_dx, mean_dy), 1.0, 12.0)),
                "regularization": 0.01,
            },
        }
    )
    conservative = [
        item
        for item in experiments
        if item["name"] != "median_5_wiener"
    ]
    preferred = max(conservative, key=lambda item: item["metrics"]["text_laplacian"])
    manifest = {
        "video": str(video),
        "motion_estimate": {"mean_dx": mean_dx, "mean_dy": mean_dy},
        "experiments": experiments,
        "preferred_conservative": preferred["name"],
        "wiener_review": {
            "name": "median_5_wiener",
            "decision": "rejected_for_final",
            "reason": "Higher edge score is accompanied by ringing/overshoot visible around sign borders and text; not trusted as recovered detail.",
        },
        "limitations": [
            "Wiener output is a traditional baseline, not learned deblurring.",
            "A high sharpness score can amplify ringing; visual/OCR validation remains required.",
            "No method can recover text absent from all source frames.",
        ],
    }
    (output_path / "blur_experiment_manifest.json").write_text(
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
    print(json.dumps({"preferred": manifest["preferred_conservative"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
