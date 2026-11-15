"""Contrast-only road-sign fusion and parallel-line artifact diagnostics."""

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
from preprocess.core.fusion import _align
from preprocess.core.pipeline import preprocess
from preprocess.eval.rebuild_pipeline import CANONICAL, _rectify
from preprocess.eval.run_dynamic_sign import FRAME_END, FRAME_START, _edge_density, _sign_bbox


OUTPUT_DIR = Path("outputs/contrast_only")


def _read_canonical(video: Path) -> list[TrackFrame]:
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"unable to open video: {video}")
    fps = float(capture.get(cv2.CAP_PROP_FPS)) or 30.0
    frames: list[TrackFrame] = []
    try:
        index = 0
        while True:
            ok, image = capture.read()
            if not ok:
                break
            if FRAME_START <= index <= FRAME_END:
                bbox = _sign_bbox(image)
                if bbox is not None:
                    rectified, _ = _rectify(image, bbox)
                    frames.append(
                        TrackFrame(index, index / fps, rectified, (0, 0, CANONICAL[0], CANONICAL[1]))
                    )
            index += 1
    finally:
        capture.release()
    if len(frames) < 3:
        raise ValueError("fewer than three canonical sign frames were detected")
    return frames


def _contrast_only(image: np.ndarray, clip_limit: float = 1.6, gamma: float = 0.98) -> np.ndarray:
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    light = lab[:, :, 0]
    normalized = np.clip(light.astype(np.float32) / 255.0, 0.0, 1.0)
    corrected = np.power(normalized, gamma) * 255.0
    corrected = np.clip(corrected, 0, 255).astype(np.uint8)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(corrected)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def _directional_fft(image: np.ndarray, bins: int = 18) -> dict[str, object]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray -= cv2.GaussianBlur(gray, (0, 0), 3)
    spectrum = np.fft.fftshift(np.fft.fft2(gray))
    magnitude = np.log1p(np.abs(spectrum))
    height, width = magnitude.shape
    yy, xx = np.mgrid[:height, :width]
    angle = (np.degrees(np.arctan2(yy - height / 2, xx - width / 2)) + 180.0) % 180.0
    radius = np.sqrt((xx - width / 2) ** 2 + (yy - height / 2) ** 2)
    valid = (radius > min(height, width) * 0.05) & (radius < min(height, width) * 0.45)
    edges = np.linspace(0.0, 180.0, bins + 1)
    energies = []
    for left, right in zip(edges[:-1], edges[1:]):
        mask = valid & (angle >= left) & (angle < right)
        energies.append(float(magnitude[mask].mean()) if mask.any() else 0.0)
    peak = int(np.argmax(energies))
    mean_energy = float(np.mean(energies))
    return {
        "bin_degrees": float(180.0 / bins),
        "peak_angle_degrees": float((peak + 0.5) * 180.0 / bins),
        "energies": energies,
        "anisotropy_ratio": float(max(energies) / max(mean_energy, 1e-6)),
    }


def _contrast_quality(image: np.ndarray, previous: Optional[np.ndarray]) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    flow = 0.0
    if previous is not None:
        previous_gray = cv2.cvtColor(previous, cv2.COLOR_BGR2GRAY)
        vector = cv2.calcOpticalFlowFarneback(
            previous_gray, gray, None, 0.5, 3, 21, 3, 5, 1.2, 0
        )
        flow = float(np.mean(np.linalg.norm(vector, axis=2)))
    lab_l = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)[:, :, 0]
    return {
        "sharpness": float(sharpness_score(image)),
        "local_contrast": float(lab_l.std()),
        "edge_density": float(_edge_density(image)),
        "flow_from_previous": flow,
        "selection_score": float(
            0.55 * np.log1p(sharpness_score(image))
            + 0.30 * np.log1p(lab_l.std())
            + 0.15 * np.log1p(_edge_density(image) * 1000.0)
            - 0.10 * flow
        ),
    }


def _edge_diagnostics(aligned: list[np.ndarray], reference: np.ndarray) -> dict[str, object]:
    ref_edge = cv2.Canny(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), 50, 150) > 0
    overlays = np.zeros((*ref_edge.shape, 3), dtype=np.uint8)
    edge_iou: list[float] = []
    disagreement: list[float] = []
    for image in aligned:
        edge = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 50, 150) > 0
        overlays[:, :, 1] = np.maximum(overlays[:, :, 1], edge.astype(np.uint8) * 255)
        overlays[:, :, 2] = np.maximum(overlays[:, :, 2], ref_edge.astype(np.uint8) * 255)
        union = np.logical_or(ref_edge, edge).sum()
        intersection = np.logical_and(ref_edge, edge).sum()
        edge_iou.append(float(intersection / max(union, 1)))
        disagreement.append(float(np.mean(np.logical_xor(ref_edge, edge))))
    return {
        "edge_iou_mean": float(np.mean(edge_iou)),
        "edge_iou_min": float(np.min(edge_iou)),
        "edge_disagreement_mean": float(np.mean(disagreement)),
        "edge_overlay": overlays,
    }


def _aligned_stack(frames: list[TrackFrame], reference: TrackFrame) -> tuple[list[np.ndarray], list[float], list[float]]:
    mask = np.full(reference.full_frame.shape[:2], 255, dtype=np.uint8)
    aligned: list[np.ndarray] = []
    correlations: list[float] = []
    residuals: list[float] = []
    for frame in frames:
        if frame.frame_idx == reference.frame_idx:
            candidate = frame.full_frame.copy()
            correlation = 1.0
        else:
            try:
                candidate, correlation = _align(reference.full_frame, frame.full_frame, mask)
            except cv2.error:
                candidate, correlation = frame.full_frame.copy(), 0.0
        aligned.append(candidate)
        correlations.append(float(correlation))
        residuals.append(float(np.mean(cv2.absdiff(candidate, reference.full_frame))))
    return aligned, correlations, residuals


def _metrics(image: np.ndarray) -> dict[str, float]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    edges = cv2.Canny(gray, 50, 150) > 0
    return {
        "sharpness": float(sharpness_score(image)),
        "laplacian": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
        "local_contrast": float(cv2.createCLAHE(clipLimit=1.0, tileGridSize=(8, 8)).apply(lab[:, :, 0]).std()),
        "edge_density": float(edges.mean()),
    }


def _write_contact_sheet(items: list[tuple[str, np.ndarray]], path: Path) -> None:
    tiles = []
    for label, image in items:
        tile = cv2.resize(image, CANONICAL)
        cv2.putText(tile, label, (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        tiles.append(tile)
    columns = 3
    rows = (len(tiles) + columns - 1) // columns
    sheet = np.zeros((rows * CANONICAL[1], columns * CANONICAL[0], 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        y, x = divmod(index, columns)
        sheet[y * CANONICAL[1]:(y + 1) * CANONICAL[1], x * CANONICAL[0]:(x + 1) * CANONICAL[0]] = tile
    cv2.imwrite(str(path), sheet)


def run(video: str | Path, output: str | Path = OUTPUT_DIR) -> dict[str, object]:
    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    frames = _read_canonical(Path(video))
    records = []
    previous = None
    for frame in frames:
        quality = _contrast_quality(frame.full_frame, previous)
        records.append({"frame_idx": frame.frame_idx, **quality})
        previous = frame.full_frame
    best_index = max(range(len(frames)), key=lambda index: records[index]["sharpness"])
    reference = frames[best_index]
    candidates = sorted(
        zip(frames, records),
        key=lambda pair: pair[1]["selection_score"],
        reverse=True,
    )[:5]
    selected = [pair[0] for pair in candidates]
    selected_ids = [frame.frame_idx for frame in selected]

    cfg = PreprocessConfig(
        mode="fusion",
        max_frames=len(selected),
        time_budget_ms=15000,
        ecc_min_correlation=0.75,
        fusion_method="median",
    )
    median_result = preprocess(selected, cfg)
    aligned, correlations, residuals = _aligned_stack(selected, reference)
    edge_info = _edge_diagnostics(aligned, reference.full_frame)
    residual_image = np.mean(
        [cv2.absdiff(image, reference.full_frame).astype(np.float32) for image in aligned],
        axis=0,
    ).clip(0, 255).astype(np.uint8)
    frame_difference = cv2.absdiff(selected[0].full_frame, selected[-1].full_frame)
    best_image = reference.full_frame
    median_image = median_result.image if median_result.image is not None else best_image
    contrast_image = _contrast_only(median_image)

    cv2.imwrite(str(output_path / "best_frame.png"), best_image)
    cv2.imwrite(str(output_path / "median_aligned.png"), median_image)
    cv2.imwrite(str(output_path / "contrast_after_fusion.png"), contrast_image)
    cv2.imwrite(str(output_path / "edge_overlay.png"), edge_info.pop("edge_overlay"))
    cv2.imwrite(str(output_path / "warp_residual.png"), residual_image)
    cv2.imwrite(str(output_path / "frame_difference.png"), frame_difference)
    _write_contact_sheet(
        [
            ("best_frame", best_image),
            ("median_aligned", median_image),
            ("contrast_after_fusion", contrast_image),
            ("edge_overlay", cv2.imread(str(output_path / "edge_overlay.png"))),
            ("warp_residual", cv2.imread(str(output_path / "warp_residual.png"))),
            ("frame_difference", frame_difference),
        ],
        output_path / "contrast_contact_sheet.png",
    )

    report = {
        "video": str(video),
        "frame_range": [FRAME_START, FRAME_END],
        "detected_frames": len(frames),
        "selected_frames": selected_ids,
        "reference_frame": reference.frame_idx,
        "hypothesis": {
            "parallel_lines_from_misregistration": {
                "status": "supported" if edge_info["edge_disagreement_mean"] > 0.08 else "not_supported",
                "evidence": {
                    "alignment_correlations": correlations,
                    "warp_residual_mean": float(np.mean(residuals)),
                    "edge_iou_mean": edge_info["edge_iou_mean"],
                    "edge_iou_min": edge_info["edge_iou_min"],
                    "edge_disagreement_mean": edge_info["edge_disagreement_mean"],
                },
            },
            "directional_frequency_structure": _directional_fft(median_image),
            "interpretation": (
                "Parallel edge pairs after alignment support residual misregistration or scale/warp error; "
                "if residuals are low while edge pairs remain in each frame, the artifact is more consistent "
                "with single-frame blur or sharpening halos."
            ),
        },
        "outputs": {
            "best_frame": {"path": "best_frame.png", "metrics": _metrics(best_image)},
            "median_aligned": {"path": "median_aligned.png", "metrics": _metrics(median_image)},
            "contrast_after_fusion": {"path": "contrast_after_fusion.png", "metrics": _metrics(contrast_image)},
        },
        "records": records,
        "limitations": [
            "The contrast branch preserves BGR color and does not emit a foreground mask as a final result.",
            "CLAHE and gamma improve separability but cannot recover character strokes absent from the source frames.",
            "Directional FFT reports anisotropic structure, not a unique physical blur kernel.",
        ],
        "references": [
            "https://arxiv.org/abs/2608.07120",
            "https://openreview.net/forum?id=H98n6clgHv",
            "https://www.mdpi.com/2076-3417/14/6/2630",
        ],
    }
    (output_path / "contrast_manifest.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (output_path / "contrast_report.md").write_text(
        "# Contrast-only road-sign diagnostics\n\n"
        f"- Source: `{video}`; detected frames: `{len(frames)}`; selected: `{selected_ids}`\n"
        "- Main outputs preserve color: best frame, aligned median, and Lab-L CLAHE/gamma after fusion.\n"
        f"- Alignment correlation mean: `{float(np.mean(correlations)):.4f}`; "
        f"warp residual mean: `{float(np.mean(residuals)):.3f}`; "
        f"edge IoU mean: `{edge_info['edge_iou_mean']:.4f}`.\n"
        "- A contrast increase is enhancement only; it is not deblurring or character recovery.\n"
        "- References: multiple-hypothesis flow for ambiguous matches, drift-resilient temporal fusion, "
        "and flow-error reduction methods (listed in `contrast_manifest.json`).\n",
        encoding="utf-8",
    )
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--out", default=str(OUTPUT_DIR))
    args = parser.parse_args(argv)
    result = run(args.video, args.out)
    print(json.dumps({"selected_frames": result["selected_frames"], "output": args.out}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
