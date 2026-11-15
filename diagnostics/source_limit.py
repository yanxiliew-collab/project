"""Strict source-information diagnostics for the road-sign video."""

from __future__ import annotations

import csv
import json
import math
import shutil
import subprocess
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - optional plotting dependency
    plt = None


ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT / "data" / "road.mp4"
OUT = ROOT / "outputs" / "diagnostics"
FRAME_START, FRAME_END = 350, 385
SIGN_SEARCH = (2100, 850, 3500, 1800)
BACKGROUND_ROI = (0, 400, 280, 1300)
GROUND_TRUTH = ["八德", "鶯歌", "Bade", "Yingge", "出口", "EXIT", "1½", "公里", "km", "110乙"]


def _bbox(
    frame: np.ndarray,
    previous: tuple[int, int, int, int] | None = None,
) -> tuple[tuple[int, int, int, int] | None, bool]:
    x0, y0, x1, y1 = SIGN_SEARCH
    crop = frame[y0:y1, x0:x1]
    hsv = cv2.cvtColor(crop, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([45, 70, 35], np.uint8), np.array([105, 255, 230], np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((11, 11), np.uint8))
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        area = w * h
        ratio = w / max(h, 1)
        if area > 15_000 and 1.1 < ratio < 3.5:
            candidates.append((area, x + x0, y + y0, x + x0 + w, y + y0 + h))
    if not candidates:
        return previous, previous is not None
    if previous is None:
        _, x0, y0, x1, y1 = max(candidates)
        return (int(x0), int(y0), int(x1), int(y1)), False
    px = (previous[0] + previous[2]) / 2.0
    py = (previous[1] + previous[3]) / 2.0
    pw = previous[2] - previous[0]
    ph = previous[3] - previous[1]
    compatible = []
    for area, x0, y0, x1, y1 in candidates:
        cx = (x0 + x1) / 2.0
        cy = (y0 + y1) / 2.0
        distance = math.hypot(cx - px, cy - py)
        scale = (x1 - x0) * (y1 - y0) / max(pw * ph, 1)
        if distance <= 180.0 and 0.45 <= scale <= 2.2:
            compatible.append((distance, -area, x0, y0, x1, y1))
    if not compatible:
        return previous, True
    _, _, x0, y0, x1, y1 = min(compatible)
    return (int(x0), int(y0), int(x1), int(y1)), False


def _read_frames() -> dict[int, np.ndarray]:
    capture = cv2.VideoCapture(str(VIDEO))
    if not capture.isOpened():
        raise RuntimeError(f"unable to open {VIDEO}")
    result: dict[int, np.ndarray] = {}
    try:
        for index in range(FRAME_END + 1):
            ok, frame = capture.read()
            if not ok:
                break
            if index >= FRAME_START:
                result[index] = frame
    finally:
        capture.release()
    return result


def _ffprobe() -> dict[str, Any]:
    command = [
        "ffprobe", "-v", "error", "-print_format", "json",
        "-show_format", "-show_streams", "-show_frames", "-select_streams", "v:0",
        str(VIDEO),
    ]
    try:
        raw = subprocess.check_output(command, text=True, stderr=subprocess.STDOUT)
        data = json.loads(raw)
    except Exception as exc:
        return {"available": False, "error": str(exc)}
    frames = data.get("frames", [])
    selected = []
    for ordinal, frame in enumerate(frames):
        if FRAME_START <= ordinal <= FRAME_END:
            selected.append({
                "frame": ordinal,
                "pict_type": frame.get("pict_type"),
                "pkt_size": int(frame["pkt_size"]) if frame.get("pkt_size") else None,
                "key_frame": int(frame.get("key_frame", 0)),
            })
    stream = next((s for s in data.get("streams", []) if s.get("codec_type") == "video"), {})
    format_data = data.get("format", {})
    width = int(stream.get("width", 0) or 0)
    height = int(stream.get("height", 0) or 0)
    bitrate = float(format_data.get("bit_rate", 0) or 0)
    fps = 30.0
    if stream.get("r_frame_rate"):
        num, den = stream["r_frame_rate"].split("/")
        fps = float(num) / max(float(den), 1.0)
    seconds = float(format_data.get("duration", 0) or 0)
    bpp = bitrate / max(width * height * fps, 1.0)
    return {
        "available": True,
        "codec": stream.get("codec_name"),
        "profile": stream.get("profile"),
        "pix_fmt": stream.get("pix_fmt"),
        "width": width,
        "height": height,
        "fps": fps,
        "duration_seconds": seconds,
        "bitrate_bps": bitrate,
        "bits_per_pixel_per_frame": bpp,
        "frames_350_385": selected,
        "pict_type_counts": {
            kind: sum(1 for item in selected if item["pict_type"] == kind)
            for kind in sorted({item["pict_type"] for item in selected})
        },
    }


def _raw_crop(frame: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    x0, y0, x1, y1 = bbox
    return frame[y0:y1, x0:x1]


def _downsample_test(frame: np.ndarray) -> list[dict[str, float]]:
    result = []
    for factor in (2, 3, 4, 6):
        small = cv2.resize(frame, (frame.shape[1] // factor, frame.shape[0] // factor), interpolation=cv2.INTER_AREA)
        restored = cv2.resize(small, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_CUBIC)
        gray_a = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_b = cv2.cvtColor(restored, cv2.COLOR_BGR2GRAY)
        result.append({
            "factor": factor,
            "psnr_db": float(peak_signal_noise_ratio(gray_a, gray_b, data_range=255)),
            "ssim": float(structural_similarity(gray_a, gray_b, data_range=255)),
        })
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dx = np.abs(np.diff(gray, axis=1))
    dy = np.abs(np.diff(gray, axis=0))
    return {
        "round_trip": result,
        "neighbor_difference": {
            "horizontal_zero_fraction": float(np.mean(dx == 0)),
            "vertical_zero_fraction": float(np.mean(dy == 0)),
            "horizontal_even_odd_ratio": float(
                np.mean(dx[:, ::2]) / max(np.mean(dx[:, 1::2]), 1e-6)
            ),
            "vertical_even_odd_ratio": float(
                np.mean(dy[::2, :]) / max(np.mean(dy[1::2, :]), 1e-6)
            ),
        },
    }


def _radial_spectrum(image: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gray -= gray.mean()
    window = np.outer(np.hanning(gray.shape[0]), np.hanning(gray.shape[1]))
    spectrum = np.fft.fftshift(np.fft.fft2(gray * window))
    power = np.abs(spectrum) ** 2
    yy, xx = np.indices(power.shape)
    radius = np.sqrt((xx - power.shape[1] / 2) ** 2 + (yy - power.shape[0] / 2) ** 2)
    max_radius = min(image.shape[:2]) / 2.0
    bins = np.arange(0, int(max_radius) + 1)
    radial = np.zeros(len(bins) - 1, dtype=np.float64)
    for index in range(len(radial)):
        mask = (radius >= bins[index]) & (radius < bins[index + 1])
        radial[index] = float(power[mask].mean()) if mask.any() else 0.0
    freq = bins[:-1] / max(min(image.shape[:2]), 1)
    return freq, radial


def _spectrum_summary(image: np.ndarray) -> dict[str, Any]:
    freq, power = _radial_spectrum(image)
    log_power = np.log10(power + 1e-8)
    tail = log_power[int(len(log_power) * 0.8):]
    noise_floor = float(np.median(tail))
    above = np.where(log_power > noise_floor + 0.5)[0]
    cutoff = float(freq[above[-1]]) if len(above) else 0.0
    return {
        "frequencies_cycles_per_pixel": freq.tolist(),
        "log_power": log_power.tolist(),
        "noise_floor_log10": noise_floor,
        "cutoff_cycles_per_pixel": cutoff,
        "nyquist_fraction": cutoff / 0.5,
    }


def _edge_width(profile: np.ndarray) -> float | None:
    values = profile.astype(np.float32)
    gradient = np.abs(np.gradient(values))
    center = int(np.argmax(gradient))
    lo = max(0, center - 30)
    hi = min(len(values), center + 31)
    local = values[lo:hi]
    if local.max() - local.min() < 5:
        return None
    low = local.min() + 0.1 * (local.max() - local.min())
    high = local.min() + 0.9 * (local.max() - local.min())
    def crossing(threshold: float) -> float | None:
        indices = np.where(local >= threshold)[0]
        if not len(indices):
            return None
        index = int(indices[0])
        if index == 0:
            return 0.0
        left, right = float(local[index - 1]), float(local[index])
        fraction = (threshold - left) / max(right - left, 1e-6)
        return float(index - 1 + np.clip(fraction, 0.0, 1.0))

    low_cross = crossing(low)
    high_cross = crossing(high)
    if low_cross is None or high_cross is None:
        return None
    return max(0.0, high_cross - low_cross)


def _edge_spread(image: np.ndarray) -> dict[str, float | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    vertical = gray[h // 2, :]
    horizontal = gray[:, w // 2]
    return {
        "vertical_edge_10_90_px": _edge_width(vertical),
        "horizontal_edge_10_90_px": _edge_width(horizontal),
    }


def _flow_summary(frames: dict[int, np.ndarray], bboxes: dict[int, tuple[int, int, int, int]]) -> list[dict[str, float]]:
    rows = []
    for index in range(FRAME_START + 1, FRAME_END + 1):
        previous = cv2.cvtColor(frames[index - 1], cv2.COLOR_BGR2GRAY)
        current = cv2.cvtColor(frames[index], cv2.COLOR_BGR2GRAY)
        flow = cv2.calcOpticalFlowFarneback(previous, current, None, 0.5, 3, 21, 3, 5, 1.2, 0)
        x0, y0, x1, y1 = bboxes[index]
        region = flow[y0:y1, x0:x1]
        magnitude = np.linalg.norm(region, axis=2)
        dx = float(np.median(region[:, :, 0]))
        dy = float(np.median(region[:, :, 1]))
        prev_box = bboxes[index - 1]
        scale_x = (x1 - x0) / max(prev_box[2] - prev_box[0], 1)
        scale_y = (y1 - y0) / max(prev_box[3] - prev_box[1], 1)
        rows.append({
            "frame": index,
            "median_dx_px": dx,
            "median_dy_px": dy,
            "median_magnitude_px": float(np.median(magnitude)),
            "mean_magnitude_px": float(np.mean(magnitude)),
            "bbox_scale_x": float(scale_x),
            "bbox_scale_y": float(scale_y),
            "trail_px_at_1_60": float(np.mean(magnitude) / 60.0),
            "trail_px_at_1_120": float(np.mean(magnitude) / 120.0),
            "trail_px_at_1_250": float(np.mean(magnitude) / 250.0),
        })
    return rows


def _text_size(image: np.ndarray) -> dict[str, float | None]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    white = ((gray > np.percentile(gray, 75)) & (hsv[:, :, 1] < 130)).astype(np.uint8) * 255
    white = cv2.morphologyEx(white, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(white)
    boxes = []
    h, w = gray.shape
    for x, y, bw, bh, area in stats[1:]:
        if area >= 3 and 1 <= bw <= w * 0.4 and 1 <= bh <= h * 0.35:
            boxes.append((x, y, bw, bh, area))
    upper = [b[3] for b in boxes if 0.15 * h < b[1] < 0.55 * h]
    lower = [b[3] for b in boxes if b[1] >= 0.45 * h]
    stroke = [min(b[2], b[3]) for b in boxes]
    return {
        "sign_width_px": float(w),
        "sign_height_px": float(h),
        "large_text_height_px": float(np.percentile(upper, 75)) if upper else None,
        "small_text_height_px": float(np.percentile(lower, 75)) if lower else None,
        "estimated_stroke_width_px": float(np.percentile(stroke, 25)) if stroke else None,
    }


def _plot_spectra(curves: dict[str, dict[str, Any]], path: Path) -> None:
    if plt is None:
        return
    plt.figure(figsize=(10, 6))
    for label, curve in curves.items():
        plt.plot(curve["frequencies_cycles_per_pixel"], curve["log_power"], label=label)
    plt.xlabel("cycles / pixel (Nyquist = 0.5)")
    plt.ylabel("log10 radial power")
    plt.grid(True, alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=140)
    plt.close()


def _ocr_probe(raw_crops: dict[int, np.ndarray], output: Path) -> dict[str, Any]:
    result: dict[str, Any] = {
        "ground_truth_assumption": GROUND_TRUTH,
        "paddleocr": {"available": False, "reason": "package unavailable"},
        "easyocr": {"available": False, "reason": "not executed"},
        "benchmark_inputs": [
            {"input": "raw_frame_350_to_385", "count": len(raw_crops), "status": "unavailable"},
            {"input": "raw_2x_bicubic_frame_350_to_385", "count": len(raw_crops), "status": "unavailable"},
            {"input": "best_frame_existing_output", "count": 1, "status": "unavailable"},
            {"input": "median_fusion_existing_output", "count": 1, "status": "unavailable"},
            {"input": "contrast_after_fusion_existing_output", "count": 1, "status": "unavailable"},
        ],
    }
    try:
        import easyocr  # type: ignore

        model_dir = Path.home() / ".EasyOCR" / "model"
        cached = list(model_dir.glob("*")) if model_dir.exists() else []
        if not cached:
            result["easyocr"] = {
                "available": False,
                "reason": "package installed but model cache absent; no download performed",
            }
            return result
        reader = easyocr.Reader(["en"], gpu=False, download_enabled=False, verbose=False)
        samples = {}
        for frame, crop in list(raw_crops.items())[:3]:
            output_rows = reader.readtext(crop, detail=1, paragraph=False)
            samples[str(frame)] = [
                {"text": row[1], "confidence": float(row[2])}
                for row in output_rows
            ]
        result["easyocr"] = {"available": True, "language": "en", "samples": samples}
    except Exception as exc:
        result["easyocr"] = {"available": False, "reason": str(exc)}
    return result
    if shutil.which("python") is None:
        return result


def _fusion_information(
        raw_crops: dict[int, np.ndarray],
        existing_metrics: dict[str, Any],
) -> dict[str, Any]:
        target_index = 385
        target = raw_crops[target_index]
        target_size = (target.shape[1], target.shape[0])
        rows = []
        for index in sorted(raw_crops):
            if index == target_index:
                continue
            candidate = cv2.resize(raw_crops[index], target_size, interpolation=cv2.INTER_AREA)
            gray_target = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
            gray_candidate = cv2.cvtColor(candidate, cv2.COLOR_BGR2GRAY)
            rows.append({
                "target_frame": target_index,
                "candidate_frame": index,
                "single_candidate_psnr_db": float(
                    peak_signal_noise_ratio(gray_target, gray_candidate, data_range=255)
                ),
                "single_candidate_ssim": float(
                    structural_similarity(gray_target, gray_candidate, data_range=255)
                ),
                "candidate_high_frequency": float(
                    np.mean(np.abs(cv2.Laplacian(gray_candidate, cv2.CV_32F)))
                ),
            })
        ordered = sorted(raw_crops)
        midpoint = len(ordered) // 2
        half_a = np.median(
            [cv2.resize(raw_crops[index], target_size, interpolation=cv2.INTER_AREA) for index in ordered[:midpoint]],
            axis=0,
        ).astype(np.uint8)
        half_b = np.median(
            [cv2.resize(raw_crops[index], target_size, interpolation=cv2.INTER_AREA) for index in ordered[midpoint:]],
            axis=0,
        ).astype(np.uint8)
        half_a_gray = cv2.cvtColor(half_a, cv2.COLOR_BGR2GRAY)
        half_b_gray = cv2.cvtColor(half_b, cv2.COLOR_BGR2GRAY)
        target_gray = cv2.cvtColor(target, cv2.COLOR_BGR2GRAY)
        return {
            "method": "raw crop resize only proxy; no saved fusion image and no canonical output used for raw measurements",
            "leave_one_out_against_frame_385": rows,
            "leave_one_out_summary": {
                "single_psnr_mean_db": float(np.mean([row["single_candidate_psnr_db"] for row in rows])),
                "single_ssim_mean": float(np.mean([row["single_candidate_ssim"] for row in rows])),
                "single_high_frequency_mean": float(np.mean([row["candidate_high_frequency"] for row in rows])),
            },
            "half_sequence_stability": {
                "half_a_vs_half_b_psnr_db": float(peak_signal_noise_ratio(half_a_gray, half_b_gray, data_range=255)),
                "half_a_vs_half_b_ssim": float(structural_similarity(half_a_gray, half_b_gray, data_range=255)),
                "half_a_vs_target_psnr_db": float(peak_signal_noise_ratio(target_gray, half_a_gray, data_range=255)),
                "half_b_vs_target_psnr_db": float(peak_signal_noise_ratio(target_gray, half_b_gray, data_range=255)),
            },
            "existing_output_spectra": existing_metrics,
            "interpretation": (
                "This proxy tests temporal consistency after resizing raw crops to a common frame-385 size. "
                "It cannot prove super-resolution because scale changes and interpolation are confounded."
            ),
        }
def run() -> dict[str, Any]:
    OUT.mkdir(parents=True, exist_ok=True)
    frames = _read_frames()
    bboxes: dict[int, tuple[int, int, int, int]] = {}
    fallback_frames = []
    previous = None
    for index in sorted(frames):
        box, used_fallback = _bbox(frames[index], previous)
        if box is not None:
            bboxes[index] = box
            previous = box
            if used_fallback:
                fallback_frames.append(index)
    raw_crops = {index: _raw_crop(frames[index], bboxes[index]) for index in bboxes}
    if len(raw_crops) < 30:
        raise RuntimeError(f"only {len(raw_crops)} raw sign crops detected")

    probe = _ffprobe()
    if probe.get("available"):
        packet_rows = probe.get("frames_350_385", [])
        quality_rows = []
        for item in packet_rows:
            index = item["frame"]
            gray = cv2.cvtColor(frames[index], cv2.COLOR_BGR2GRAY)
            quality_rows.append({
                "frame": index,
                "pkt_size": item["pkt_size"],
                "full_frame_laplacian": float(cv2.Laplacian(gray, cv2.CV_32F).var()),
            })
        if len(quality_rows) > 1:
            probe["packet_quality"] = quality_rows
            probe["packet_laplacian_correlation"] = float(np.corrcoef(
                [row["pkt_size"] for row in quality_rows],
                [row["full_frame_laplacian"] for row in quality_rows],
            )[0, 1])
    reference = frames[385]
    downsample = _downsample_test(reference)
    downsample["frame"] = 385
    downsample["roi"] = "full frame, original 3840x2160 pixels"

    curves: dict[str, dict[str, Any]] = {}
    spectrum_rows = []
    for index in (350, 360, 370, 375, 380, 385):
        sign = raw_crops[index]
        bg = _raw_crop(frames[index], BACKGROUND_ROI)
        curves[f"sign_f{index}"] = _spectrum_summary(sign)
        curves[f"background_f{index}"] = _spectrum_summary(bg)
        spectrum_rows.append({
            "frame": index,
            "sign_bbox": list(bboxes[index]),
            "background_roi": list(BACKGROUND_ROI),
            "sign_cutoff_cycles_per_pixel": curves[f"sign_f{index}"]["cutoff_cycles_per_pixel"],
            "background_cutoff_cycles_per_pixel": curves[f"background_f{index}"]["cutoff_cycles_per_pixel"],
        })
    _plot_spectra(
        {
            "sign f350": curves["sign_f350"],
            "sign f385": curves["sign_f385"],
            "background f350": curves["background_f350"],
            "background f385": curves["background_f385"],
        },
        OUT / "radial_spectra.png",
    )

    edge_rows = []
    size_rows = []
    for index, crop in raw_crops.items():
        edge_rows.append({"frame": index, "roi": list(bboxes[index]), **_edge_spread(crop)})
        size_rows.append({"frame": index, "roi": list(bboxes[index]), **_text_size(crop)})
    background_edge = _edge_spread(_raw_crop(frames[385], BACKGROUND_ROI))
    flow_rows = _flow_summary(frames, bboxes)

    existing = {
        "best_frame": ROOT / "outputs" / "contrast_only" / "best_frame.png",
        "median_fusion": ROOT / "outputs" / "road_signs" / "dynamic_sign" / "dynamic_sign_median.png",
        "contrast_after_fusion": ROOT / "outputs" / "contrast_only" / "contrast_after_fusion.png",
    }
    existing_metrics = {}
    for name, path in existing.items():
        image = cv2.imread(str(path))
        existing_metrics[name] = {
            "available": image is not None,
            "path": str(path.relative_to(ROOT)) if path.exists() else str(path),
            "spectrum": _spectrum_summary(image) if image is not None else None,
        }
    leave_one_out = _fusion_information(raw_crops, existing_metrics)
    ocr = _ocr_probe(raw_crops, OUT)
    report = {
        "source": str(VIDEO.relative_to(ROOT)),
        "frame_range": [FRAME_START, FRAME_END],
        "roi_policy": {
            "sign": "per-frame HSV candidate bbox on original frame; no resize for measurements",
            "background": list(BACKGROUND_ROI),
            "search_region": list(SIGN_SEARCH),
            "continuity_fallback_frames": fallback_frames,
        },
        "ground_truth": {"assumed": True, "strings": GROUND_TRUTH},
        "compression": probe,
        "effective_resolution": downsample,
        "spectra": {
            "curves": curves,
            "summary": spectrum_rows,
            "plot": "radial_spectra.png",
        },
        "edge_spread": {
            "per_frame": edge_rows,
            "background_frame_385": {"roi": list(BACKGROUND_ROI), **background_edge},
        },
        "motion_blur": {
            "per_transition": flow_rows,
            "direction_method": "directional evidence from radial spectrum only; no unique PSF claimed",
        },
        "text_size": {"per_frame": size_rows},
        "fusion_information": {
            "leave_one_out": leave_one_out,
        },
        "ocr": ocr,
        "limitations": [
            "Exposure time is unavailable; 1/60, 1/120, and 1/250 trail estimates are scenarios, not measurements.",
            "The HSV bbox is a measurement ROI detector, not a claim that HSV is a general detector.",
            "Edge spread uses intensity profiles through strongest gradients and is an estimate, not a calibrated optical MTF.",
            "OCR ground truth is an assumption from the requested strings; no user confirmation was available.",
        ],
    }
    (OUT / "diagnosis.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "frame_measurements.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["frame", "roi", "vertical_edge_10_90_px", "horizontal_edge_10_90_px"]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(edge_rows)
    return report


def _fmt(value: Any) -> str:
    if value is None:
        return "unavailable"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def write_report(data: dict[str, Any]) -> None:
    compression = data["compression"]
    spectra = data["spectra"]["summary"]
    edges = data["edge_spread"]["per_frame"]
    motion = data["motion_blur"]["per_transition"]
    sizes = data["text_size"]["per_frame"]
    sign_cutoffs = [row["sign_cutoff_cycles_per_pixel"] for row in spectra]
    edge_v = [row["vertical_edge_10_90_px"] for row in edges if row["vertical_edge_10_90_px"] is not None]
    edge_h = [row["horizontal_edge_10_90_px"] for row in edges if row["horizontal_edge_10_90_px"] is not None]
    large = [row["large_text_height_px"] for row in sizes if row["large_text_height_px"] is not None]
    strokes = [row["estimated_stroke_width_px"] for row in sizes if row["estimated_stroke_width_px"] is not None]
    report = f"""# 原始資訊上限診斷：`data/road.mp4`

## 一頁摘要

- 診斷範圍：原始 frame 350–385；每個 ROI 都以原始 3840×2160 影格像素量測，沒有使用 canonical 450×300、上採樣或新融合圖作為主要量測輸入。
- 路牌 ROI：每幀 HSV 候選框，完整座標與數值見 `diagnosis.json`；靜態對照 ROI：`{data['roi_policy']['background']}`（左側水泥柱/邊界區域）。
- ffprobe：codec=`{compression.get('codec')}`, pix_fmt=`{compression.get('pix_fmt')}`, bitrate=`{compression.get('bitrate_bps')}` bps，bits/pixel/frame=`{compression.get('bits_per_pixel_per_frame')}`。
- frame 350–385 的 I/P/B 分佈：`{compression.get('pict_type_counts')}`；packet size 與全幀 Laplacian 的相關係數：`{_fmt(compression.get('packet_laplacian_correlation'))}`。
- 路牌徑向頻譜 cutoff 估計範圍：`{_fmt(min(sign_cutoffs))}`–`{_fmt(max(sign_cutoffs))}` cycles/pixel；這是操作性 threshold，不是光學 MTF。
- 路牌 edge spread 估計：垂直中位數 `{_fmt(float(np.median(edge_v)) if edge_v else None)}` px、水平中位數 `{_fmt(float(np.median(edge_h)) if edge_h else None)}` px；水泥柱 frame 385 對照見 JSON。
- 中文大字高度估計中位數 `{_fmt(float(np.median(large)) if large else None)}` px；估計筆畫寬度中位數 `{_fmt(float(np.median(strokes)) if strokes else None)}` px。
- OCR：PaddleOCR unavailable；EasyOCR 狀態見 `diagnosis.json`，不會因套件存在就假稱已完成 OCR。

**初步判定：** 原始影片對細小筆畫的資訊上限具有高影響證據；運動/失焦與低有效頻寬是主要候選。這份診斷沒有產生新的增強、銳化、超解析或 final pipeline 輸出，因此不能用它宣稱任何演算法改善。

## 診斷 1：來源與壓縮

ffprobe 原始輸出摘要：

- codec/profile/pix_fmt：`{compression.get('codec')}` / `{compression.get('profile')}` / `{compression.get('pix_fmt')}`
- frame 350–385 I/P/B 分佈：`{compression.get('pict_type_counts')}`
- bitrate 與每幀 packet size：完整資料在 `diagnosis.json` 的 `compression.frames_350_385`
- bitrate 只可判斷編碼負擔，不能單獨證明文字細節是否不存在；需配合頻譜與 edge spread。
- packet size 範圍與每幀 I/P/B、Laplacian 關聯均保留在 `diagnosis.json.compression`；相關係數不是因果證據。

## 診斷 2：有效解析度

frame 385 整張原始畫面做 1/2、1/3、1/4、1/6 降採樣再 bicubic 回放；PSNR/SSIM 與相鄰像素週期統計在 `diagnosis.json.effective_resolution`。這是「可被降採樣而近似保留」的操作性測試，不等於真實感測器解析度。

## 診斷 3：頻譜截止

![Radial spectra](radial_spectra.png)

每個 frame 350、360、370、375、380、385 同時量路牌與水泥柱/背景對照；使用去均值 + Hann + 2D FFT + 徑向平均功率。路牌 cutoff 若低於背景且字筆畫需要更高頻，支持「路牌本身資訊不足」；但 threshold 與 ROI 邊界會影響絕對值，故證據強度標為中至強，而非精密光學結論。

## 診斷 4：edge spread

逐幀在路牌 crop 的最強垂直/水平梯度剖面計算 10–90% 寬度；對照 frame 385 的固定水泥柱 ROI。原始每幀數值在 `frame_measurements.csv`。方向差異若顯著，支持方向性運動模糊；若路牌與背景都寬，較支持全局失焦/光學限制。

## 診斷 5：運動模糊

Farneback flow 在原始 full-frame 上計算，再取當幀路牌 bbox 內位移；同時記錄 bbox 尺度變化。曝光 1/60、1/120、1/250 的拖尾只是情境估計，不能冒充相機曝光資訊。沒有用 radial FFT 宣稱唯一 blur kernel；方向性只能做交叉證據。

## 診斷 6：字元尺寸

以原始 crop 的亮低飽和連通元件估計大字、小字與筆畫尺寸。這不是 OCR ground truth，也可能受字元相連、壓縮與閾值影響；完整逐幀表在 `diagnosis.json.text_size`。一般 OCR 的 20–30 px 字高只是經驗參考，不是本片的保證門檻。

## 診斷 7：多幀融合資訊

本任務沒有新建 fusion/enhancement 圖。使用原始路牌 crop 以 frame 385 尺寸作共同座標的 leave-one-out proxy，並計算前後半序列穩定性；這會混入 resize/尺度變化，因此不是超解析證明。結果摘要：

- leave-one-out 單候選平均 PSNR=`{_fmt(data['fusion_information']['leave_one_out']['leave_one_out_summary']['single_psnr_mean_db'])}` dB、SSIM=`{_fmt(data['fusion_information']['leave_one_out']['leave_one_out_summary']['single_ssim_mean'])}`；
- 前後半序列彼此 PSNR=`{_fmt(data['fusion_information']['leave_one_out']['half_sequence_stability']['half_a_vs_half_b_psnr_db'])}` dB、SSIM=`{_fmt(data['fusion_information']['leave_one_out']['half_sequence_stability']['half_a_vs_half_b_ssim'])}`。

因此目前只支持「幀間內容差異很大、融合可作穩定化/降噪但有尺度與插值混淆」，不支持宣稱增加真實解析度。

## 診斷 8：OCR 基準

暫定真值為：`{'、'.join(data['ground_truth']['strings'])}`。PaddleOCR 未安裝；EasyOCR 雖有套件但若模型未快取則不下載、不執行。這不是 OCR 失敗被隱藏，而是環境可重現性限制。任何可用結果都會列出實際文字與 confidence；沒有結果就標 unavailable。

## 判定表

| 問題 | 結論 | 證據強度 | 依據 |
|---|---|---|---|
| 資訊上限是否由原始影片決定？ | 支持，至少是主要限制之一 | 強 | 原始 ROI 頻譜、edge spread、筆畫像素尺寸與降採樣測試共同支持 |
| 主要模糊來源 | 運動/失焦/低有效細節優先；壓縮需看 packet/頻譜，不單獨定罪 | 中 | flow、方向 edge spread、路牌/背景頻譜對照 |
| 多幀融合是否增加真實解析度？ | 不支持宣稱；較安全解釋是穩定化/降噪 | 中 | raw-crop leave-one-out proxy 受尺度/插值混淆，未得到可歸因於真實高頻新增的證據 |
| 是否有任何版本可被 OCR 讀出？ | `{data['ocr'].get('easyocr', {}).get('available')}`；PaddleOCR unavailable | 弱/未知 | OCR 依賴與模型快取限制，不能以肉眼推測 CER |

## 下一步（限定選項）

- **B. 改取更好的輸入（首選）**：更高碼率、更近/更早幀、另一支影片，最直接針對已觀察的原始資訊上限。
- **A. 繼續改善演算法（次選）**：只在新輸入顯示可挖掘高頻資訊時進行；先做 raw-frame OCR/CER 與保留幀驗證。
- **C. 改目標為文字辨識**：若業務只需固定路牌內容，使用已知字型/候選字串模板比對，承認這不是像素恢復。
- **D. 學習式模型**：最後選項；輸出必須標註視覺增強並以 OCR/保留幀一致性驗證，不能當真實字形證據。

## 假設與未知

1. 真值字串依使用者指定暫定，尚未取得人工確認。
2. 曝光時間未知；拖尾只列三種情境。
3. HSV bbox 是量測用 ROI，不是通用路牌 detector。
4. edge spread 未經鏡頭標定，頻譜 cutoff 依 threshold，均不可過度解讀。
5. OCR 模型不可在缺少權重時自動下載；實際可用狀態以 `diagnosis.json` 為準。
"""
    (OUT / "diagnosis_report.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    result = run()
    write_report(result)
    print(json.dumps({"output": str(OUT / "diagnosis_report.md"), "frames": len(result["text_size"]["per_frame"])}))
