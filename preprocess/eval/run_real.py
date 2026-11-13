"""CLI harness for evaluating preprocessing modes on tracked video."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from time import perf_counter
from typing import Optional, Sequence

import cv2

from common.types import TrackFrame
from preprocess.config import PreprocessConfig
from preprocess.core.pipeline import preprocess
from preprocess.eval.track_source import Detector, iter_tracks


MODES = ("passthrough", "best_frame", "enhance", "fusion")


def _save_image(path: Path, image) -> Optional[str]:
    if image is None:
        return "no image returned"
    if not cv2.imwrite(str(path), image):
        return "cv2.imwrite failed"
    return None


def _row_for_result(
    track_id: int,
    mode: str,
    max_frames: int,
    frames: Sequence[TrackFrame],
    result,
    elapsed_ms: float,
) -> dict[str, object]:
    diagnostics = result.diagnostics
    accepted = diagnostics.get("accepted")
    used = len(accepted) if isinstance(accepted, list) else (1 if result.image is not None else 0)
    used = min(used, len(frames))
    reasons = []
    for item in diagnostics.get("rejected", []):
        reasons.append(
            f"{item.get('frame_idx', '?')}:{item.get('reason', 'rejected')}"
        )
    if len(frames) > used and not reasons:
        reasons.append("not_selected")
    if diagnostics.get("fallback"):
        reasons.append(diagnostics.get("error", "fallback"))
    if diagnostics.get("fallback_best_frame"):
        reasons.append("fallback_best_frame")
    return {
        "track_id": track_id,
        "mode": mode,
        "N": max_frames,
        "input_frames": len(frames),
        "used_frames": used,
        "discarded_frames": max(0, len(frames) - used),
        "discarded_reasons": json.dumps(reasons, ensure_ascii=False),
        "elapsed_ms": elapsed_ms,
        "fallback": bool(diagnostics.get("fallback") or diagnostics.get("fallback_best_frame")),
    }


def run_evaluation(
    video: str | Path,
    weights: str,
    output: str | Path = "outputs/real",
    max_frames: Sequence[int] = (3, 5),
    detector: Optional[Detector] = None,
) -> list[dict[str, object]]:
    """Run all requested modes and write output images plus ``results.csv``."""

    output_path = Path(output)
    output_path.mkdir(parents=True, exist_ok=True)
    max_n = max(max_frames)
    tracks = iter_tracks(
        video,
        weights,
        max_frames_per_track=max_n,
        detector=detector,
    )
    rows: list[dict[str, object]] = []
    for track_id, track_frames in tracks.items():
        track_dir = output_path / f"track_{track_id}"
        track_dir.mkdir(parents=True, exist_ok=True)
        for frame in track_frames:
            x1, y1, x2, y2 = frame.bbox
            _save_image(
                track_dir / f"input_{frame.frame_idx}.png",
                frame.full_frame[y1:y2, x1:x2],
            )
        for mode in MODES:
            for n in max_frames:
                frames = track_frames[-n:]
                started = perf_counter()
                try:
                    result = preprocess(
                        frames,
                        PreprocessConfig(mode=mode, max_frames=n),
                    )
                    elapsed_ms = (perf_counter() - started) * 1000.0
                    row = _row_for_result(track_id, mode, n, frames, result, elapsed_ms)
                    save_error = _save_image(
                        track_dir / f"{mode}_n{n}.png", result.image
                    )
                    if save_error:
                        row["fallback"] = True
                        row["discarded_reasons"] = json.dumps(
                            json.loads(str(row["discarded_reasons"])) + [save_error],
                            ensure_ascii=False,
                        )
                except Exception as exc:
                    elapsed_ms = (perf_counter() - started) * 1000.0
                    row = {
                        "track_id": track_id,
                        "mode": mode,
                        "N": n,
                        "input_frames": len(frames),
                        "used_frames": 0,
                        "discarded_frames": len(frames),
                        "discarded_reasons": json.dumps(
                            [f"{type(exc).__name__}: {exc}"], ensure_ascii=False
                        ),
                        "elapsed_ms": elapsed_ms,
                        "fallback": True,
                    }
                rows.append(row)

    fields = [
        "track_id",
        "mode",
        "N",
        "input_frames",
        "used_frames",
        "discarded_frames",
        "discarded_reasons",
        "elapsed_ms",
        "fallback",
    ]
    with (output_path / "results.csv").open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--video", required=True)
    parser.add_argument("--weights", required=True)
    parser.add_argument("--out", default="outputs/real")
    parser.add_argument("--max-frames", nargs="+", type=int, default=[3, 5])
    args = parser.parse_args(argv)
    run_evaluation(args.video, args.weights, args.out, args.max_frames)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
