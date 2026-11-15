import csv

import cv2
import numpy as np

from preprocess.eval.run_real import run_evaluation
from preprocess.eval.track_source import _sign_candidates, iter_tracks


def _write_video(path, frame_count=7) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (64, 48),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.zeros((48, 64, 3), dtype=np.uint8)
        cv2.rectangle(frame, (8 + index % 3, 10), (42, 35), (80, 160, 220), -1)
        cv2.putText(frame, str(index), (2, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.4, 255, 1)
        writer.write(frame)
    writer.release()


def _detector_with_short_track():
    calls = 0

    def detector(_frame):
        nonlocal calls
        calls += 1
        detections = [
            {"track_id": 1, "bbox": (-4, 3, 70, 50), "conf": 0.9, "cls": 0}
        ]
        if calls == 1:
            detections.append(
                {"track_id": 2, "bbox": (1, 1, 5, 5), "conf": 0.9, "cls": 0}
            )
        return detections

    return detector


def test_iter_tracks_drop_history_bbox_and_short_track(tmp_path) -> None:
    video = tmp_path / "input.avi"
    _write_video(video)

    tracks = iter_tracks(
        video,
        weights="unused.pt",
        max_frames_per_track=3,
        drop_every=3,
        detector=_detector_with_short_track(),
    )

    assert set(tracks) == {1}
    frames = tracks[1]
    assert len(frames) == 3
    timestamps = [frame.timestamp for frame in frames]
    assert any(
        not np.isclose(current - previous, 0.1)
        for previous, current in zip(timestamps, timestamps[1:])
    )
    for frame in frames:
        assert frame.bbox == (0, 3, 64, 48)
        assert all(isinstance(value, int) for value in frame.bbox)
        x1, y1, x2, y2 = frame.bbox
        assert 0 <= x1 <= x2 <= 64
        assert 0 <= y1 <= y2 <= 48


def test_sign_candidates_find_colored_sign_region() -> None:
    frame = np.zeros((100, 140, 3), dtype=np.uint8)
    cv2.rectangle(frame, (50, 25), (90, 65), (0, 0, 255), -1)

    detections = _sign_candidates(frame)

    assert detections
    assert any(
        detection["bbox"][0] <= 50
        and detection["bbox"][1] <= 25
        and detection["bbox"][2] >= 90
        and detection["bbox"][3] >= 65
        for detection in detections
    )


def test_track_source_single_occurrence_is_ignored(tmp_path) -> None:
    video = tmp_path / "input.avi"
    _write_video(video, frame_count=2)
    calls = 0

    def detector(_frame):
        nonlocal calls
        calls += 1
        if calls == 1:
            return [{"track_id": 7, "bbox": (5, 5, 20, 20), "conf": 0.9}]
        return []

    tracks = iter_tracks(video, "unused.pt", detector=detector)
    assert 7 not in tracks


def test_run_real_smoke_writes_csv_and_images(tmp_path) -> None:
    video = tmp_path / "input.avi"
    output = tmp_path / "outputs"
    _write_video(video, frame_count=5)

    def detector(_frame):
        return [{"track_id": 1, "bbox": (5, 5, 50, 40), "conf": 0.9, "cls": 0}]

    rows = run_evaluation(video, "unused.pt", output, [2], detector=detector)

    assert len(rows) == 4
    csv_path = output / "results.csv"
    assert csv_path.exists()
    with csv_path.open(encoding="utf-8", newline="") as file:
        saved_rows = list(csv.DictReader(file))
    assert len(saved_rows) == 4
    assert {row["mode"] for row in saved_rows} == {
        "passthrough",
        "best_frame",
        "enhance",
        "fusion",
    }
    assert all((output / "track_1" / f"{mode}_n2.png").exists() for mode in (
        "passthrough",
        "best_frame",
        "enhance",
        "fusion",
    ))
