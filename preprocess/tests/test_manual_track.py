import cv2
import numpy as np

from preprocess.eval.manual_track import manual_track


def _write_video(path, frame_count=5) -> None:
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        10.0,
        (40, 30),
    )
    assert writer.isOpened()
    for index in range(frame_count):
        frame = np.full((30, 40, 3), index * 10, dtype=np.uint8)
        writer.write(frame)
    writer.release()


class _FakeTracker:
    def __init__(self) -> None:
        self.updates = 0

    def init(self, _frame, _bbox):
        return True

    def update(self, _frame):
        self.updates += 1
        if self.updates == 3:
            return False, (0, 0, 1, 1)
        return True, (-2, -1, 20, 15)


def test_manual_track_clips_bbox_limits_history_and_logs_stop(tmp_path, caplog) -> None:
    video = tmp_path / "input.avi"
    _write_video(video)

    result = manual_track(
        video,
        max_frames_per_track=2,
        roi_selector=lambda _name, _frame: (5, 4, 10, 8),
        tracker_factory=_FakeTracker,
    )

    assert len(result[0]) == 2
    assert result[0][0].frame_idx == 1
    assert result[0][1].frame_idx == 2
    assert all(frame.bbox == (0, 0, 18, 14) for frame in result[0])
    assert "stopped at frame 3" in caplog.text


def test_manual_track_omits_single_frame_track(tmp_path) -> None:
    video = tmp_path / "input.avi"
    _write_video(video, frame_count=1)

    result = manual_track(
        video,
        roi_selector=lambda _name, _frame: (5, 4, 10, 8),
        tracker_factory=_FakeTracker,
    )

    assert result == {}
