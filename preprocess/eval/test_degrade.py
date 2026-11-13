import cv2
import numpy as np

from preprocess.eval.degrade import degrade, make_track


def _image() -> np.ndarray:
    image = np.zeros((80, 100, 3), dtype=np.uint8)
    cv2.rectangle(image, (20, 20), (80, 60), (220, 180, 40), -1)
    cv2.line(image, (0, 0), (99, 79), (255, 255, 255), 2)
    return image


def test_degrade_preserves_shape_and_dtype() -> None:
    result = degrade(
        _image(),
        gamma=1.4,
        brightness_scale=0.8,
        gaussian_sigma=1.2,
        motion_length=5,
        motion_angle=25,
        noise_std=4,
        downsample_scale=0.5,
        seed=3,
    )

    assert result.shape == (80, 100, 3)
    assert result.dtype == np.uint8


def test_degrade_and_make_track_are_seed_reproducible() -> None:
    image = _image()
    first = degrade(image, noise_std=8, seed=10)
    second = degrade(image, noise_std=8, seed=10)
    frames_a, matrices_a = make_track(image, 3, seed=10)
    frames_b, matrices_b = make_track(image, 3, seed=10)

    assert np.array_equal(first, second)
    assert all(np.array_equal(a.full_frame, b.full_frame) for a, b in zip(frames_a, frames_b))
    assert all(np.array_equal(a, b) for a, b in zip(matrices_a, matrices_b))
    assert [frame.timestamp for frame in frames_a] == [
        frame.timestamp for frame in frames_b
    ]


def test_true_transform_restores_frame_near_original() -> None:
    image = _image()
    frames, matrices = make_track(image, 1, seed=7)
    restored = cv2.warpAffine(
        frames[0].full_frame,
        cv2.invertAffineTransform(matrices[0]),
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT,
    )

    assert np.mean(np.abs(restored.astype(np.float32) - image)) < 18.0
