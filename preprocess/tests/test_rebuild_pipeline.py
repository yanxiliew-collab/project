import cv2
import numpy as np

from preprocess.eval.rebuild_pipeline import _order_points, _rectify


def test_order_points_returns_clockwise_corners() -> None:
    points = np.array([[90, 80], [10, 10], [100, 10], [10, 90]], dtype=np.float32)

    ordered = _order_points(points)

    assert np.allclose(ordered[0], [10, 10])
    assert np.allclose(ordered[1], [100, 10])
    assert np.allclose(ordered[2], [90, 80])
    assert np.allclose(ordered[3], [10, 90])


def test_rectify_returns_canonical_contract() -> None:
    image = np.zeros((160, 220, 3), dtype=np.uint8)
    cv2.rectangle(image, (35, 30), (185, 130), (40, 130, 80), -1)

    result, method = _rectify(image, (25, 20, 195, 140))

    assert result.shape == (300, 450, 3)
    assert result.dtype == np.uint8
    assert method in {"four_corner_homography", "padded_bbox_resize"}
