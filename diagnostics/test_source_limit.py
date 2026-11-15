import cv2
import numpy as np

from diagnostics.source_limit import _downsample_test, _edge_spread, _spectrum_summary, _text_size


def test_source_measurements_are_finite() -> None:
    image = np.zeros((120, 180, 3), dtype=np.uint8)
    cv2.rectangle(image, (25, 30), (155, 90), (40, 150, 80), -1)
    cv2.line(image, (90, 30), (90, 90), (240, 240, 240), 3)

    downsample = _downsample_test(image)
    spectrum = _spectrum_summary(image)
    edge = _edge_spread(image)
    sizes = _text_size(image)

    assert len(downsample["round_trip"]) == 4
    assert 0.0 <= spectrum["cutoff_cycles_per_pixel"] <= 0.5
    assert edge["vertical_edge_10_90_px"] is None or edge["vertical_edge_10_90_px"] >= 0.0
    assert sizes["sign_width_px"] == 180.0
