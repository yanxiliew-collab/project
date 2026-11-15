import numpy as np

from preprocess.eval.contrast_only import _contrast_only, _directional_fft


def test_contrast_only_preserves_shape_and_color_contract() -> None:
    image = np.zeros((40, 60, 3), dtype=np.uint8)
    image[:, :, 1] = 90
    image[12:28, 20:40] = (220, 220, 220)

    enhanced = _contrast_only(image)

    assert enhanced.shape == image.shape
    assert enhanced.dtype == np.uint8
    assert enhanced.ndim == 3


def test_directional_fft_reports_energy_bins() -> None:
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, 20:24] = 255

    result = _directional_fft(image)

    assert len(result["energies"]) == 18
    assert result["anisotropy_ratio"] >= 1.0
