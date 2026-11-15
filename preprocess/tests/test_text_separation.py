import cv2
import numpy as np

from preprocess.eval.run_text_separation import _glare_mask, _text_features


def test_text_features_separate_bright_text_from_green_sign() -> None:
    image = np.full((120, 180, 3), (40, 130, 80), dtype=np.uint8)
    cv2.putText(image, "EXIT", (25, 75), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (245, 245, 245), 3)

    focused, foreground, stats = _text_features(image)

    assert focused.shape == image.shape
    assert foreground.dtype == np.uint8
    assert stats["foreground_ratio"] > 0.0001
    assert stats["foreground_ratio"] < 0.5
    assert foreground[35:80, 20:160].mean() > 0


def test_glare_mask_excludes_neutral_highlight() -> None:
    image = np.full((80, 100, 3), (40, 130, 80), dtype=np.uint8)
    image[20:35, 40:60] = (255, 255, 255)

    mask = _glare_mask(image)

    assert mask[25, 50] > 0
    assert mask[5, 5] == 0
