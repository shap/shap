import os
import tempfile

import numpy as np

import shap.utils.image as img


def test_resize_image_behavior():
    dummy = np.ones((10, 10, 3), dtype=np.uint8) * 255

    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        path = tmp.name

    try:
        import cv2

        cv2.imwrite(path, dummy)

        result = img.resize_image(path, (5, 5))

        assert isinstance(result, tuple)

        resized = result[0]

        assert isinstance(resized, np.ndarray)
        assert resized.ndim == 3
        assert resized.shape[2] == 3  # RGB channels

    finally:
        os.remove(path)


def test_resize_image_invalid_path():
    import pytest

    with pytest.raises(Exception):
        img.resize_image("invalid_path.png", (5, 5))
