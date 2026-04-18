import numpy as np
import shap.utils.image as img
import cv2
import os


def test_resize_image_behavior():
    dummy = np.ones((10, 10, 3), dtype=np.uint8) * 255
    path = "test_img.png"
    cv2.imwrite(path, dummy)

    result = img.resize_image(path, (5, 5))

    # check it returns tuple
    assert isinstance(result, tuple)

    resized = result[0]

    # check output is numpy array
    assert isinstance(resized, np.ndarray)

    os.remove(path)