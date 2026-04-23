
import sys
import types

# Mock shap to avoid loading full package
sys.modules['shap'] = types.ModuleType("shap")

import os
import sys
import numpy as np
import pytest

# Add path to directly access image.py (bypass shap import)
sys.path.append(os.path.abspath("shap/utils"))

from image import (
    check_valid_image,
    is_empty,
    make_dir,
    save_image,
    resize_image
)

def test_check_valid_image():
    assert check_valid_image("image.jpg") == True
    assert check_valid_image("image.png") == True
    assert check_valid_image("file.txt") is None


def test_is_empty_with_empty_dir(tmp_path):
    assert is_empty(tmp_path) == True


def test_is_empty_with_files(tmp_path):
    file = tmp_path / "test.jpg"
    file.write_text("data")
    assert is_empty(tmp_path) == False


def test_make_dir_creates_directory(tmp_path):
    new_dir = tmp_path / "new_folder"
    make_dir(new_dir)
    assert os.path.exists(new_dir)


def test_make_dir_clears_existing_files(tmp_path):
    file = tmp_path / "file.txt"
    file.write_text("data")
    make_dir(tmp_path)
    assert len(os.listdir(tmp_path)) == 0


def test_save_image(tmp_path):
    img = np.zeros((10, 10, 3))
    file_path = tmp_path / "test.png"
    save_image(img, file_path)
    assert os.path.exists(file_path)


def test_resize_image_no_resize(tmp_path):
    img = np.zeros((100, 100, 3))
    file_path = tmp_path / "img.png"

    import matplotlib.pyplot as plt
    plt.imsave(file_path, img)

    resized_img, path = resize_image(file_path, tmp_path)

    assert resized_img.shape == (100, 100, 3)
    assert path is None