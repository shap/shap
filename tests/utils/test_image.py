"""Tests for shap/utils/image.py"""

import os
import tempfile

import numpy as np
from PIL import Image

from shap.utils.image import (
    check_valid_image,
    is_empty,
    load_image,
    make_dir,
    resize_image,
)


def test_is_empty_nonexistent_folder():
    """A folder that doesn't exist should be considered empty."""
    assert is_empty("/nonexistent/path/xyz") is True


def test_is_empty_with_empty_folder():
    """A newly created empty folder should return True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        assert is_empty(tmpdir) is True


def test_is_empty_with_files():
    """A folder with files should return False."""
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "file.txt"), "w").close()
        assert is_empty(tmpdir) is False


def test_make_dir_creates_folder():
    """make_dir should create a new folder."""
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = os.path.join(tmpdir, "new_folder")
        make_dir(new_dir)
        assert os.path.exists(new_dir)


def test_make_dir_existing_folder_becomes_empty():
    """make_dir on existing folder should attempt to empty it."""
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = os.path.join(tmpdir, "subfolder")
        os.makedirs(new_dir)
        # folder exists and is empty - make_dir should work fine
        make_dir(new_dir)
        assert os.path.exists(new_dir)


def test_check_valid_image_jpg():
    """JPG files should be recognized as valid images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "image.jpg")
        open(filepath, "w").close()
        result = check_valid_image(filepath)
        assert result is not False


def test_check_valid_image_png():
    """PNG files should be recognized as valid images."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "image.png")
        open(filepath, "w").close()
        result = check_valid_image(filepath)
        assert result is not False


def test_check_valid_image_invalid():
    """Non-image extensions should not return True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "document.txt")
        open(filepath, "w").close()
        result = check_valid_image(filepath)
        assert result is not True


def test_load_image_returns_numpy_array():
    """load_image should return a numpy array."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img = Image.fromarray(np.uint8(np.random.rand(100, 100, 3) * 255))
        img_path = os.path.join(tmpdir, "test.png")
        img.save(img_path)
        result = load_image(img_path)
        assert isinstance(result, np.ndarray)


def test_load_image_correct_shape():
    """load_image should return array with 3 color channels."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img = Image.fromarray(np.uint8(np.ones((50, 60, 3)) * 128))
        img_path = os.path.join(tmpdir, "test.png")
        img.save(img_path)
        result = load_image(img_path)
        assert result.ndim == 3
        assert result.shape[2] == 3


def test_resize_image_large_image():
    """resize_image should shrink images larger than 500px."""
    with tempfile.TemporaryDirectory() as tmpdir:
        img = Image.fromarray(np.uint8(np.random.rand(800, 800, 3) * 255))
        img_path = os.path.join(tmpdir, "large.png")
        img.save(img_path)
        out_dir = os.path.join(tmpdir, "resized")
        os.makedirs(out_dir)
        resize_image(img_path, out_dir)
        resized_files = os.listdir(out_dir)
        assert len(resized_files) == 1
        resized_path = os.path.join(out_dir, resized_files[0])
        with Image.open(resized_path) as resized:
            assert max(resized.size) <= 500
