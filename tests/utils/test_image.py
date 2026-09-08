import os

import numpy as np
import pytest

from shap.utils.image import (
    check_valid_image,
    is_empty,
    load_image,
    make_dir,
    save_image,
)


class TestIsEmpty:
    def test_nonexistent_directory(self):
        assert is_empty("/nonexistent/path/that/does/not/exist") is True

    def test_empty_directory(self, tmp_path):
        assert is_empty(str(tmp_path)) is True

    def test_non_empty_directory(self, tmp_path):
        (tmp_path / "file.txt").write_text("content")
        assert is_empty(str(tmp_path)) is False


class TestMakeDir:
    def test_creates_new_directory(self, tmp_path):
        new_dir = str(tmp_path / "new_folder")
        make_dir(new_dir)
        assert os.path.isdir(new_dir)

    def test_empties_existing_directory(self, tmp_path):
        (tmp_path / "file.txt").write_text("content")
        make_dir(str(tmp_path))
        assert os.path.isdir(str(tmp_path))


class TestCheckValidImage:
    @pytest.mark.parametrize(
        "filename",
        [
            "photo.png",
            "photo.jpg",
            "photo.jpeg",
            "photo.gif",
            "photo.bmp",
            "photo.jfif",
        ],
    )
    def test_valid_extensions(self, filename):
        check_valid_image(filename)

    def test_invalid_extension(self):
        with pytest.raises(Exception):
            check_valid_image("document.pdf")


class TestSaveAndLoadImage:
    def test_save_creates_file(self, tmp_path):
        img_array = np.ones((100, 100, 3)) * 128.0
        img_path = str(tmp_path / "output.png")
        save_image(img_array, img_path)
        assert os.path.isfile(img_path)
        assert os.path.getsize(img_path) > 0

    def test_save_and_load_roundtrip(self, tmp_path):
        img_array = np.random.randint(0, 255, (50, 50, 3)).astype(np.float64)
        img_path = str(tmp_path / "test_image.png")
        save_image(img_array, img_path)
        assert os.path.exists(img_path)
        loaded = load_image(img_path)
        assert loaded is not None
        assert loaded.shape[0] > 0
        assert loaded.shape[1] > 0
