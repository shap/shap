import pytest
import numpy as np
import os
import tempfile
import matplotlib.pyplot as plt
import cv2

from shap.utils.image import (
    is_empty,
    make_dir,
    add_sample_images,
    load_image,
    check_valid_image,
    save_image,
    resize_image,
    display_grid_plot,
)


# ================== is_empty ==================

def test_is_empty_cases(tmp_path):
    # empty dir
    assert is_empty(tmp_path) is True

    # dir with file
    (tmp_path / "file.txt").write_text("data")
    assert is_empty(tmp_path) is False

    # nonexistent
    assert is_empty("nonexistent_path_123") is True


# ================== make_dir ==================

def test_make_dir_create_and_clear(tmp_path):
    new_dir = tmp_path / "new"
    make_dir(str(new_dir))
    assert new_dir.exists()

    # add files then clear
    (new_dir / "file.txt").write_text("data")
    make_dir(str(new_dir))
    assert list(new_dir.iterdir()) == []


# ================== check_valid_image ==================

@pytest.mark.parametrize("filename", [
    "a.png", "b.jpg", "c.jpeg", "d.gif", "e.bmp", "f.jfif"
])
def test_check_valid_image_valid(filename):
    assert check_valid_image(filename) is True


@pytest.mark.parametrize("filename", ["a.txt", "b", "c.pdf"])
def test_check_valid_image_invalid(filename):
    assert check_valid_image(filename) is None


# ================== save + load ==================

def test_save_and_load_image(tmp_path):
    img = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    path = tmp_path / "img.jpg"

    save_image(img, str(path))
    assert path.exists()

    loaded = load_image(str(path))
    assert isinstance(loaded, np.ndarray)
    assert loaded.shape == (100, 100, 3)


def test_load_image_invalid():
    with pytest.raises((cv2.error, AttributeError, FileNotFoundError)):
        load_image("invalid_path.jpg")


# ================== resize_image ==================

def test_resize_image_no_resize(tmp_path):
    img = np.random.randint(0, 256, (200, 200, 3), dtype=np.uint8)
    path = tmp_path / "small.jpg"
    save_image(img, str(path))

    out, new_path = resize_image(str(path), str(tmp_path))
    assert new_path is None
    assert out.shape[:2] == (200, 200)


def test_resize_image_large_square(tmp_path):
    img = np.random.randint(0, 256, (600, 600, 3), dtype=np.uint8)
    path = tmp_path / "large.jpg"
    save_image(img, str(path))

    out, new_path = resize_image(str(path), str(tmp_path))
    assert new_path is not None
    assert out.shape[0] == 500 and out.shape[1] == 500


def test_resize_aspect_ratio(tmp_path):
    img = np.random.randint(0, 256, (400, 800, 3), dtype=np.uint8)
    path = tmp_path / "wide.jpg"
    save_image(img, str(path))

    out, _ = resize_image(str(path), str(tmp_path))
    ratio = out.shape[1] / out.shape[0]
    assert abs(ratio - 2.0) < 0.2


# ================== display ==================

def test_display_grid_plot_runs(tmp_path):
    img = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    path = tmp_path / "img.jpg"
    save_image(img, str(path))

    # avoid UI pop
    with pytest.MonkeyPatch().context() as m:
        m.setattr("matplotlib.pyplot.show", lambda: None)
        display_grid_plot(["caption"], [str(path)])


# ================== add_sample_images ==================

def test_add_sample_images(monkeypatch, tmp_path):
    def fake_dataset():
        return np.random.randint(0, 256, (50, 50, 50, 3), dtype=np.uint8), None

    monkeypatch.setattr("shap.datasets.imagenet50", fake_dataset)

    add_sample_images(str(tmp_path))
    assert len(list(tmp_path.iterdir())) == 4


# ================== integration ==================

def test_full_workflow(tmp_path):
    img = np.random.randint(0, 256, (600, 600, 3), dtype=np.uint8)

    path = tmp_path / "img.jpg"
    save_image(img, str(path))

    loaded = load_image(str(path))
    assert loaded.shape == (600, 600, 3)

    resized, new_path = resize_image(str(path), str(tmp_path))
    assert resized.shape[0] <= 500

    assert os.path.exists(path)