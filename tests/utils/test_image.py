import os
import tempfile

import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from shap.utils.image import (
    check_valid_image,
    display_grid_plot,
    is_empty,
    load_image,
    make_dir,
    resize_image,
    save_image,
)

# ── is_empty ──────────────────────────────────────────────────────────────────


def test_is_empty_nonexistent_path():
    assert is_empty("/nonexistent/path/xyz") is True


def test_is_empty_empty_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        assert is_empty(tmpdir) is True


def test_is_empty_nonempty_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        open(os.path.join(tmpdir, "file.txt"), "w").close()
        assert is_empty(tmpdir) is False


def test_is_empty_file_path():
    with tempfile.NamedTemporaryFile() as f:
        assert is_empty(f.name) is True


# ── make_dir ──────────────────────────────────────────────────────────────────


def test_make_dir_creates_new_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        new_dir = os.path.join(tmpdir, "new_folder")
        make_dir(new_dir)
        assert os.path.exists(new_dir)


def test_make_dir_empties_existing_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        filepath = os.path.join(tmpdir, "file.txt")
        open(filepath, "w").close()
        make_dir(tmpdir + "/")
        assert os.listdir(tmpdir) == []


def test_make_dir_existing_empty_directory():
    with tempfile.TemporaryDirectory() as tmpdir:
        make_dir(tmpdir)
        assert os.path.exists(tmpdir)


# ── check_valid_image ─────────────────────────────────────────────────────────


@pytest.mark.parametrize("ext", [".png", ".jpg", ".jpeg", ".gif", ".bmp", ".jfif"])
def test_check_valid_image_valid_extensions(ext):
    assert check_valid_image(f"image{ext}") is True


def test_check_valid_image_invalid_extension():
    assert check_valid_image("file.txt") is None


def test_check_valid_image_no_extension():
    assert check_valid_image("file") is None


# ── save_image / load_image ───────────────────────────────────────────────────


def test_save_and_load_image():
    arr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.png")
        save_image(arr, path)
        assert os.path.exists(path)
        loaded = load_image(path)
        assert loaded.shape == (50, 50, 3)


def test_load_image_returns_float():
    arr = np.random.randint(0, 256, (30, 30, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.png")
        save_image(arr, path)
        loaded = load_image(path)
        assert loaded.dtype == float


# ── resize_image ──────────────────────────────────────────────────────────────


def test_resize_image_large_square():
    arr = np.random.randint(0, 256, (600, 600, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "large.png")
        save_image(arr, src)
        resized, path = resize_image(src, tmpdir)
        assert resized.shape[0] <= 500
        assert resized.shape[1] <= 500
        assert path is not None


def test_resize_image_tall():
    arr = np.random.randint(0, 256, (800, 200, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "tall.png")
        save_image(arr, src)
        resized, path = resize_image(src, tmpdir)
        assert resized.shape[0] <= 500


def test_resize_image_wide():
    arr = np.random.randint(0, 256, (200, 800, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "wide.png")
        save_image(arr, src)
        resized, path = resize_image(src, tmpdir)
        assert resized.shape[1] <= 500


def test_resize_image_small_no_reshape():
    arr = np.random.randint(0, 256, (100, 100, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        src = os.path.join(tmpdir, "small.png")
        save_image(arr, src)
        resized, path = resize_image(src, tmpdir)
        assert path is None
        assert resized.shape == (100, 100, 3)


# ── display_grid_plot ─────────────────────────────────────────────────────────


def test_display_grid_plot_with_captions():
    arr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "img.png")
        save_image(arr, path)
        display_grid_plot(["caption"], [path])


def test_display_grid_plot_without_captions():
    arr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "img.png")
        save_image(arr, path)
        display_grid_plot([], [path])


def test_display_grid_plot_exceeds_max_columns():
    arr = np.random.randint(0, 256, (50, 50, 3), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as tmpdir:
        paths = []
        for i in range(6):
            p = os.path.join(tmpdir, f"img{i}.png")
            save_image(arr, p)
            paths.append(p)
        display_grid_plot(["cap"] * 6, paths, max_columns=2)
