import os

import matplotlib.pyplot as plt
import numpy as np
import pytest

pytest.importorskip("cv2")

import shap
import shap.utils.image as image_utils


def test_is_empty_for_missing_path(capsys, tmp_path):
    missing_path = tmp_path / "missing"

    assert image_utils.is_empty(str(missing_path)) is True
    assert "There is no 'test_images' folder" in capsys.readouterr().out


def test_is_empty_for_empty_and_non_empty_directories(capsys, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    assert image_utils.is_empty(str(empty_dir)) is True
    assert "'test_images' folder is empty" in capsys.readouterr().out

    non_empty_dir = tmp_path / "non_empty"
    non_empty_dir.mkdir()
    (non_empty_dir / "image.png").write_text("content")

    assert image_utils.is_empty(str(non_empty_dir)) is False
    assert capsys.readouterr().out == ""


def test_make_dir_creates_missing_directory(tmp_path):
    new_dir = tmp_path / "created"

    image_utils.make_dir(str(new_dir))

    assert new_dir.is_dir()


def test_make_dir_clears_existing_files(tmp_path):
    existing_dir = tmp_path / "existing"
    existing_dir.mkdir()
    (existing_dir / "first.txt").write_text("first")
    (existing_dir / "second.txt").write_text("second")

    image_utils.make_dir(f"{existing_dir}{os.sep}")

    assert existing_dir.is_dir()
    assert list(existing_dir.iterdir()) == []


def test_add_sample_images_saves_expected_dataset_entries(monkeypatch, tmp_path):
    images = np.arange(50 * 2 * 2 * 3).reshape(50, 2, 2, 3)
    saved = []

    monkeypatch.setattr(shap.datasets, "imagenet50", lambda: (images, None))

    def fake_save_image(array, path_to_image):
        saved.append((np.array(array), path_to_image))

    monkeypatch.setattr(image_utils, "save_image", fake_save_image)

    image_utils.add_sample_images(str(tmp_path))

    assert [path for _, path in saved] == [
        os.path.join(str(tmp_path), "1.jpg"),
        os.path.join(str(tmp_path), "2.jpg"),
        os.path.join(str(tmp_path), "3.jpg"),
        os.path.join(str(tmp_path), "4.jpg"),
    ]
    np.testing.assert_array_equal(saved[0][0], images[25])
    np.testing.assert_array_equal(saved[1][0], images[26])
    np.testing.assert_array_equal(saved[2][0], images[30])
    np.testing.assert_array_equal(saved[3][0], images[44])


@pytest.mark.parametrize(
    ("path_to_image", "expected"),
    [
        ("example.png", True),
        ("example.jpg", True),
        ("example.jpeg", True),
        ("example.gif", True),
        ("example.bmp", True),
        ("example.jfif", True),
        ("example.txt", None),
    ],
)
def test_check_valid_image(path_to_image, expected):
    assert image_utils.check_valid_image(path_to_image) is expected


def test_save_image_and_load_image_round_trip(tmp_path):
    image = np.array(
        [
            [[255, 0, 0], [0, 255, 0]],
            [[0, 0, 255], [255, 255, 0]],
        ],
        dtype=np.uint8,
    )
    image_path = tmp_path / "roundtrip.png"

    image_utils.save_image(image, str(image_path))

    loaded = image_utils.load_image(str(image_path))

    assert loaded.shape == image.shape
    np.testing.assert_allclose(loaded, image.astype(float), atol=1)


def test_resize_image_returns_original_for_small_images(tmp_path):
    image = np.full((100, 80, 3), 128, dtype=np.uint8)
    image_path = tmp_path / "small.png"
    reshaped_dir = tmp_path / "reshaped"
    reshaped_dir.mkdir()
    image_utils.save_image(image, str(image_path))

    resized, reshaped_path = image_utils.resize_image(str(image_path), str(reshaped_dir))

    assert resized.shape == image.shape
    assert reshaped_path is None


@pytest.mark.parametrize(
    ("shape", "expected_shape"),
    [
        ((600, 600, 3), (500, 500, 3)),
        ((600, 300, 3), (500, 250, 3)),
        ((300, 600, 3), (250, 500, 3)),
    ],
)
def test_resize_image_resizes_large_images(tmp_path, shape, expected_shape):
    image = np.full(shape, 200, dtype=np.uint8)
    image_path = tmp_path / f"{shape[0]}x{shape[1]}.png"
    reshaped_dir = tmp_path / "reshaped"
    reshaped_dir.mkdir(exist_ok=True)
    image_utils.save_image(image, str(image_path))

    resized, reshaped_path = image_utils.resize_image(str(image_path), str(reshaped_dir))

    assert resized.shape == expected_shape
    assert reshaped_path == os.path.join(str(reshaped_dir), f"{image_path.stem}.png")
    assert os.path.exists(reshaped_path)


def test_display_grid_plot_creates_axes_with_titles(tmp_path):
    plt.close("all")
    image_paths = []
    for index in range(2):
        image = np.full((4, 4, 3), index * 50, dtype=np.uint8)
        image_path = tmp_path / f"image_{index}.png"
        image_utils.save_image(image, str(image_path))
        image_paths.append(str(image_path))

    image_utils.display_grid_plot(["first caption", "second caption"], image_paths, max_columns=4, figsize=(4, 4))

    fig = plt.gcf()
    assert len(fig.axes) == 2
    assert [axis.get_title() for axis in fig.axes] == ["first caption", "second caption"]
    plt.close("all")
