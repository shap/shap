import os
import sys
import types

import numpy as np
import pytest

if "cv2" not in sys.modules:
    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.COLOR_BGR2RGB = 1
    fake_cv2.imread = lambda _path: np.zeros((1, 1, 3), dtype=np.uint8)
    fake_cv2.cvtColor = lambda image, _code: image
    fake_cv2.resize = lambda image, dsize: np.zeros((dsize[1], dsize[0], image.shape[2]), dtype=image.dtype)
    sys.modules["cv2"] = fake_cv2

import shap.utils.image as image_utils


def test_is_empty_for_missing_empty_and_nonempty_paths(tmp_path, capsys):
    missing_path = tmp_path / "missing"
    assert image_utils.is_empty(str(missing_path)) is True
    assert "There is no 'test_images' folder" in capsys.readouterr().out

    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert image_utils.is_empty(str(empty_dir)) is True
    assert "'test_images' folder is empty" in capsys.readouterr().out

    non_empty_dir = tmp_path / "non_empty"
    non_empty_dir.mkdir()
    (non_empty_dir / "sample.txt").write_text("x", encoding="utf-8")
    assert image_utils.is_empty(str(non_empty_dir)) is False


def test_make_dir_creates_and_cleans_directory(tmp_path):
    created_path = tmp_path / "created"
    image_utils.make_dir(str(created_path))
    assert created_path.exists()

    existing_path = tmp_path / "existing"
    existing_path.mkdir()
    (existing_path / "a.txt").write_text("a", encoding="utf-8")
    (existing_path / "b.txt").write_text("b", encoding="utf-8")

    image_utils.make_dir(str(existing_path) + os.sep)

    assert list(existing_path.iterdir()) == []


def test_make_dir_invalid_folder_path_branch(monkeypatch, capsys):
    monkeypatch.setattr(image_utils.os.path, "exists", lambda _: False)
    monkeypatch.setattr(image_utils.os.path, "isfile", lambda _: True)

    image_utils.make_dir("invalid")

    assert "Please give a valid folder path." in capsys.readouterr().out


def test_add_sample_images_saves_expected_subset(monkeypatch, tmp_path):
    images = [np.full((1, 1, 3), i, dtype=np.uint8) for i in range(50)]
    monkeypatch.setattr(image_utils.shap.datasets, "imagenet50", lambda: (images, None))

    saved = []

    def fake_save_image(array, path_to_image):
        saved.append((int(array[0, 0, 0]), path_to_image))

    monkeypatch.setattr(image_utils, "save_image", fake_save_image)

    image_utils.add_sample_images(str(tmp_path))

    assert [x[0] for x in saved] == [25, 26, 30, 44]
    assert [os.path.basename(x[1]) for x in saved] == ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]


def test_load_image_converts_bgr_to_rgb_float(monkeypatch):
    bgr_image = np.array([[[5, 10, 20]]], dtype=np.uint8)
    monkeypatch.setattr(image_utils.cv2, "imread", lambda _: bgr_image)
    monkeypatch.setattr(image_utils.cv2, "cvtColor", lambda img, _: img[:, :, ::-1])

    loaded = image_utils.load_image("dummy.jpg")

    assert loaded.dtype == float
    np.testing.assert_allclose(loaded, np.array([[[20.0, 10.0, 5.0]]]))


def test_check_valid_image_and_save_image(monkeypatch):
    assert image_utils.check_valid_image("sample.jpg") is True
    assert image_utils.check_valid_image("sample.txt") is None

    captured = {}

    def fake_imsave(path_to_image, image):
        captured["path"] = path_to_image
        captured["image"] = image

    monkeypatch.setattr(image_utils.plt, "imsave", fake_imsave)

    array = np.array([[[255, 128, 0]]], dtype=np.float64)
    image_utils.save_image(array, "out.png")

    assert captured["path"] == "out.png"
    np.testing.assert_allclose(captured["image"], array / 255.0)


@pytest.mark.parametrize(
    "shape, expected_dim",
    [
        ((600, 600, 3), (500, 500)),
        ((800, 400, 3), (500, 250)),
        ((400, 800, 3), (250, 500)),
    ],
)
def test_resize_image_reshapes_large_inputs(monkeypatch, shape, expected_dim, capsys):
    image = np.zeros(shape, dtype=np.uint8)
    monkeypatch.setattr(image_utils, "load_image", lambda _: image)

    resize_calls = []

    def fake_resize(img, dsize):
        resize_calls.append(dsize)
        return np.zeros((dsize[1], dsize[0], 3), dtype=np.uint8)

    monkeypatch.setattr(image_utils.cv2, "resize", fake_resize)

    saved = {}

    def fake_save_image(array, path_to_image):
        saved["path"] = path_to_image
        saved["shape"] = array.shape

    monkeypatch.setattr(image_utils, "save_image", fake_save_image)

    resized, reshaped_path = image_utils.resize_image("folder/sample.jpg", "reshaped")

    assert resize_calls == [(expected_dim[1], expected_dim[0])]
    assert reshaped_path == os.path.join("reshaped", "sample.png")
    assert saved["path"] == reshaped_path
    assert saved["shape"] == resized.shape
    assert resized.dtype == float
    assert "Reshaped image size" in capsys.readouterr().out


def test_resize_image_returns_original_when_small(monkeypatch):
    image = np.zeros((300, 400, 3), dtype=np.uint8)
    monkeypatch.setattr(image_utils, "load_image", lambda _: image)
    monkeypatch.setattr(image_utils.cv2, "resize", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError()))
    monkeypatch.setattr(image_utils, "save_image", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError()))

    resized, reshaped_path = image_utils.resize_image("folder/sample.jpg", "reshaped")

    assert reshaped_path is None
    assert resized is image


def test_display_grid_plot_handles_columns_and_titles(monkeypatch):
    monkeypatch.setattr(image_utils, "load_image", lambda _: np.ones((2, 2, 3), dtype=np.uint8) * 10)

    class FakeFigure:
        def __init__(self):
            self.subplots = []

        def add_subplot(self, *args):
            self.subplots.append(args)

    figures = []
    imshow_calls = []
    axis_calls = []
    title_calls = []

    def fake_figure(figsize):
        fig = FakeFigure()
        figures.append((figsize, fig))
        return fig

    monkeypatch.setattr(image_utils.plt, "figure", fake_figure)
    monkeypatch.setattr(image_utils.plt, "imshow", lambda img: imshow_calls.append(img.shape))
    monkeypatch.setattr(image_utils.plt, "axis", lambda arg: axis_calls.append(arg))
    monkeypatch.setattr(image_utils.plt, "title", lambda title: title_calls.append(title))

    image_utils.display_grid_plot(["cap1", "cap2", "cap3"], ["a.png", "b.png", "c.png"], max_columns=2)

    assert len(figures) == 2
    assert len(imshow_calls) == 3
    assert axis_calls == ["off", "off", "off"]
    assert len(title_calls) == 3

    title_calls.clear()
    image_utils.display_grid_plot(["cap1"], ["a.png", "b.png"], max_columns=2)
    assert title_calls == []
