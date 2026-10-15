import os

import numpy as np
import pytest

from shap.utils import image as image_utils

# Public functions in shap/utils/image.py:
# - is_empty(path): returns True when path does not exist or directory is empty; otherwise False.
# - make_dir(path): creates a directory if missing; empties files in an existing directory.
# - add_sample_images(path): writes four selected sample images from shap.datasets.imagenet50().
# - load_image(path_to_image): reads image via cv2, converts BGR->RGB, returns float numpy array.
# - check_valid_image(path_to_image): returns True for supported image extensions.
# - save_image(array, path_to_image): saves RGB image data normalized by 255.0 using matplotlib.
# - resize_image(path_to_image, reshaped_dir): conditionally resizes large images while preserving aspect ratio.
# - display_grid_plot(list_of_captions, list_of_images, ...): renders images/captions in a grid.


def test_is_empty_non_existent_path_returns_true_and_message(tmp_path, capsys):
    missing_dir = tmp_path / "does_not_exist"

    assert image_utils.is_empty(str(missing_dir)) is True
    out = capsys.readouterr().out
    assert "There is no 'test_images' folder" in out


def test_is_empty_existing_empty_directory_returns_true_and_message(tmp_path, capsys):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()

    assert image_utils.is_empty(str(empty_dir)) is True
    out = capsys.readouterr().out
    assert "'test_images' folder is empty" in out


def test_is_empty_existing_non_empty_directory_returns_false(tmp_path, capsys):
    non_empty = tmp_path / "non_empty"
    non_empty.mkdir()
    (non_empty / "x.txt").write_text("data")

    assert image_utils.is_empty(str(non_empty)) is False
    assert capsys.readouterr().out == ""


def test_make_dir_creates_missing_directory(tmp_path):
    target = tmp_path / "new_dir"

    image_utils.make_dir(str(target))

    assert target.exists() and target.is_dir()


def test_make_dir_empties_existing_directory(tmp_path):
    target = tmp_path / "existing"
    target.mkdir()
    (target / "a.txt").write_text("a")
    (target / "b.txt").write_text("b")

    image_utils.make_dir(str(target) + os.sep)

    assert list(target.iterdir()) == []


def test_make_dir_given_file_path_raises_not_a_directory_error(tmp_path):
    file_path = tmp_path / "not_a_dir"
    file_path.write_text("content")

    with pytest.raises(NotADirectoryError):
        image_utils.make_dir(str(file_path))


def test_add_sample_images_saves_expected_imagenet_indices(monkeypatch, tmp_path):
    baseline_image = image_utils.load_image("tests/plots/baseline/test_bar.png").astype(np.uint8)

    images = np.zeros((50, *baseline_image.shape), dtype=np.uint8)
    images[25] = np.full_like(baseline_image, 10)
    images[26] = np.full_like(baseline_image, 60)
    images[30] = np.full_like(baseline_image, 120)
    images[44] = np.full_like(baseline_image, 200)
    labels = np.arange(50)

    monkeypatch.setattr(image_utils.shap.datasets, "imagenet50", lambda: (images, labels))

    image_utils.add_sample_images(str(tmp_path))

    written = [tmp_path / f"{idx}.jpg" for idx in range(1, 5)]
    assert [path.name for path in written] == ["1.jpg", "2.jpg", "3.jpg", "4.jpg"]
    assert all(path.exists() for path in written)

    means = [image_utils.load_image(str(path)).mean() for path in written]
    assert means[0] < means[1] < means[2] < means[3]


def test_load_image_returns_float_array_with_real_image():
    out = image_utils.load_image("tests/plots/baseline/test_bar.png")

    assert out.dtype == float
    assert out.ndim == 3
    assert out.shape[2] == 3
    assert 0 <= out.min() <= out.max() <= 255


@pytest.mark.parametrize(
    "path,expected",
    [
        ("x.png", True),
        ("x.jpg", True),
        ("x.jpeg", True),
        ("x.gif", True),
        ("x.bmp", True),
        ("x.jfif", True),
        ("x.txt", None),
        ("x.JPG", None),
    ],
)
def test_check_valid_image_extensions(path, expected):
    assert image_utils.check_valid_image(path) is expected


def test_save_image_creates_valid_file(tmp_path):
    arr: np.ndarray = np.full((10, 10, 3), 128.0, dtype=float)
    out_path = tmp_path / "img.png"

    image_utils.save_image(arr, str(out_path))

    assert out_path.exists()
    loaded = image_utils.load_image(str(out_path))
    assert loaded.shape == arr.shape


def test_resize_image_large_image_gets_resized(tmp_path):
    large: np.ndarray = np.full((250, 1000, 3), 180.0, dtype=float)
    input_path = tmp_path / "large.png"
    image_utils.save_image(large, str(input_path))

    output_dir = tmp_path / "resized"
    output_dir.mkdir()

    out_image, out_path = image_utils.resize_image(str(input_path), str(output_dir))

    assert out_path is not None
    assert os.path.exists(out_path)
    assert out_image.shape == (125, 500, 3)
    assert out_image.dtype == float


def test_resize_image_small_image_unchanged(tmp_path):
    small: np.ndarray = np.full((100, 150, 3), 90.0, dtype=float)
    input_path = tmp_path / "small.png"
    image_utils.save_image(small, str(input_path))

    output_dir = tmp_path / "resized"
    output_dir.mkdir()

    out_image, out_path = image_utils.resize_image(str(input_path), str(output_dir))

    assert out_path is None
    assert out_image.shape == (100, 150, 3)


def test_load_image_propagates_error_for_missing_image(monkeypatch):
    monkeypatch.setattr(image_utils.cv2, "imread", lambda _: None)

    def fake_cvtcolor(_img, _code):
        raise ValueError("invalid image")

    monkeypatch.setattr(image_utils.cv2, "cvtColor", fake_cvtcolor)

    with pytest.raises(ValueError, match="invalid image"):
        image_utils.load_image("missing.jpg")


def test_save_image_propagates_imsave_errors(monkeypatch):
    def fake_imsave(_path, _arr):
        raise PermissionError("no permission")

    monkeypatch.setattr(image_utils.plt, "imsave", fake_imsave)

    with pytest.raises(PermissionError, match="no permission"):
        image_utils.save_image(np.zeros((1, 1, 3), dtype=float), "/root/forbidden.png")


def test_resize_image_invalid_loaded_data_raises(monkeypatch, tmp_path):
    monkeypatch.setattr(image_utils, "load_image", lambda _: None)

    with pytest.raises(AttributeError):
        image_utils.resize_image("/img/bad.jpg", str(tmp_path))


def test_display_grid_plot_creates_new_figure_when_columns_exceeded_and_sets_titles(
    monkeypatch,
):
    images: dict[str, np.ndarray] = {
        "a": np.ones((1, 1, 3), dtype=np.uint8),
        "b": np.ones((1, 1, 3), dtype=np.uint8) * 2,
        "c": np.ones((1, 1, 3), dtype=np.uint8) * 3,
    }
    monkeypatch.setattr(image_utils, "load_image", lambda name: images[name])

    figure_calls = []
    subplot_calls = []
    imshow_calls = []
    axis_calls = []
    titles = []

    class DummyFigure:
        def add_subplot(self, *args):
            subplot_calls.append(args)

    def fake_figure(*, figsize):
        figure_calls.append(figsize)
        return DummyFigure()

    monkeypatch.setattr(image_utils.plt, "figure", fake_figure)
    monkeypatch.setattr(image_utils.plt, "imshow", lambda img: imshow_calls.append(img.copy()))
    monkeypatch.setattr(image_utils.plt, "axis", lambda arg: axis_calls.append(arg))
    monkeypatch.setattr(image_utils.plt, "title", lambda txt: titles.append(txt))

    image_utils.display_grid_plot(
        list_of_captions=["first caption", "second caption", "third caption"],
        list_of_images=["a", "b", "c"],
        max_columns=2,
        figsize=(10, 10),
    )

    assert figure_calls == [(10, 10), (10, 10)]
    assert subplot_calls == [(1, 2, 1), (1, 2, 2), (1, 2, 1)]
    assert len(imshow_calls) == 3
    assert axis_calls == ["off", "off", "off"]
    assert titles == ["first caption", "second caption", "third caption"]


def test_display_grid_plot_skips_titles_when_insufficient_captions(monkeypatch):
    monkeypatch.setattr(image_utils, "load_image", lambda _name: np.zeros((1, 1, 3), dtype=np.uint8))

    class DummyFigure:
        def add_subplot(self, *_):
            return None

    monkeypatch.setattr(image_utils.plt, "figure", lambda **_: DummyFigure())
    monkeypatch.setattr(image_utils.plt, "imshow", lambda *_: None)
    monkeypatch.setattr(image_utils.plt, "axis", lambda *_: None)

    title_calls = []
    monkeypatch.setattr(image_utils.plt, "title", lambda txt: title_calls.append(txt))

    image_utils.display_grid_plot(
        list_of_captions=["only one"],
        list_of_images=["i1", "i2"],
        max_columns=4,
    )

    assert title_calls == []
