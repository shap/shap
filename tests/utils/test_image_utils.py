import os
import sys
import types

import numpy as np
import pytest


def _make_cv2_stub():
    """Minimal cv2 stand-in when opencv-python is missing or unloadable."""
    from PIL import Image

    fake_cv2 = types.ModuleType("cv2")
    fake_cv2.COLOR_BGR2RGB = 1

    def imread(path):
        rgb = np.array(Image.open(path).convert("RGB"))
        return rgb[..., ::-1]  # BGR to match cv2.imread convention

    def cvtColor(image, code):
        assert code == fake_cv2.COLOR_BGR2RGB
        return image[..., ::-1]  # BGR -> RGB

    def resize(image, dsize):
        # cv2 expects dsize as (width, height)
        width, height = dsize
        pil = Image.fromarray(image.astype(np.uint8))
        resized = pil.resize((width, height), resample=Image.BILINEAR)
        return np.array(resized)

    fake_cv2.imread = imread
    fake_cv2.cvtColor = cvtColor
    fake_cv2.resize = resize
    return fake_cv2


def _import_image_utils():
    """Import `shap.utils.image` without leaving a fake `cv2` in `sys.modules`.

    The library imports `cv2` at module level. Many CI environments omit OpenCV;
    we stub it only for that import so other tests still see a failed `import cv2`
    when OpenCV is absent. `ImportError` covers `ModuleNotFoundError` and broken
    native-library installs that raise `ImportError`.
    """
    used_stub = False
    try:
        import cv2  # noqa: F401
    except ImportError:
        sys.modules["cv2"] = _make_cv2_stub()
        used_stub = True
    try:
        import shap.utils.image as mod  # noqa: E402

        return mod
    finally:
        if used_stub:
            sys.modules.pop("cv2", None)


image_utils = _import_image_utils()


def test_is_empty_missing_path(capsys, tmp_path):
    missing_dir = tmp_path / "does_not_exist"
    assert image_utils.is_empty(str(missing_dir)) is True
    captured = capsys.readouterr()
    assert "There is no 'test_images' folder" in captured.out


def test_is_empty_empty_dir(capsys, tmp_path):
    empty_dir = tmp_path / "empty"
    empty_dir.mkdir()
    assert image_utils.is_empty(str(empty_dir)) is True
    captured = capsys.readouterr()
    assert "'test_images' folder is empty" in captured.out


def test_is_empty_non_empty_dir(tmp_path):
    non_empty_dir = tmp_path / "non_empty"
    non_empty_dir.mkdir()
    (non_empty_dir / "x.txt").write_text("x")
    assert image_utils.is_empty(str(non_empty_dir)) is False


def test_make_dir_creates_and_clears(tmp_path):
    # create
    created = tmp_path / "created"
    image_utils.make_dir(str(created))
    assert created.is_dir()

    # clear
    (created / "a.txt").write_text("a")
    (created / "b.txt").write_text("b")
    image_utils.make_dir(str(created))
    assert list(created.iterdir()) == []


@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("a.png", True),
        ("a.jpg", True),
        ("a.jpeg", True),
        ("a.gif", True),
        ("a.bmp", True),
        ("a.jfif", True),
        ("a.tiff", None),
        ("a.txt", None),
    ],
)
def test_check_valid_image_extensions(path, expected):
    assert image_utils.check_valid_image(path) is expected


def test_save_load_and_resize_noreshape(tmp_path):
    # Create a small image that should not be reshaped
    array = np.zeros((32, 16, 3), dtype=np.uint8)
    array[..., 0] = 255  # red
    path = tmp_path / "small.png"
    image_utils.save_image(array, str(path))

    loaded = image_utils.load_image(str(path))
    assert loaded.shape == (32, 16, 3)

    reshaped_dir = tmp_path / "reshaped"
    reshaped_dir.mkdir()
    resized, reshaped_path = image_utils.resize_image(str(path), str(reshaped_dir))
    assert resized.shape == (32, 16, 3)
    assert reshaped_path is None


def test_resize_image_reshapes_large_height(tmp_path, capsys):
    # Create a large image (height > 500) that triggers resize.
    large = np.zeros((600, 300, 3), dtype=np.uint8)
    large[..., 1] = 200  # green-ish
    path = tmp_path / "large.png"
    image_utils.save_image(large, str(path))

    reshaped_dir = tmp_path / "reshaped"
    reshaped_dir.mkdir()
    resized, reshaped_path = image_utils.resize_image(str(path), str(reshaped_dir))

    assert resized.shape[0] == 500
    assert resized.shape[1] == int(300 * 500 / 600)
    assert reshaped_path is not None
    assert os.path.exists(reshaped_path)

    captured = capsys.readouterr()
    assert "Reshaped image size:" in captured.out
