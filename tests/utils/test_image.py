from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("cv2")

from shap.utils.image import load_image, resize_image, save_image


@pytest.mark.parametrize("filename", ["sample.png", "sample.before.png", ".sample.png"])
def test_resize_image_preserves_filename_stem(tmp_path, filename):
    source = tmp_path / filename
    output_dir = tmp_path / "resized"
    output_dir.mkdir()
    save_image(np.zeros((600, 800, 3)), str(source))

    image, output_path = resize_image(str(source), str(output_dir))

    assert output_path == str(output_dir / f"{Path(filename).stem}.png")
    assert image.shape == (375, 500, 3)
    np.testing.assert_array_equal(load_image(output_path), image)


def test_resize_image_keeps_distinct_dotted_filenames(tmp_path):
    output_dir = tmp_path / "resized"
    output_dir.mkdir()
    output_paths = []
    for filename, value in [("sample.before.png", 0), ("sample.after.png", 255)]:
        source = tmp_path / filename
        save_image(np.full((600, 800, 3), value), str(source))
        _, output_path = resize_image(str(source), str(output_dir))
        output_paths.append(output_path)

    assert output_paths[0] != output_paths[1]
    assert len(list(output_dir.iterdir())) == 2
    assert np.all(load_image(output_paths[0]) == 0)
    assert np.all(load_image(output_paths[1]) == 255)
