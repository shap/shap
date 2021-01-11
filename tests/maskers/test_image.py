from shap.utils import assert_import

try:
    assert_import("cv2")
except ImportError:
    pytestmark = pytest.mark.skip("opencv not installed")

def test_serialization_image_masker_inpaint_telea():
    import shap
    import numpy as np
    import tempfile

    test_image_height = 500
    test_image_width = 500
    test_data = np.ones((test_image_height,test_image_width,3)) * 50
    test_shape = (test_image_height, test_image_width, 3)
    # initialize image masker
    original_image_masker = shap.maskers.Image("inpaint_telea", test_shape)

    temp_serialization_file = tempfile.TemporaryFile()

    # serialize independent masker
    original_image_masker.save(temp_serialization_file)

    temp_serialization_file.seek(0)

    # deserialize masker
    new_image_masker = shap.maskers.Image.load(temp_serialization_file)

    temp_serialization_file.close()

    mask = np.ones((test_image_height, test_image_width, 3))
    mask = mask.astype(int)
    mask[0][0] = 0
    mask[4][0] = 0

    # comparing masked values
    assert np.array_equal(original_image_masker(mask, test_data), new_image_masker(mask, test_data))

def test_serialization_image_masker_inpaint_ns():
    import shap
    import numpy as np
    import tempfile

    test_image_height = 500
    test_image_width = 500
    test_data = np.ones((test_image_height,test_image_width,3)) * 50
    test_shape = (test_image_height, test_image_width, 3)
    # initialize image masker
    original_image_masker = shap.maskers.Image("inpaint_ns", test_shape)

    temp_serialization_file = tempfile.TemporaryFile()

    # serialize independent masker
    original_image_masker.save(temp_serialization_file)

    temp_serialization_file.seek(0)

    # deserialize masker
    new_image_masker = shap.maskers.Image.load(temp_serialization_file)

    temp_serialization_file.close()

    mask = np.ones((test_image_height, test_image_width, 3))
    mask = mask.astype(int)
    mask[0][0] = 0
    mask[4][0] = 0

    # comparing masked values
    assert np.array_equal(original_image_masker(mask, test_data), new_image_masker(mask, test_data))

def test_serialization_image_masker_blur():
    import shap
    import numpy as np
    import tempfile

    test_image_height = 500
    test_image_width = 500
    test_data = np.ones((test_image_height,test_image_width,3)) * 50
    test_shape = (test_image_height, test_image_width, 3)
    # initialize image masker
    original_image_masker = shap.maskers.Image("blur(10,10)", test_shape)

    temp_serialization_file = tempfile.TemporaryFile()

    # serialize independent masker
    original_image_masker.save(temp_serialization_file)

    temp_serialization_file.seek(0)

    # deserialize masker
    new_image_masker = shap.maskers.Image.load(temp_serialization_file)
    
    temp_serialization_file.close()

    mask = np.ones((test_image_height, test_image_width, 3))
    mask = mask.astype(int)
    mask[0][0] = 0
    mask[4][0] = 0

    # comparing masked values
    assert np.array_equal(original_image_masker(mask, test_data), new_image_masker(mask, test_data))

def test_serialization_image_masker_mask():
    import shap
    import numpy as np
    import tempfile

    test_image_height = 500
    test_image_width = 500
    test_data = np.ones((test_image_height,test_image_width,3)) * 50
    test_shape = (test_image_height, test_image_width, 3)
    test_mask = np.ones((test_image_height, test_image_width, 3))
    # initialize image masker
    original_image_masker = shap.maskers.Image(test_mask, test_shape)

    temp_serialization_file = tempfile.TemporaryFile()

    # serialize independent masker
    original_image_masker.save(temp_serialization_file)

    temp_serialization_file.seek(0)

    # deserialize masker
    new_image_masker = shap.maskers.Image.load(temp_serialization_file)

    temp_serialization_file.close()

    mask = np.ones((test_image_height, test_image_width, 3))
    mask = mask.astype(int)
    mask[0][0] = 0
    mask[4][0] = 0

    # comparing masked values
    assert np.array_equal(original_image_masker(mask, test_data), new_image_masker(mask, test_data))