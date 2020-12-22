def test_serialization_image_masker_inpaint_telea():
    import shap
    import numpy as np

    test_image_height = 500
    test_image_width = 500
    test_data = np.ones((test_image_height,test_image_width,3)) * 50
    test_shape = (test_image_height, test_image_width, 3)
    # initialize image masker
    original_image_masker = shap.maskers.Image("inpaint_telea", test_shape)

    # serialize independent masker
    out_file = open(r'test_serialization_image_masker.bin', "wb")
    original_image_masker.save(out_file)
    out_file.close()


    # deserialize masker
    in_file = open(r'test_serialization_image_masker.bin', "rb")
    new_image_masker = shap.maskers.Image.load(in_file)
    in_file.close()

    mask = np.ones((test_image_height, test_image_width, 3))
    mask = mask.astype(int)
    mask[0][0] = 0
    mask[4][0] = 0

    # comparing masked values
    assert np.array_equal(original_image_masker(mask, test_data), new_image_masker(mask, test_data))

def test_serialization_image_masker_inpaint_ns():
    import shap
    import numpy as np

    test_image_height = 500
    test_image_width = 500
    test_data = np.ones((test_image_height,test_image_width,3)) * 50
    test_shape = (test_image_height, test_image_width, 3)
    # initialize image masker
    original_image_masker = shap.maskers.Image("inpaint_ns", test_shape)

    # serialize independent masker
    out_file = open(r'test_serialization_image_masker.bin', "wb")
    original_image_masker.save(out_file)
    out_file.close()


    # deserialize masker
    in_file = open(r'test_serialization_image_masker.bin', "rb")
    new_image_masker = shap.maskers.Image.load(in_file)
    in_file.close()

    mask = np.ones((test_image_height, test_image_width, 3))
    mask = mask.astype(int)
    mask[0][0] = 0
    mask[4][0] = 0

    # comparing masked values
    assert np.array_equal(original_image_masker(mask, test_data), new_image_masker(mask, test_data))

def test_serialization_image_masker_blur():
    import shap
    import numpy as np

    test_image_height = 500
    test_image_width = 500
    test_data = np.ones((test_image_height,test_image_width,3)) * 50
    test_shape = (test_image_height, test_image_width, 3)
    # initialize image masker
    original_image_masker = shap.maskers.Image("blur(10,10)", test_shape)

    # serialize independent masker
    out_file = open(r'test_serialization_image_masker.bin', "wb")
    original_image_masker.save(out_file)
    out_file.close()


    # deserialize masker
    in_file = open(r'test_serialization_image_masker.bin', "rb")
    new_image_masker = shap.maskers.Image.load(in_file)
    in_file.close()

    mask = np.ones((test_image_height, test_image_width, 3))
    mask = mask.astype(int)
    mask[0][0] = 0
    mask[4][0] = 0

    # comparing masked values
    assert np.array_equal(original_image_masker(mask, test_data), new_image_masker(mask, test_data))

def test_serialization_image_masker_mask():
    import shap
    import numpy as np

    test_image_height = 500
    test_image_width = 500
    test_data = np.ones((test_image_height,test_image_width,3)) * 50
    test_shape = (test_image_height, test_image_width, 3)
    test_mask = np.ones((test_image_height, test_image_width, 3))
    # initialize image masker
    original_image_masker = shap.maskers.Image(test_mask, test_shape)

    # serialize independent masker
    out_file = open(r'test_serialization_image_masker.bin', "wb")
    original_image_masker.save(out_file)
    out_file.close()


    # deserialize masker
    in_file = open(r'test_serialization_image_masker.bin', "rb")
    new_image_masker = shap.maskers.Image.load(in_file)
    in_file.close()

    mask = np.ones((test_image_height, test_image_width, 3))
    mask = mask.astype(int)
    mask[0][0] = 0
    mask[4][0] = 0

    # comparing masked values
    assert np.array_equal(original_image_masker(mask, test_data), new_image_masker(mask, test_data))