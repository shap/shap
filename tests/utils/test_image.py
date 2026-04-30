from shap.utils.image import check_valid_image


def test_valid_image():
    assert check_valid_image("test.jpg") == True


def test_invalid_image():
    assert check_valid_image("test.txt") is None
