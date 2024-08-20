from io import BytesIO
from typing import Any

import numpy as np

from shap.explainers.other._ubjson import _decode_simple_key_value_pair


def test_decode_simple_key_value_pair():
    # todo: this is not correct, fix this
    num_class = b"L\x00\x00\x00\x00\x00\x00\x00\tnum_classL\x00\x00\x00\x00\x00\x00\x00\x0b"
    fp = BytesIO(num_class)
    key_type = fp.read(1)
    key, value = _decode_simple_key_value_pair(fp, key_type=key_type)
    assert key == "num_class" and value == 11

    boost_from_average = b"L\x00\x00\x00\x00\x00\x00\x00\x12boost_from_averageSL\x00\x00\x00\x00\x00\x00\x00\x011"

    fp = BytesIO(boost_from_average)
    key_type = fp.read(1)
    key, value = _decode_simple_key_value_pair(fp, key_type)
    assert key == "boost_from_average" and value == "1"

    num_feature = b"L\x00\x00\x00\x00\x00\x00\x00\x0bnum_featureSL\x00\x00\x00\x00\x00\x00\x00\x013"
    fp = BytesIO(num_feature)
    key_type = fp.read(1)
    key, value = _decode_simple_key_value_pair(fp, key_type)
    assert key == "num_feature" and value == "3"

    num_class = b"L\x00\x00\x00\x00\x00\x00\x00\tnum_classSL\x00\x00\x00\x00\x00\x00\x00\x010"
    fp = BytesIO(num_class)
    key_type = fp.read(1)
    key, value = _decode_simple_key_value_pair(fp, key_type)
    assert key == "num_class" and value == "0"


def test_decode_object():
    expected_value: dict[str, Any]
    regression_loss = b"L\x00\x00\x00\x00\x00\x00\x00\x0ereg_loss_param{L\x00\x00\x00\x00\x00\x00\x00\x10scale_pos_weightSL\x00\x00\x00\x00\x00\x00\x00\x011}"
    fp = BytesIO(regression_loss)
    key_type = fp.read(1)
    key, value = _decode_simple_key_value_pair(fp, key_type)
    expected_key = "reg_loss_param"
    expected_value = {"scale_pos_weight": "1"}
    assert expected_key == key and value == expected_value

    objective_dict = b"L\x00\x00\x00\x00\x00\x00\x00\tobjective{L\x00\x00\x00\x00\x00\x00\x00\x04nameSL\x00\x00\x00\x00\x00\x00\x00\x0fbinary:logisticL\x00\x00\x00\x00\x00\x00\x00\x0ereg_loss_param{L\x00\x00\x00\x00\x00\x00\x00\x10scale_pos_weightSL\x00\x00\x00\x00\x00\x00\x00\x011}}"
    fp = BytesIO(objective_dict)
    key_type = fp.read(1)
    key, value = _decode_simple_key_value_pair(fp, key_type)
    expected_key = "objective"
    expected_value = {"name": "binary:logistic", "reg_loss_param": {"scale_pos_weight": "1"}}
    assert expected_key == key and value == expected_value

    objective_reversed_dict = b"L\x00\x00\x00\x00\x00\x00\x00\tobjective{L\x00\x00\x00\x00\x00\x00\x00\x0ereg_loss_param{L\x00\x00\x00\x00\x00\x00\x00\x10scale_pos_weightSL\x00\x00\x00\x00\x00\x00\x00\x011}L\x00\x00\x00\x00\x00\x00\x00\x04nameSL\x00\x00\x00\x00\x00\x00\x00\x0fbinary:logisticL\x00\x00\x00\x00\x00\x00\x00\x0e}"
    fp = BytesIO(objective_reversed_dict)
    key_type = fp.read(1)
    key, value = _decode_simple_key_value_pair(fp, key_type)
    expected_key = "objective"
    expected_value = {
        "reg_loss_param": {"scale_pos_weight": "1"},
        "name": "binary:logistic",
    }
    assert expected_key == key and value == expected_value

    empty_object_dict = b"L\x00\x00\x00\x00\x00\x00\x00\nattributes{}"
    fp = BytesIO(empty_object_dict)
    key_type = fp.read(1)
    key, value = _decode_simple_key_value_pair(fp, key_type)
    assert key == "attributes" and value == {}


def test_decode_array():
    left_children = b"L\x00\x00\x00\x00\x00\x00\x00\rleft_children[$l#L\x00\x00\x00\x00\x00\x00\x00\x01\xff\xff\xff\xffL\x00\x00\x00\x00\x00\x00\x00\x0c"
    fp = BytesIO(left_children)
    key_type = fp.read(1)
    key, value = _decode_simple_key_value_pair(fp, key_type)
    assert key == "left_children" and value == np.array([-1], dtype=np.int32)

    # string array
    feature_names = b"L\x00\x00\x00\x00\x00\x00\x00\rfeature_names[#L\x00\x00\x00\x00\x00\x00\x00\x00L\x00\x00\x00\x00\x00\x00\x00\r"
    fp = BytesIO(feature_names)
    key_type = fp.read(1)
    key, value = _decode_simple_key_value_pair(fp, key_type)
    assert key == "feature_names" and value == []

    base_weights = b"L\x00\x00\x00\x00\x00\x00\x00\x0cbase_weights[$d#L\x00\x00\x00\x00\x00\x00\x00\x01\xba\xa2\xe1&"
    fp = BytesIO(base_weights)
    key_type = fp.read(1)
    key, value = _decode_simple_key_value_pair(fp, key_type)
    assert key == "base_weights" and value == [-0.0012426718603819609]
