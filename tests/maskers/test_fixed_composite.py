""" This file contains tests for the FixedComposite masker.
"""

import pytest
import numpy as np
import shap


def test_fixed_composite_masker_call():
    """ Test to make sure the FixedComposite masker works when masking everything.
    """

    AutoTokenizer = pytest.importorskip("transformers").AutoTokenizer

    args = ("This is a test statement for fixed composite masker",)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    masker = shap.maskers.Text(tokenizer)
    mask = np.zeros(masker.shape(*args)[1], dtype=bool)

    fixed_composite_masker = shap.maskers.FixedComposite(masker)

    expected_fixed_composite_masked_output = (np.array(['']), np.array(["This is a test statement for fixed composite masker"]))
    fixed_composite_masked_output = fixed_composite_masker(mask, *args)

    assert fixed_composite_masked_output == expected_fixed_composite_masked_output
