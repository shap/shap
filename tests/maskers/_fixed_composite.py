''' This file contains tests for the FixedComposite masker.
'''

def test_fixed_composite_masker_call():
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForCausalLM
    import shap
    from shap import maskers

    args=("This is a test statement for fixed composite masker",)

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    masker = maskers.Text(tokenizer)
    masker._update_s_cache(*args)
    mask = np.zeros(masker._segments_s.shape, dtype=bool)

    fixed_composite_masker = maskers.FixedComposite(masker)

    expected_fixed_composite_masked_output = [(np.array(['']), np.array(["This is a test statement for fixed composite masker"]))]
    fixed_composite_masked_output = fixed_composite_masker(mask, *args)

    assert fixed_composite_masked_output == expected_fixed_composite_masked_output