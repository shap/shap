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

    expected_fixed_composite_masked_output = (np.array(['']), np.array(["This is a test statement for fixed composite masker"]))
    fixed_composite_masked_output = fixed_composite_masker(mask, *args)

    assert fixed_composite_masked_output == expected_fixed_composite_masked_output

def test_serialization_fixedcomposite_masker():
    import shap
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased",use_fast=False)
    underlying_masker = shap.maskers.Text(tokenizer)
    original_masker = shap.maskers.FixedComposite(underlying_masker)

    out_file = open(r'test_serialization_fixedcomposite_masker.bin', "wb")
    original_masker.save(out_file)
    out_file.close()


    # deserialize masker
    in_file = open(r'test_serialization_fixedcomposite_masker.bin', "rb")
    new_masker = shap.maskers.FixedComposite.load(in_file)
    in_file.close()


    test_text = "I ate a Cannoli"
    test_input_mask = np.array([True, False, True, True, False, True, True, True])

    original_masked_output = original_masker(test_input_mask,test_text)
    new_masked_output = new_masker(test_input_mask,test_text)

    assert original_masked_output == new_masked_output