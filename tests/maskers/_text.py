def test_serialization_text_masker():
    import shap
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased",use_fast=False)
    original_masker = shap.maskers.Text(tokenizer)

    out_file = open(r'test_serialization_text_masker.bin', "wb")
    original_masker.save(out_file)
    out_file.close()


    # deserialize masker
    in_file = open(r'test_serialization_text_masker.bin', "rb")
    new_masker = shap.maskers.Text.load(in_file)
    in_file.close()


    test_text = "I ate a Cannoli"
    test_input_mask = np.array([True, False, True, True, False, True, True, True])

    original_masked_output = original_masker(test_input_mask,test_text)
    new_masked_output = new_masker(test_input_mask,test_text)

    assert original_masked_output == new_masked_output

def test_serialization_text_masker_custom_mask():
    import shap
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased",use_fast=True)
    original_masker = shap.maskers.Text(tokenizer, mask_token = '[CUSTOM-MASK]')

    out_file = open(r'test_serialization_text_masker.bin', "wb")
    original_masker.save(out_file)
    out_file.close()


    # deserialize masker
    in_file = open(r'test_serialization_text_masker.bin', "rb")
    new_masker = shap.maskers.Text.load(in_file)
    in_file.close()


    test_text = "I ate a Cannoli"
    test_input_mask = np.array([True, False, True, True, False, True, True, True])

    original_masked_output = original_masker(test_input_mask, test_text)
    new_masked_output = new_masker(test_input_mask, test_text)

    assert original_masked_output == new_masked_output

def test_serialization_text_masker_collapse_mask_token():
    import shap
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased",use_fast=True)
    original_masker = shap.maskers.Text(tokenizer, collapse_mask_token=True)

    out_file = open(r'test_serialization_text_masker.bin', "wb")
    original_masker.save(out_file)
    out_file.close()


    # deserialize masker
    in_file = open(r'test_serialization_text_masker.bin', "rb")
    new_masker = shap.maskers.Text.load(in_file)
    in_file.close()


    test_text = "I ate a Cannoli"
    test_input_mask = np.array([True, False, True, True, False, True, True, True])

    original_masked_output = original_masker(test_input_mask, test_text)
    new_masked_output = new_masker(test_input_mask, test_text)

    assert original_masked_output == new_masked_output