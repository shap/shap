''' This file contains tests for the Text masker.
'''



def test_method_tokenize_pretrained_tokenizer():
    import shap
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased",use_fast=False)
    masker = shap.maskers.Text(tokenizer)

    test_text = "I have a joke about deep learning but I can't explain it."
    output_ids = masker.tokenize(test_text)['input_ids']
    correct_ids = tokenizer.encode_plus(test_text)['input_ids']

    print("Test logging",output_ids)
    print(correct_ids)
    assert output_ids == correct_ids

def test_method_tokenize_pretrained_tokenizer_fast():
    import shap
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased",use_fast=True)
    masker = shap.maskers.Text(tokenizer)

    test_text = "I have a joke about deep learning but I can't explain it."
    output_ids = masker.tokenize(test_text)['input_ids']
    correct_ids = tokenizer.encode_plus(test_text)['input_ids']

    assert output_ids == correct_ids


def test_method_token_segments_pretrained_tokenizer():
    import shap
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased",use_fast=False)
    masker = shap.maskers.Text(tokenizer)

    test_text = "I ate a Cannoli"
    output_token_segments = masker.token_segments(test_text)
    correct_token_segments = ['','I','ate','a','Can','##no','##li','']
    print(output_token_segments)

    assert output_token_segments == correct_token_segments


def test_method_token_segments_pretrained_tokenizer_fast():
    import shap
    from transformers import AutoTokenizer, AutoModelForSequenceClassification

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased",use_fast=True)
    masker = shap.maskers.Text(tokenizer)

    test_text = "I ate a Cannoli"
    output_token_segments = masker.token_segments(test_text)
    correct_token_segments = ['','I ','ate ','a ','Can','no','li','']

    assert output_token_segments == correct_token_segments


def test_masker_call_pretrained_tokenizer():
    import shap
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import numpy as np

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased",use_fast=False)
    masker = shap.maskers.Text(tokenizer)

    test_text = "I ate a Cannoli"
    test_input_mask = np.array([True, False, True, True, False, True, True, True])

    output_masked_text = masker(test_input_mask,test_text)
    correct_masked_text = '[MASK] ate a [MASK]noli'

    assert output_masked_text == correct_masked_text

def test_masker_call_pretrained_tokenizer_fast():
    import shap
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import numpy as np

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased",use_fast=True)
    masker = shap.maskers.Text(tokenizer)

    test_text = "I ate a Cannoli"
    test_input_mask = np.array([True, False, True, True, False, True, True, True])

    output_masked_text = masker(test_input_mask,test_text)
    correct_masked_text = '[MASK]ate a [MASK]noli'
    
    assert output_masked_text == correct_masked_text

def test_text_infiling():
    from shap import maskers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    masker = maskers.Text(tokenizer, mask_token='<infill>')

    s = "This is a test string to be infilled"
    s_masked_with_infill_ex1 = "This is a test string <infill> <infill> <infill>"
    s_masked_with_infill_ex2 = "This is a <infill> <infill> to be infilled"
    s_masked_with_infill_ex3 = "<infill> <infill> <infill> test string to be infilled"
    s_masked_with_infill_ex4 = "<infill> <infill> <infill> <infill> <infill> <infill> <infill> <infill>"

    text_infilled_ex1 = masker.text_infill(s_masked_with_infill_ex1)
    expected_text_infilled_ex1 = "This is a test string..."

    text_infilled_ex2 = masker.text_infill(s_masked_with_infill_ex2)
    expected_text_infilled_ex2 = "This is a... to be infilled"

    text_infilled_ex3 = masker.text_infill(s_masked_with_infill_ex3)
    expected_text_infilled_ex3 = "... test string to be infilled"

    text_infilled_ex4 = masker.text_infill(s_masked_with_infill_ex4)
    expected_text_infilled_ex4 = "..."

    assert  text_infilled_ex1 == expected_text_infilled_ex1 and text_infilled_ex2 == expected_text_infilled_ex2 and \
            text_infilled_ex3 == expected_text_infilled_ex3 and text_infilled_ex4 == expected_text_infilled_ex4