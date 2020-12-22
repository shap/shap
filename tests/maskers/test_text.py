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

    assert output_masked_text[0] == correct_masked_text

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
    
    assert output_masked_text[0] == correct_masked_text

def test_sentencepiece_tokenizer_output():
    """ Tests for output for sentencepiece tokenizers to not have '_' in output of masker when passed a mask of ones.
    """
    import numpy as np
    from transformers import AutoTokenizer
    from shap import maskers
    from shap.utils import safe_isinstance
    
    tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    masker = maskers.Text(tokenizer)

    s="This is a test statement for sentencepiece tokenizer"
    masker._update_s_cache(s)
    mask = np.ones(masker._segments_s.shape, dtype=bool)

    sentencepiece_tokenizer_output_processed = masker(mask, s)
    expected_sentencepiece_tokenizer_output_processed = "This is a test statement for sentencepiece tokenizer"
    # since we expect output wrapped in a tuple hence the indexing [0][0] to extract the string
    assert sentencepiece_tokenizer_output_processed[0][0] == expected_sentencepiece_tokenizer_output_processed

def test_keep_prefix_suffix_tokenizer_parsing():
    """ Checks parsed keep prefix and keep suffix for different tokenizers.
    """ 
    from transformers import AutoTokenizer
    from shap import maskers

    tokenizer_mt = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-es")
    tokenizer_gpt = AutoTokenizer.from_pretrained("gpt2")
    tokenizer_bart = AutoTokenizer.from_pretrained("sshleifer/distilbart-xsum-12-6")

    masker_mt = maskers.Text(tokenizer_mt)
    masker_gpt = maskers.Text(tokenizer_gpt)
    masker_bart = maskers.Text(tokenizer_bart)

    masker_mt_expected_keep_prefix, masker_mt_expected_keep_suffix = 0, 1
    masker_gpt_expected_keep_prefix, masker_gpt_expected_keep_suffix = 0, 0
    masker_bart_expected_keep_prefix, masker_bart_expected_keep_suffix = 1, 1

    assert masker_mt.keep_prefix == masker_mt_expected_keep_prefix and masker_mt.keep_suffix == masker_mt_expected_keep_suffix and \
           masker_gpt.keep_prefix == masker_gpt_expected_keep_prefix and masker_gpt.keep_suffix == masker_gpt_expected_keep_suffix and \
           masker_bart.keep_prefix == masker_bart_expected_keep_prefix and masker_bart.keep_suffix == masker_bart_expected_keep_suffix


def test_text_infill_with_collapse_mask_token():
    """ Tests for different text infilling output combinations with collapsing mask token.
    """
    import numpy as np
    from shap import maskers
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    masker = maskers.Text(tokenizer, mask_token='...', collapse_mask_token=True)

    s = "This is a test string to be infilled"

    # s_masked_with_infill_ex1 = "This is a test string ... ... ..."
    mask_ex1 = np.array([True, True, True, True, True, False, False, False, False])
    # s_masked_with_infill_ex2 = "This is a ... ... to be infilled"
    mask_ex2 = np.array([True, True, True, False, False, True, True, True, True])
    # s_masked_with_infill_ex3 = "... ... ... test string to be infilled"
    mask_ex3 = np.array([False, False, False, True, True, True, True, True, True])
    # s_masked_with_infill_ex4 = "... ... ... ... ... ... ... ..."
    mask_ex4 = np.array([False, False, False, False, False, False, False, False, False])

    text_infilled_ex1 = masker(mask_ex1, s)[0][0]
    expected_text_infilled_ex1 = "This is a test string ..."

    text_infilled_ex2 = masker(mask_ex2, s)[0][0]
    expected_text_infilled_ex2 = "This is a ... to be infilled"

    text_infilled_ex3 = masker(mask_ex3, s)[0][0]
    expected_text_infilled_ex3 = "... test string to be infilled"

    text_infilled_ex4 = masker(mask_ex4, s)[0][0]
    expected_text_infilled_ex4 = "..."

    assert  text_infilled_ex1 == expected_text_infilled_ex1 and text_infilled_ex2 == expected_text_infilled_ex2 and \
            text_infilled_ex3 == expected_text_infilled_ex3 and text_infilled_ex4 == expected_text_infilled_ex4

def test_serialization_text_masker():
    import shap
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import tempfile

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased",use_fast=False)
    original_masker = shap.maskers.Text(tokenizer)

    temp_serialization_file = tempfile.TemporaryFile()

    original_masker.save(temp_serialization_file)

    temp_serialization_file.seek(0)


    # deserialize masker
    new_masker = shap.maskers.Text.load(temp_serialization_file)

    temp_serialization_file.close()


    test_text = "I ate a Cannoli"
    test_input_mask = np.array([True, False, True, True, False, True, True, True])

    original_masked_output = original_masker(test_input_mask,test_text)
    new_masked_output = new_masker(test_input_mask,test_text)

    assert original_masked_output == new_masked_output

def test_serialization_text_masker_custom_mask():
    import shap
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import tempfile

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased",use_fast=True)
    original_masker = shap.maskers.Text(tokenizer, mask_token = '[CUSTOM-MASK]')

    temp_serialization_file = tempfile.TemporaryFile()

    original_masker.save(temp_serialization_file)

    temp_serialization_file.seek(0)

    # deserialize masker
    new_masker = shap.maskers.Text.load(temp_serialization_file)

    temp_serialization_file.close()


    test_text = "I ate a Cannoli"
    test_input_mask = np.array([True, False, True, True, False, True, True, True])

    original_masked_output = original_masker(test_input_mask, test_text)
    new_masked_output = new_masker(test_input_mask, test_text)

    assert original_masked_output == new_masked_output

def test_serialization_text_masker_collapse_mask_token():
    import shap
    import numpy as np
    from transformers import AutoTokenizer, AutoModelForSequenceClassification
    import tempfile

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-cased",use_fast=True)
    original_masker = shap.maskers.Text(tokenizer, collapse_mask_token=True)

    temp_serialization_file = tempfile.TemporaryFile()

    original_masker.save(temp_serialization_file)

    temp_serialization_file.seek(0)

    # deserialize masker
    new_masker = shap.maskers.Text.load(temp_serialization_file)

    temp_serialization_file.close()


    test_text = "I ate a Cannoli"
    test_input_mask = np.array([True, False, True, True, False, True, True, True])

    original_masked_output = original_masker(test_input_mask, test_text)
    new_masked_output = new_masker(test_input_mask, test_text)

    assert original_masked_output == new_masked_output