""" Tests for Explainer class.
"""

import pytest
import shap


def test_wrapping_for_text_to_text_teacher_forcing_logits_model():
    """ This tests using the Explainer class to auto choose a text to text setup.
    """

    transformers = pytest.importorskip("transformers")

    def f(x): # pylint: disable=unused-argument
        pass

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    wrapped_model = shap.models.TeacherForcingLogits(f, similarity_model=model, similarity_tokenizer=tokenizer)
    masker = shap.maskers.Text(tokenizer, mask_token="...")

    explainer = shap.Explainer(wrapped_model, masker)

    assert shap.utils.safe_isinstance(explainer.masker, "shap.maskers.FixedComposite")
