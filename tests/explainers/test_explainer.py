""" Tests for Explainer class.
"""

import numpy as np
import pandas as pd
import pytest

import shap


def test_wrapping_for_text_to_text_teacher_forcing_model():
    """ This tests using the Explainer class to auto wrap a masker in a text to text scenario.
    """

    transformers = pytest.importorskip("transformers")

    def f(x): # pylint: disable=unused-argument
        pass

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    wrapped_model = shap.models.TeacherForcing(f, similarity_model=model, similarity_tokenizer=tokenizer)
    masker = shap.maskers.Text(tokenizer, mask_token="...")

    explainer = shap.Explainer(wrapped_model, masker, seed=1)

    assert shap.utils.safe_isinstance(explainer.masker, "shap.maskers.OutputComposite")

def test_wrapping_for_topk_lm_model():
    """ This tests using the Explainer class to auto wrap a masker in a language modelling scenario.
    """

    transformers = pytest.importorskip("transformers")

    tokenizer = transformers.AutoTokenizer.from_pretrained("gpt2")
    model = transformers.AutoModelForCausalLM.from_pretrained("gpt2")
    wrapped_model = shap.models.TopKLM(model, tokenizer)
    masker = shap.maskers.Text(tokenizer, mask_token="...")

    explainer = shap.Explainer(wrapped_model, masker, seed=1)

    assert shap.utils.safe_isinstance(explainer.masker, "shap.maskers.FixedComposite")

def test_max_samples():
    # GH 3174
    N = 103
    x = np.arange(N)
    X = pd.DataFrame(dict(x1=x, x2=np.flip(x)))

    def true_model(X):
        return X.x1

    ex = shap.Explainer(true_model, masker=X, algorithm="exact", max_samples=1000)
    X_sliced = X[0:2]
    ps = ex(X_sliced)
    assert (ps.base_values == 51.).all()


def test_max_samples_not_defined():
    def true_model(X):
        return X.x1

    with pytest.raises(AssertionError):
        shap.Explainer(true_model, masker=None, max_samples=100)
