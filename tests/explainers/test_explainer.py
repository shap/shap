"""Tests for Explainer class."""

import pytest
import sklearn

import shap


def test_explainer_to_permutationexplainer():
    """Checks that Explainer maps to PermutationExplainer as expected."""
    X_train, X_test, y_train, _ = sklearn.model_selection.train_test_split(
        *shap.datasets.adult(), test_size=0.1, random_state=0
    )
    lr = sklearn.linear_model.LogisticRegression(solver="liblinear")
    lr.fit(X_train, y_train)

    explainer = shap.Explainer(lr.predict_proba, masker=X_train)
    assert isinstance(explainer, shap.PermutationExplainer)

    # ensures a proper error message is raised if a masker is not provided (GH #3310)
    with pytest.raises(
        ValueError,
        match=r"masker cannot be None",
    ):
        explainer = shap.Explainer(lr.predict_proba)
        _ = explainer(X_test)


def test_explainer_algorithm_kernel():
    """Checks that Explainer dispatches to KernelExplainer when ``algorithm='kernel'`` is set.

    GH #3996.
    """
    X_train, _, y_train, _ = sklearn.model_selection.train_test_split(
        *shap.datasets.adult(), test_size=0.1, random_state=0
    )
    lr = sklearn.linear_model.LogisticRegression(solver="liblinear")
    lr.fit(X_train, y_train)

    # KernelExplainer warns about background sets > 100 rows; keep the sample
    # small so the test stays fast and quiet.
    explainer = shap.Explainer(lr.predict_proba, masker=X_train.iloc[:50], algorithm="kernel")
    assert isinstance(explainer, shap.KernelExplainer)


def test_explainer_rejects_unknown_algorithm():
    """Unknown algorithm strings should raise InvalidAlgorithmError.

    Guards the else-branch of the dispatch against silent fallthrough.
    """
    from shap.utils._exceptions import InvalidAlgorithmError

    X_train, _, y_train, _ = sklearn.model_selection.train_test_split(
        *shap.datasets.adult(), test_size=0.1, random_state=0
    )
    lr = sklearn.linear_model.LogisticRegression(solver="liblinear")
    lr.fit(X_train, y_train)

    with pytest.raises(InvalidAlgorithmError, match="Unknown algorithm"):
        shap.Explainer(lr.predict_proba, masker=X_train.iloc[:20], algorithm="not-a-real-algorithm")  # type: ignore[arg-type]


def test_wrapping_for_text_to_text_teacher_forcing_model():
    """This tests using the Explainer class to auto wrap a masker in a text to text scenario."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    def f(x):
        pass

    name = "hf-internal-testing/tiny-random-BartForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name)
    wrapped_model = shap.models.TeacherForcing(f, similarity_model=model, similarity_tokenizer=tokenizer)
    masker = shap.maskers.Text(tokenizer, mask_token="...")

    explainer = shap.Explainer(wrapped_model, masker, seed=1)

    assert shap.utils.safe_isinstance(explainer.masker, "shap.maskers.OutputComposite")


def test_transformers_label_to_id_mapping_enforces_ints():
    """This tests that when we construct our TransformersPipeline, we enforce that label2id values are ints."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"
    pipe = transformers.pipeline("text-classification", name)

    # Make the model label2id mapping have str values
    # to test that our TransformersPipeline converts them to int
    pipe.model.config.label2id = {k: str(v) for k, v in pipe.model.config.label2id.items()}

    # Finish constructing the Explainer
    explainer = shap.Explainer(pipe, seed=1)

    # Check that the label2id values are all ints after construction
    assert isinstance(explainer.model, shap.models.TransformersPipeline)
    assert all(isinstance(v, int) for v in explainer.model.label2id.values())


def test_wrapping_for_topk_lm_model():
    """This tests using the Explainer class to auto wrap a masker in a language modelling scenario."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")

    name = "hf-internal-testing/tiny-random-BartForCausalLM"
    tokenizer = transformers.AutoTokenizer.from_pretrained(name)
    model = transformers.AutoModelForCausalLM.from_pretrained(name)
    wrapped_model = shap.models.TopKLM(model, tokenizer)
    masker = shap.maskers.Text(tokenizer, mask_token="...")

    explainer = shap.Explainer(wrapped_model, masker, seed=1)

    assert shap.utils.safe_isinstance(explainer.masker, "shap.maskers.FixedComposite")


def test_explainer_xgboost():
    """Check the explainer class wraps a TreeExplainer as expected"""
    # train an XGBoost model
    xgboost = pytest.importorskip("xgboost")
    X, y = shap.datasets.california(n_points=500)
    model = xgboost.XGBRegressor().fit(X, y)

    # explain the model's predictions
    explainer = shap.Explainer(model)
    explanation = explainer(X)

    # check the properties of Explanation object
    assert explanation.values.shape == (*X.shape,)  # type: ignore[union-attr]
    assert explanation.base_values.shape == (len(X),)  # type: ignore[union-attr]
