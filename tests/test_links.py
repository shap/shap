"""Tests for the `shap.links` module via the public Explainer interface."""

import numpy as np
import pytest
from sklearn.linear_model import LogisticRegression

import shap


@pytest.mark.parametrize("link", ["identity", "logit"])
def test_kernel_explainer_link(link):
    """Both link functions and their inverses are exercised by KernelExplainer.

    KernelExplainer stores ``self.link = convert_to_link(link)`` and applies
    ``link.f`` to reference outputs and SHAP value contributions. Additivity
    (SHAP values + expected_value == link(f(x))) requires both the forward and
    inverse directions to be consistent.
    """
    rs = np.random.RandomState(0)
    X = rs.randn(40, 3)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)
    model = LogisticRegression().fit(X, y)

    # single-output f so shap_values has a predictable shape
    def f(x):
        return model.predict_proba(x)[:, 1]

    explainer = shap.KernelExplainer(f, X[:20], link=link)
    shap_values = explainer.shap_values(X[:3], nsamples=50, silent=True)

    f_x = f(X[:3])
    link_fn = shap.links.logit if link == "logit" else shap.links.identity
    expected = link_fn(f_x)
    reconstructed = shap_values.sum(axis=-1) + explainer.expected_value
    np.testing.assert_allclose(reconstructed, expected, atol=1e-6)


def test_logit_inverse_roundtrip_via_explainer_expected_value():
    """link.inverse round-trips through the explainer's stored expected_value.

    The KernelExplainer applies ``link.f`` to ``fnull`` to get ``expected_value``,
    so ``link.inverse(expected_value)`` must recover the original mean prediction.
    """
    rs = np.random.RandomState(0)
    X = rs.randn(30, 2)
    y = (X[:, 0] > 0).astype(int)
    model = LogisticRegression().fit(X, y)

    def f(x):
        return model.predict_proba(x)[:, 1]

    explainer = shap.KernelExplainer(f, X[:15], link="logit")
    mean_prob = f(X[:15]).mean()
    np.testing.assert_allclose(shap.links.logit.inverse(explainer.expected_value), mean_prob, atol=1e-6)
