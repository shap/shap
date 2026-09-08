"""Tests for the embedding plot."""

import numpy as np
import pytest

import shap


def test_embedding_unsupported_method_raises():
    """Check that a ValueError is raised when an unsupported method is passed."""
    rng = np.random.default_rng(0)
    shap_values = rng.standard_normal((20, 5))
    emsg = "Unsupported embedding method"
    with pytest.raises(ValueError, match=emsg):
        shap.plots.embedding(0, shap_values, method="tsne", show=False)
