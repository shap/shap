import numpy as np
import pytest

import shap


def test_embedding_invalid_method():
    """Ensure unsupported embedding methods raise a clean ValueError."""

    sv = np.random.randn(30, 5)

    with pytest.raises(ValueError, match="Unsupported embedding method"):
        shap.plots.embedding(0, sv, method="tsne")
