import numpy as np
from scipy.sparse import csr_matrix
from sklearn.ensemble import RandomForestClassifier

import shap


def test_sparse_input_tree_explainer():
    X = csr_matrix(np.random.randint(0, 2, (10, 5)))
    y = np.random.randint(0, 2, 10)

    model = RandomForestClassifier().fit(X, y)
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X)

    assert isinstance(shap_values, np.ndarray)
