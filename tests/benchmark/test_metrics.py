import numpy as np

from shap.benchmark import methods, metrics


class _Model:
    def fit(self, X, y):
        self.training_rows = X.copy()
        return self


def test_score_method_caches_a_model_per_train_test_split(tmp_path, monkeypatch):
    models = []

    def model_generator():
        model = _Model()
        models.append(model)
        return model

    def attribution_method(model, X_train):
        np.testing.assert_array_equal(model.training_rows, X_train)
        return lambda X: np.zeros_like(X)

    monkeypatch.setattr(methods, "test_attribution_method", attribution_method, raising=False)

    X = np.arange(24).reshape(12, 2)
    y = np.arange(12)

    def score_function(X_train, X_test, y_train, y_test, attr_function, model, random_state):
        np.testing.assert_array_equal(model.training_rows, X_train)
        return random_state

    score_method = metrics.__dict__["__score_method"]
    score_method(
        X,
        y,
        fcounts=None,
        model_generator=model_generator,
        score_function=score_function,
        method_name="test_attribution_method",
        nreps=2,
        test_size=2,
        cache_dir=tmp_path,
    )

    assert len(models) == 2
    assert len(list(tmp_path.glob("model_cache__v*.pickle"))) == 2
