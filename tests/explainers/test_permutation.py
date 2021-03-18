""" Unit tests for the Permutation explainer.
"""

# pylint: disable=missing-function-docstring
import pickle
import shap
from . import common


def test_tabular_single_output_auto_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.Permutation, model.predict, data, data)

def test_tabular_multi_output_auto_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.Permutation, model.predict_proba, data, data)

def test_tabular_single_output_partition_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.Permutation, model.predict, shap.maskers.Partition(data), data)

def test_tabular_multi_output_partition_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.Permutation, model.predict_proba, shap.maskers.Partition(data), data)

def test_tabular_single_output_independent_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.Permutation, model.predict, shap.maskers.Independent(data), data)

def test_tabular_multi_output_independent_masker():
    model, data = common.basic_xgboost_scenario(100)
    common.test_additivity(shap.explainers.Permutation, model.predict_proba, shap.maskers.Independent(data), data)

def test_serialization():
    model, data = common.basic_xgboost_scenario()
    common.test_serialization(
        shap.explainers.Permutation, model.predict, data, data,
        rtol=0.1, atol=0.05, max_evals=100000
    )

def test_serialization_no_model_or_masker():
    model, data = common.basic_xgboost_scenario()
    common.test_serialization(
        shap.explainers.Permutation, model.predict, data, data,
        model_saver=False, masker_saver=False,
        model_loader=lambda _: model.predict, masker_loader=lambda _: data,
        rtol=0.1, atol=0.05, max_evals=100000
    )

def test_serialization_custom_model_save():
    model, data = common.basic_xgboost_scenario()
    common.test_serialization(
        shap.explainers.Permutation, model.predict, data, data,
        model_saver=pickle.dump, model_loader=pickle.load, rtol=0.1, atol=0.05, max_evals=100000
    )

def test_downstream_integration_mlflow():
    
    import mlflow
    import sklearn
    import numpy as np
    from mlflow.tracking.artifact_utils import _download_artifact_from_uri
    from mlflow.utils.model_utils import _get_flavor_configuration

    with mlflow.start_run() as run:

        run_id = run.info.run_id

        X, y = shap.datasets.boston()
        model = sklearn.ensemble.RandomForestRegressor(n_estimators=100)
        model.fit(X, y)

        explainer_original = shap.Explainer(model.predict, X, algorithm="permutation")
        shap_values_original = explainer_original(X[:5])

        mlflow.shap.log_explainer(explainer_original, "test_explainer")

        explainer_uri = "runs:/" + run_id + "/test_explainer"

        explainer_loaded = mlflow.shap.load_explainer(explainer_uri)
        shap_values_new = explainer_loaded(X[:5])

        explainer_path = _download_artifact_from_uri(artifact_uri=explainer_uri)
        flavor_conf = _get_flavor_configuration(
            model_path=explainer_path, flavor_name=mlflow.shap.FLAVOR_NAME
        )
        underlying_model_flavor = flavor_conf["underlying_model_flavor"]

        assert underlying_model_flavor == mlflow.sklearn.FLAVOR_NAME
        np.testing.assert_array_equal(shap_values_original.base_values, shap_values_new.base_values)
        np.testing.assert_allclose(
            shap_values_original.values, shap_values_new.values, rtol=100, atol=100
        )
