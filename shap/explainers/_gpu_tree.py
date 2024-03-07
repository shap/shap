"""GPU accelerated tree explanations"""
import numpy as np

from ..utils import assert_import, record_import_error
from ._tree import (
    TreeExplainer,
    _xgboost_cat_unsupported,
    feature_perturbation_codes,
    output_transform_codes,
)

try:
    from .. import _cext_gpu
except ImportError as e:
    record_import_error("cext_gpu", "cuda extension was not built during install!", e)


class GPUTreeExplainer(TreeExplainer):
    """Experimental GPU accelerated version of TreeExplainer. Currently requires source build with
    cuda available and 'CUDA_PATH' environment variable defined.

    Examples
    --------
    See `GPUTree explainer examples <https://shap.readthedocs.io/en/latest/api_examples/explainers/GPUTreeExplainer.html>`_

    """

    def shap_values(self, X, y=None, tree_limit=None, approximate=False, check_additivity=True,
                    from_call=False):
        """Estimate the SHAP values for a set of samples.

        Parameters
        ----------
        X : numpy.array, pandas.DataFrame or catboost.Pool (for catboost)
            A matrix of samples (# samples x # features) on which to explain the model's output.

        y : numpy.array
            An array of label values for each sample. Used when explaining loss functions.

        tree_limit : None (default) or int
            Limit the number of trees used by the model. By default None means no use the limit
            of the
            original model, and -1 means no limit.

        approximate : bool
            Not supported.

        check_additivity : bool
            Run a validation check that the sum of the SHAP values equals the output of the
            model. This
            check takes only a small amount of time, and will catch potential unforeseen errors.
            Note that this check only runs right now when explaining the margin of the model.

        Returns
        -------
        array or list
            For models with a single output this returns a matrix of SHAP values
            (# samples x # features). Each row sums to the difference between the model output
            for that
            sample and the expected value of the model output (which is stored in the expected_value
            attribute of the explainer when it is constant). For models with vector outputs this
            returns
            a list of such matrices, one for each output.

        """
        assert not approximate, "approximate not supported"

        X, y, X_missing, flat_output, tree_limit, check_additivity = \
            self._validate_inputs(X, y, tree_limit, check_additivity)

        model = self.model
        _xgboost_cat_unsupported(model)
        transform = model.get_transform()

        # run the core algorithm using the C extension
        assert_import("cext_gpu")
        phi = np.zeros((X.shape[0], X.shape[1] + 1, self.model.num_outputs))
        _cext_gpu.dense_tree_shap(
            self.model.children_left, self.model.children_right, self.model.children_default,
            self.model.features, self.model.thresholds, self.model.values,
            self.model.node_sample_weight,
            self.model.max_depth, X, X_missing, y, self.data, self.data_missing, tree_limit,
            self.model.base_offset, phi, feature_perturbation_codes[self.feature_perturbation],
            output_transform_codes[transform], False
        )

        out = self._get_shap_output(phi, flat_output)
        if check_additivity and self.model.model_output == "raw":
            self.assert_additivity(out, self.model.predict(X))

        return out

    def shap_interaction_values(self, X, y=None, tree_limit=None):
        """Estimate the SHAP interaction values for a set of samples.

        Parameters
        ----------
        X : numpy.array, pandas.DataFrame or catboost.Pool (for catboost)
            A matrix of samples (# samples x # features) on which to explain the model's output.

        y : numpy.array
            An array of label values for each sample. Used when explaining loss functions (not
            yet supported).

        tree_limit : None (default) or int
            Limit the number of trees used by the model. By default None means no use the limit
            of the
            original model, and -1 means no limit.

        Returns
        -------
        array or list
            For models with a single output this returns a tensor of SHAP values
            (# samples x # features x # features). The matrix (# features x # features) for each
            sample sums
            to the difference between the model output for that sample and the expected value of
            the model output
            (which is stored in the expected_value attribute of the explainer). Each row of this
            matrix sums to the
            SHAP value for that feature for that sample. The diagonal entries of the matrix
            represent the
            "main effect" of that feature on the prediction and the symmetric off-diagonal
            entries represent the
            interaction effects between all pairs of features for that sample. For models with
            vector outputs
            this returns a list of tensors, one for each output.

        """
        assert self.model.model_output == "raw", "Only model_output = \"raw\" is supported for " \
                                                 "SHAP interaction values right now!"
        assert self.feature_perturbation != "interventional", 'feature_perturbation="interventional" is not yet supported for ' + \
                                                              'interaction values. Use feature_perturbation="tree_path_dependent" instead.'
        transform = "identity"

        X, y, X_missing, flat_output, tree_limit, _ = self._validate_inputs(X, y, tree_limit,
                                                                            False)
        # run the core algorithm using the C extension
        assert_import("cext_gpu")
        phi = np.zeros((X.shape[0], X.shape[1] + 1, X.shape[1] + 1, self.model.num_outputs))
        _cext_gpu.dense_tree_shap(
            self.model.children_left, self.model.children_right, self.model.children_default,
            self.model.features, self.model.thresholds, self.model.values,
            self.model.node_sample_weight,
            self.model.max_depth, X, X_missing, y, self.data, self.data_missing, tree_limit,
            self.model.base_offset, phi, feature_perturbation_codes[self.feature_perturbation],
            output_transform_codes[transform], True
        )

        return self._get_shap_interactions_output(phi, flat_output)
