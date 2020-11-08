from ._tree import Tree, feature_perturbation_codes, output_transform_codes
import numpy as np
import warnings
from ..utils import assert_import,  safe_isinstance
from .. import _cext_gpu
from .. import _cext


class GPUTree(Tree):

    def shap_values(self, X, y=None, tree_limit=None, approximate=False, check_additivity=True,
                    from_call=False):
        """ Estimate the SHAP values for a set of samples.

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
            Run fast, but only roughly approximate the Tree SHAP values. This runs a method
            previously proposed by Saabas which only considers a single feature ordering. Take care
            since this does not have the consistency guarantees of Shapley values and places too
            much weight on lower splits in the tree.

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

        if check_additivity and self.model.model_type == "pyspark":
            warnings.warn(
                "check_additivity requires us to run predictions which is not supported with "
                "spark, "
                "ignoring."
                " Set check_additivity=False to remove this warning")
            check_additivity = False

        # see if we have a default tree_limit in place.
        if tree_limit is None:
            tree_limit = -1 if self.model.tree_limit is None else self.model.tree_limit

        # convert dataframes
        if safe_isinstance(X, "pandas.core.series.Series"):
            X = X.values
        elif safe_isinstance(X, "pandas.core.frame.DataFrame"):
            X = X.values
        flat_output = False
        if len(X.shape) == 1:
            flat_output = True
            X = X.reshape(1, X.shape[0])
        if X.dtype != self.model.input_dtype:
            X = X.astype(self.model.input_dtype)
        X_missing = np.isnan(X, dtype=np.bool)
        assert isinstance(X, np.ndarray), "Unknown instance type: " + str(type(X))
        assert len(X.shape) == 2, "Passed input data matrix X must have 1 or 2 dimensions!"

        if tree_limit < 0 or tree_limit > self.model.values.shape[0]:
            tree_limit = self.model.values.shape[0]

        if self.model.model_output == "log_loss":
            assert y is not None, "Both samples and labels must be provided when model_output = " \
                                  "\"log_loss\" (i.e. `explainer.shap_values(X, y)`)!"
            assert X.shape[0] == len(
                y), "The number of labels (%d) does not match the number of samples to explain (" \
                    "%d)!" \
                    % (
                        len(y), X.shape[0])
        transform = self.model.get_transform()

        if self.feature_perturbation == "tree_path_dependent":
            assert self.model.fully_defined_weighting, "The background dataset you provided does " \
                                                       "not " \
                                                       "cover all the leaves in the model, " \
                                                       "so TreeExplainer cannot run with the " \
                                                       "feature_perturbation=\"tree_path_dependent\" " \
                                                       "option! " \
                                                       "Try providing a larger background " \
                                                       "dataset, " \
                                                       "or using " \
                                                       "feature_perturbation=\"interventional\"."

        # run the core algorithm using the C extension
        assert_import("cext")
        phi = np.zeros((X.shape[0], X.shape[1] + 1, self.model.num_outputs))
        _cext_gpu.dense_tree_shap(
            self.model.children_left, self.model.children_right, self.model.children_default,
            self.model.features, self.model.thresholds, self.model.values,
            self.model.node_sample_weight,
            self.model.max_depth, X, X_missing, y, self.data, self.data_missing, tree_limit,
            self.model.base_offset, phi, feature_perturbation_codes[self.feature_perturbation],
            output_transform_codes[transform], False
        )

        # note we pull off the last column and keep it as our expected_value
        if self.model.num_outputs == 1:
            if self.expected_value is None and self.model.model_output != "log_loss":
                self.expected_value = phi[0, -1, 0]
            if flat_output:
                out = phi[0, :-1, 0]
            else:
                out = phi[:, :-1, 0]
        else:
            if self.expected_value is None and self.model.model_output != "log_loss":
                self.expected_value = [phi[0, -1, i] for i in range(phi.shape[2])]
            if flat_output:
                out = [phi[0, :-1, i] for i in range(self.model.num_outputs)]
            else:
                out = [phi[:, :-1, i] for i in range(self.model.num_outputs)]

        if check_additivity and self.model.model_output == "raw":
            self.assert_additivity(out, self.model.predict(X))

        # if our output format requires binary classificaiton to be represented as two outputs
        # then we do that here
        if self.model.model_output == "probability_doubled":
            out = [-out, out]

        return out
