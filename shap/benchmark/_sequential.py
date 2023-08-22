import time

import matplotlib.pyplot as pl
import numpy as np
import pandas as pd
import sklearn
from tqdm.auto import tqdm

from shap import Explanation, links
from shap.maskers import FixedComposite, Image, Text
from shap.utils import MaskedModel, safe_isinstance

from ._result import BenchmarkResult


class SequentialMasker:
    def __init__(self, mask_type, sort_order, masker, model, *model_args, batch_size=500):

        for arg in model_args:
            if isinstance(arg, pd.DataFrame):
                raise TypeError("DataFrame arguments dont iterate correctly, pass numpy arrays instead!")

        # convert any DataFrames to numpy arrays
        # self.model_arg_cols = []
        # self.model_args = []
        # self.has_df = False
        # for arg in model_args:
        #     if isinstance(arg, pd.DataFrame):
        #         self.model_arg_cols.append(arg.columns)
        #         self.model_args.append(arg.values)
        #         self.has_df = True
        #     else:
        #         self.model_arg_cols.append(None)
        #         self.model_args.append(arg)

        # if self.has_df:
        #     given_model = model
        #     def new_model(*args):
        #         df_args = []
        #         for i, arg in enumerate(args):
        #             if self.model_arg_cols[i] is not None:
        #                 df_args.append(pd.DataFrame(arg, columns=self.model_arg_cols[i]))
        #             else:
        #                 df_args.append(arg)
        #         return given_model(*df_args)
        #     model = new_model

        self.inner = SequentialPerturbation(
            model, masker, sort_order, mask_type
        )
        self.model_args = model_args
        self.batch_size = batch_size

    def __call__(self, explanation, name, **kwargs):
        return self.inner(name, explanation, *self.model_args, batch_size=self.batch_size, **kwargs)

class SequentialPerturbation:
    def __init__(self, model, masker, sort_order, perturbation, linearize_link=False):
        # self.f = lambda masked, x, index: model.predict(masked)
        self.model = model if callable(model) else model.predict
        self.masker = masker
        self.sort_order = sort_order
        self.perturbation = perturbation
        self.linearize_link = linearize_link

        # define our sort order
        if self.sort_order == "positive":
            self.sort_order_map = lambda x: np.argsort(-x)
        elif self.sort_order == "negative":
            self.sort_order_map = lambda x: np.argsort(x)
        elif self.sort_order == "absolute":
            self.sort_order_map = lambda x: np.argsort(-abs(x))
        else:
            raise ValueError("sort_order must be either \"positive\", \"negative\", or \"absolute\"!")

        # user must give valid masker
        underlying_masker = masker.masker if isinstance(masker, FixedComposite) else masker
        if isinstance(underlying_masker, Text):
            self.data_type = "text"
        elif isinstance(underlying_masker, Image):
            self.data_type = "image"
        else:
            self.data_type = "tabular"
            #raise ValueError("masker must be for \"tabular\", \"text\", or \"image\"!")

        self.score_values = []
        self.score_aucs = []
        self.labels = []

    def __call__(self, name, explanation, *model_args, percent=0.01, indices=[], y=None, label=None, silent=False, debug_mode=False, batch_size=10):
        # if explainer is already the attributions
        if safe_isinstance(explanation, "numpy.ndarray"):
            attributions = explanation
        elif isinstance(explanation, Explanation):
            attributions = explanation.values
        else:
            raise ValueError("The passed explanation must be either of type numpy.ndarray or shap.Explanation!")

        assert len(attributions) == len(model_args[0]), "The explanation passed must have the same number of rows as the model_args that were passed!"

        if label is None:
            label = "Score %d" % len(self.score_values)

        # convert dataframes
        # if safe_isinstance(X, "pandas.core.series.Series") or safe_isinstance(X, "pandas.core.frame.DataFrame"):
        #     X = X.values

        # convert all single-sample vectors to matrices
        # if not hasattr(attributions[0], "__len__"):
        #     attributions = np.array([attributions])
        # if not hasattr(X[0], "__len__") and self.data_type == "tabular":
        #     X = np.array([X])

        pbar = None
        start_time = time.time()
        svals = []
        mask_vals = []

        for i, args in enumerate(zip(*model_args)):
            # if self.data_type == "image":
            #     x_shape, y_shape = attributions[i].shape[0], attributions[i].shape[1]
            #     feature_size = np.prod([x_shape, y_shape])
            #     sample_attributions = attributions[i].mean(2).reshape(feature_size, -1)
            #     data = X[i].flatten()
            #     mask_shape = X[i].shape
            # else:
            feature_size = np.prod(attributions[i].shape)
            sample_attributions = attributions[i].flatten()
            # data = X[i]
            # mask_shape = feature_size

            self.masked_model = MaskedModel(self.model, self.masker, links.identity, self.linearize_link, *args)

            masks = []

            mask = np.ones(feature_size, dtype=bool) * (self.perturbation == "remove")
            masks.append(mask.copy())

            ordered_inds = self.sort_order_map(sample_attributions)
            increment = max(1,int(feature_size*percent))
            for j in range(0, feature_size, increment):
                oind_list = [ordered_inds[l] for l in range(j, min(feature_size, j+increment))]

                for oind in oind_list:
                    if not ((self.sort_order == "positive" and sample_attributions[oind] <= 0) or \
                            (self.sort_order == "negative" and sample_attributions[oind] >= 0)):
                        mask[oind] = self.perturbation == "keep"

                masks.append(mask.copy())

            mask_vals.append(masks)

            # mask_size = len(range(0, feature_size, increment)) + 1
            values = []
            masks_arr = np.array(masks)
            for j in range(0, len(masks_arr), batch_size):
                values.append(self.masked_model(masks_arr[j:j + batch_size]))
            values = np.concatenate(values)

            svals.append(values)

            if pbar is None and time.time() - start_time > 5:
                pbar = tqdm(total=len(model_args[0]), disable=silent, leave=False, desc="SequentialMasker")
                pbar.update(i+1)
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self.score_values.append(np.array(svals))

        # if self.sort_order == "negative":
        #     curve_sign = -1
        # else:
        curve_sign = 1

        self.labels.append(label)

        xs = np.linspace(0, 1, 100)
        curves = np.zeros((len(self.score_values[-1]), len(xs)))
        for j in range(len(self.score_values[-1])):
            xp = np.linspace(0, 1, len(self.score_values[-1][j]))
            yp = self.score_values[-1][j]
            curves[j,:] = np.interp(xs, xp, yp)
        ys = curves.mean(0)
        std = curves.std(0) / np.sqrt(curves.shape[0])
        auc = sklearn.metrics.auc(np.linspace(0, 1, len(ys)), curve_sign*(ys-ys[0]))

        if not debug_mode:
            return BenchmarkResult(self.perturbation + " " + self.sort_order, name, curve_x=xs, curve_y=ys, curve_y_std=std)
        else:
            aucs = []
            for j in range(len(self.score_values[-1])):
                curve = curves[j,:]
                auc = sklearn.metrics.auc(np.linspace(0, 1, len(curve)), curve_sign*(curve-curve[0]))
                aucs.append(auc)
            return mask_vals, curves, aucs

    def score(self, explanation, X, percent=0.01, y=None, label=None, silent=False, debug_mode=False):
        '''
        Will be deprecated once MaskedModel is in complete support
        '''
        # if explainer is already the attributions
        if safe_isinstance(explanation, "numpy.ndarray"):
            attributions = explanation
        elif isinstance(explanation, Explanation):
            attributions = explanation.values

        if label is None:
            label = "Score %d" % len(self.score_values)

        # convert dataframes
        if safe_isinstance(X, "pandas.core.series.Series") or safe_isinstance(X, "pandas.core.frame.DataFrame"):
            X = X.values

        # convert all single-sample vectors to matrices
        if not hasattr(attributions[0], "__len__"):
            attributions = np.array([attributions])
        if not hasattr(X[0], "__len__") and self.data_type == "tabular":
            X = np.array([X])

        pbar = None
        start_time = time.time()
        svals = []
        mask_vals = []

        for i in range(len(X)):
            if self.data_type == "image":
                x_shape, y_shape = attributions[i].shape[0], attributions[i].shape[1]
                feature_size = np.prod([x_shape, y_shape])
                sample_attributions = attributions[i].mean(2).reshape(feature_size, -1)
            else:
                feature_size = attributions[i].shape[0]
                sample_attributions = attributions[i]

            if len(attributions[i].shape) == 1 or self.data_type == "tabular":
                output_size = 1
            else:
                output_size = attributions[i].shape[-1]

            for k in range(output_size):
                if self.data_type == "image":
                    mask_shape = X[i].shape
                else:
                    mask_shape = feature_size

                mask = np.ones(mask_shape, dtype=bool) * (self.perturbation == "remove")
                masks = [mask.copy()]

                values = np.zeros(feature_size+1)
                # masked, data = self.masker(mask, X[i])
                masked = self.masker(mask, X[i])
                data = None
                curr_val = self.f(masked, data, k).mean(0)

                values[0] = curr_val

                if output_size != 1:
                    test_attributions = sample_attributions[:,k]
                else:
                    test_attributions = sample_attributions

                ordered_inds = self.sort_order_map(test_attributions)
                increment = max(1,int(feature_size*percent))
                for j in range(0, feature_size, increment):
                    oind_list = [ordered_inds[l] for l in range(j, min(feature_size, j+increment))]

                    for oind in oind_list:
                        if not ((self.sort_order == "positive" and test_attributions[oind] <= 0) or \
                                (self.sort_order == "negative" and test_attributions[oind] >= 0)):
                            if self.data_type == "image":
                                xoind, yoind = oind // attributions[i].shape[1], oind % attributions[i].shape[1]
                                mask[xoind][yoind] = self.perturbation == "keep"
                            else:
                                mask[oind] = self.perturbation == "keep"

                    masks.append(mask.copy())
                    # masked, data = self.masker(mask, X[i])
                    masked = self.masker(mask, X[i])
                    curr_val = self.f(masked, data, k).mean(0)

                    for l in range(j, min(feature_size, j+increment)):
                        values[l+1] = curr_val

                svals.append(values)
                mask_vals.append(masks)

            if pbar is None and time.time() - start_time > 5:
                pbar = tqdm(total=len(X), disable=silent, leave=False)
                pbar.update(i+1)
            if pbar is not None:
                pbar.update(1)

        if pbar is not None:
            pbar.close()

        self.score_values.append(np.array(svals))

        if self.sort_order == "negative":
            curve_sign = -1
        else:
            curve_sign = 1

        self.labels.append(label)

        xs = np.linspace(0, 1, 100)
        curves = np.zeros((len(self.score_values[-1]), len(xs)))
        for j in range(len(self.score_values[-1])):
            xp = np.linspace(0, 1, len(self.score_values[-1][j]))
            yp = self.score_values[-1][j]
            curves[j,:] = np.interp(xs, xp, yp)
        ys = curves.mean(0)

        if debug_mode:
            aucs = []
            for j in range(len(self.score_values[-1])):
                curve = curves[j,:]
                auc = sklearn.metrics.auc(np.linspace(0, 1, len(curve)), curve_sign*(curve-curve[0]))
                aucs.append(auc)
            return mask_vals, curves, aucs
        else:
            auc = sklearn.metrics.auc(np.linspace(0, 1, len(ys)), curve_sign*(ys-ys[0]))
            return xs, ys, auc

    def plot(self, xs, ys, auc):
        pl.plot(xs, ys, label="AUC %0.4f" % auc)
        pl.legend()
        xlabel = "Percent Unmasked" if self.perturbation == "keep" else "Percent Masked"
        pl.xlabel(xlabel)
        pl.ylabel("Model Output")
        pl.show()
