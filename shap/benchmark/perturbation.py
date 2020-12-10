from shap.utils import safe_isinstance
from shap.maskers import Text, Image
from shap import Explanation
import matplotlib.pyplot as pl
import sklearn
import numpy as np
from tqdm.auto import tqdm
import time


class SequentialPerturbation():
    def __init__(self, f, masker, sort_order, perturbation, data_type=None, **kwargs):
        self.f = f
        self.masker = masker
        self.sort_order = sort_order
        self.perturbation = perturbation
        self.data_type = data_type

        if "f_kwargs" in kwargs:
            self.kwargs = kwargs['f_kwargs']
        elif "kwargs" in kwargs:
            self.kwargs = kwargs['kwargs']
        else:
            self.kwargs = None

        # convert dataframe
        if safe_isinstance(self.masker, "pandas.core.series.Series") or safe_isinstance(self.masker, "pandas.core.frame.DataFrame"):
            self.masker = self.masker.values
        
        # define our sort order
        if self.sort_order == "positive":
            self.sort_order_map = lambda x: np.argsort(-x)
        elif self.sort_order == "negative":
            self.sort_order_map = lambda x: np.argsort(x)
        elif self.sort_order == "absolute":
            self.sort_order_map = lambda x: np.argsort(-abs(x))
        else:
            raise ValueError("sort_order must be either \"positive\", \"negative\", or \"absolute\"!")
            
        self.score_values = []
        self.score_aucs = []
        self.labels = []

    def score(self, explainer, X, percent=0.01, y=None, label=None, silent=False, debug_mode=False):
        # if explainer is already the attributions 
        if safe_isinstance(explainer, "numpy.ndarray"):
            attributions = explainer
        elif isinstance(explainer, Explanation):
            attributions = explainer.values
        else:
            attributions = explainer(X).values

        # If the user just gave a dataset as the masker
        # then we make a masker that perturbs features independently
        if safe_isinstance(self.masker, "numpy.ndarray") and self.masker.ndim <= 2 and (self.data_type == "tabular" or self.data_type == None):
            self.masker_data = self.masker
            self.masker = lambda mask, x: x * mask + self.masker_data * np.invert(mask)
            self.data_type = "tabular"
        elif safe_isinstance(self.masker, ["transformers.PreTrainedTokenizer", "transformers.tokenization_utils_base.PreTrainedTokenizerBase"]) and (self.data_type == "text" or self.data_type == None):
            self.masker_data = self.masker
            self.masker = Text(self.masker_data)
            self.data_type = "text"
        elif safe_isinstance(self.masker, "numpy.ndarray") and self.masker.ndim > 2 and (self.data_type == "image" or self.data_type == None):
            self.masker = Image("inpaint_telea", X[0].shape)
            self.data_type = "image"

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
            
            # kwargs is function or parameter 
            if callable(self.kwargs): 
                kwargs = self.kwargs(X[i])
            else:
                kwargs = self.kwargs 

            if len(attributions[i].shape) == 1 or self.data_type == "tabular": 
                output_size = 1 
            else: 
                output_size = attributions[i].shape[-1] 
            
            for k in range(output_size): 
                if self.data_type == "image":
                    mask_shape = X[i].shape 
                else: 
                    mask_shape = feature_size 

                mask = np.ones(mask_shape, dtype=np.bool) * (self.perturbation == "remove")
                masks = [mask.copy()]

                values = np.zeros(feature_size+1)
                masked = self.masker(mask, X[i])

                if kwargs != None: 
                    curr_val = self.f(masked, sentence_ids=kwargs, index=k).mean(0)
                else: 
                    curr_val = self.f(masked, index=k).mean(0)
                
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
                    masked = self.masker(mask, X[i])
                    masks.append(mask.copy())
                
                    if kwargs != None: 
                        curr_val = self.f(masked, sentence_ids=kwargs, index=k).mean(0) 
                    else:
                        curr_val = self.f(masked, index=k).mean(0) 
                    
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

        auc = sklearn.metrics.auc(np.linspace(0, 1, len(ys)), curve_sign*(ys-ys[0]))
        
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
        if (self.sort_order == "negative") != (self.perturbation == "remove"):
            pl.gca().invert_yaxis()
        pl.legend()
        pl.show()