from ..explainer import Explainer
import numpy as np
import lime
import lime.lime_tabular

import numpy as np
class LimeTabularExplainer(Explainer):
    """ Simply wrap of lime.lime_tabular.LimeTabularExplainer into the common shap interface.
    """

    def __init__(self, model, data):
        self.model = model
        
        if str(type(data)).endswith("pandas.core.frame.DataFrame'>"):
            data = data.values
        self.data = data
        self.explainer = lime.lime_tabular.LimeTabularExplainer(data)
        
        out = self.model(data[0:1])
        if len(out.shape) == 1:
            self.out_dim = 1
            self.flat_out = True
            def pred(X): # assume that 1d outputs are probabilities
                preds = self.model(X).reshape(-1, 1)
                p0 = 1 - preds
                return np.hstack((p0, preds))
            self.model = pred 
        else:
            self.out_dim = self.model(data[0:1]).shape[1]
            self.flat_out = False

    def attributions(self, X, num_features=None):
        num_features = X.shape[1] if num_features is None else num_features
        
        if str(type(X)).endswith("pandas.core.frame.DataFrame'>"):
            X = X.values
            
        out = [np.zeros(X.shape) for j in range(self.out_dim)]
        for i in range(X.shape[0]):
            exp = self.explainer.explain_instance(X[i], self.model, labels=range(self.out_dim), num_features=num_features)
            for j in exp.local_exp:
                for k,v in exp.local_exp[j]:
                    out[j][i,k] = v
        return out
