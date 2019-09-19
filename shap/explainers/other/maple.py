
import shap
from shap.explainers.explainer import Explainer
import numpy as np
from sklearn.model_selection import train_test_split

class MapleExplainer(Explainer):
    """ Simply wraps MAPLE into the common SHAP interface.

    Parameters
    ----------
    model : function
        User supplied function that takes a matrix of samples (# samples x # features) and
        computes a the output of the model for those samples. The output can be a vector
        (# samples) or a matrix (# samples x # model outputs).

    data : numpy.array
        The background dataset.
    """

    def __init__(self, model, data):
        self.model = model

        if str(type(data)).endswith("pandas.core.frame.DataFrame'>"):
            data = data.values
        self.data = data
        self.data_mean = self.data.mean(0)
        
        out = self.model(data)
        if len(out.shape) == 1:
            self.out_dim = 1
            self.flat_out = True
        else:
            self.out_dim = out.shape[1]
            self.flat_out = False
            
        X_train, X_valid, y_train, y_valid = train_test_split(data, out, test_size=0.2, random_state=0)
        self.explainer = MAPLE(X_train, y_train, X_valid, y_valid)

    def attributions(self, X, multiply_by_input=False):
        """ Compute the MAPLE coef attributions.

        Parameters
        ----------
        multiply_by_input : bool
            If true, this multiplies the learned coeffients by the mean-centered input. This makes these
            values roughly comparable to SHAP values.
        """
        if str(type(X)).endswith("pandas.core.frame.DataFrame'>"):
            X = X.values
            
        out = [np.zeros(X.shape) for j in range(self.out_dim)]
        for i in range(X.shape[0]):
            exp = self.explainer.explain(X[i])["coefs"]
            out[0][i,:] = exp[1:] 
            if multiply_by_input:
                out[0][i,:] = out[0][i,:] * (X[i] - self.data_mean)
        
        return out[0] if self.flat_out else out


class TreeMapleExplainer(Explainer):
    """ Simply tree MAPLE into the common SHAP interface.

    Parameters
    ----------
    model : function
        User supplied function that takes a matrix of samples (# samples x # features) and
        computes a the output of the model for those samples. The output can be a vector
        (# samples) or a matrix (# samples x # model outputs).

    data : numpy.array
        The background dataset.
    """

    def __init__(self, model, data):
        self.model = model

        if str(type(model)).endswith("sklearn.ensemble.gradient_boosting.GradientBoostingRegressor'>"):
            fe_type = "gbdt"
        # elif str(type(model)).endswith("sklearn.tree.tree.DecisionTreeClassifier'>"):
            # pass
        elif str(type(model)).endswith("sklearn.ensemble.forest.RandomForestRegressor'>"):
            fe_type = "rf"
        # elif str(type(model)).endswith("sklearn.ensemble.forest.RandomForestClassifier'>"):
        #     pass
        # elif str(type(model)).endswith("xgboost.sklearn.XGBRegressor'>"):
        #     pass
        # elif str(type(model)).endswith("xgboost.sklearn.XGBClassifier'>"):
        #     pass
        else:
            raise Exception("The passed model is not yet supported by TreeMapleExplainer: " + str(type(model)))

        if str(type(data)).endswith("pandas.core.frame.DataFrame'>"):
            data = data.values
        self.data = data
        self.data_mean = self.data.mean(0)
        
        out = self.model.predict(data[0:1])
        if len(out.shape) == 1:
            self.out_dim = 1
            self.flat_out = True
        else:
            self.out_dim = self.model.predict(data[0:1]).shape[1]
            self.flat_out = False
            
        #_, X_valid, _, y_valid = train_test_split(data, self.model.predict(data), test_size=0.2, random_state=0)
        preds = self.model.predict(data)
        self.explainer = MAPLE(data, preds, data, preds, fe=self.model, fe_type=fe_type)

    def attributions(self, X, multiply_by_input=False):
        """ Compute the MAPLE coef attributions.

        Parameters
        ----------
        multiply_by_input : bool
            If true, this multiplies the learned coeffients by the mean-centered input. This makes these
            values roughly comparable to SHAP values.
        """
        if str(type(X)).endswith("pandas.core.frame.DataFrame'>"):
            X = X.values
            
        out = [np.zeros(X.shape) for j in range(self.out_dim)]
        for i in range(X.shape[0]):
            exp = self.explainer.explain(X[i])["coefs"]
            out[0][i,:] = exp[1:] 
            if multiply_by_input:
                out[0][i,:] = out[0][i,:] * (X[i] - self.data_mean)
        
        return out[0] if self.flat_out else out


#################################################
# The code below was authored by Gregory Plumb and is
# from: https://github.com/GDPlumb/MAPLE/blob/master/Code/MAPLE.py
# It has by copied here to allow for benchmark comparisions. Please see
# the original repo for the latest version, supporting material, and citations.
#################################################

# Notes:
# -  Assumes any required data normalization has already been done
# -  Can pass Y (desired response) instead of MR (model fit to Y) to make fitting MAPLE to datasets easy

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import numpy as np

class MAPLE:
 
    def __init__(self, X_train, MR_train, X_val, MR_val, fe_type = "rf", fe=None, n_estimators = 200, max_features = 0.5, min_samples_leaf = 10, regularization = 0.001):
        
        # Features and the target model response
        self.X_train = X_train
        self.MR_train = MR_train
        self.X_val = X_val
        self.MR_val = MR_val
        
        # Forest Ensemble Parameters
        self.n_estimators = n_estimators
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        
        # Local Linear Model Parameters
        self.regularization = regularization
        
        # Data parameters
        num_features = X_train.shape[1]
        self.num_features = num_features
        num_train = X_train.shape[0]
        self.num_train = num_train
        num_val = X_val.shape[0]
        
        # Fit a Forest Ensemble to the model response
        if fe is None:
            if fe_type == "rf":
                fe = RandomForestRegressor(n_estimators = n_estimators, min_samples_leaf = min_samples_leaf, max_features = max_features)
            elif fe_type == "gbrt":
                fe = GradientBoostingRegressor(n_estimators = n_estimators, min_samples_leaf = min_samples_leaf, max_features = max_features, max_depth = None)
            else:
                print("Unknown FE type ", fe)
                import sys
                sys.exit(0)
            fe.fit(X_train, MR_train)
        else:
            self.n_estimators = n_estimators = len(fe.estimators_)
        self.fe = fe
        
        train_leaf_ids = fe.apply(X_train)
        self.train_leaf_ids = train_leaf_ids
        
        val_leaf_ids_list = fe.apply(X_val)
        
        # Compute the feature importances: Non-normalized @ Root
        scores = np.zeros(num_features)
        if fe_type == "rf":
            for i in range(n_estimators):
                splits = fe[i].tree_.feature #-2 indicates leaf, index 0 is root
                if splits[0] != -2:
                    scores[splits[0]] += fe[i].tree_.impurity[0] #impurity reduction not normalized per tree
        elif fe_type == "gbrt":
            for i in range(n_estimators):
                splits = fe[i, 0].tree_.feature #-2 indicates leaf, index 0 is root
                if splits[0] != -2:
                    scores[splits[0]] += fe[i, 0].tree_.impurity[0] #impurity reduction not normalized per tree
        self.feature_scores = scores
        mostImpFeats = np.argsort(-scores)
                
        # Find the number of features to use for MAPLE
        retain_best = 0
        rmse_best = np.inf
        for retain in range(1, num_features + 1):
            
            # Drop less important features for local regression
            X_train_p = np.delete(X_train, mostImpFeats[retain:], axis = 1)
            X_val_p = np.delete(X_val, mostImpFeats[retain:], axis = 1)
                        
            lr_predictions = np.empty([num_val], dtype=float)
            
            for i in range(num_val):
                
                weights = self.training_point_weights(val_leaf_ids_list[i])
                    
                # Local linear model
                lr_model = Ridge(alpha=regularization)
                lr_model.fit(X_train_p, MR_train, weights)
                lr_predictions[i] = lr_model.predict(X_val_p[i].reshape(1, -1))
            
            rmse_curr = np.sqrt(mean_squared_error(lr_predictions, MR_val))
            
            if rmse_curr < rmse_best:
                rmse_best = rmse_curr
                retain_best = retain
                
        self.retain = retain_best
        self.X = np.delete(X_train, mostImpFeats[retain_best:], axis = 1)
                
    def training_point_weights(self, instance_leaf_ids):
        weights = np.zeros(self.num_train)
        for i in range(self.n_estimators):
            # Get the PNNs for each tree (ones with the same leaf_id)
            PNNs_Leaf_Node = np.where(self.train_leaf_ids[:, i] == instance_leaf_ids[i])[0]
            if len(PNNs_Leaf_Node) > 0: # SML: added this to fix degenerate cases
                weights[PNNs_Leaf_Node] += 1.0 / len(PNNs_Leaf_Node)
        return weights
        
    def explain(self, x):
        
        x = x.reshape(1, -1)
        
        mostImpFeats = np.argsort(-self.feature_scores)
        x_p = np.delete(x, mostImpFeats[self.retain:], axis = 1)
        
        curr_leaf_ids = self.fe.apply(x)[0]
        weights = self.training_point_weights(curr_leaf_ids)
           
        # Local linear model
        lr_model = Ridge(alpha = self.regularization)
        lr_model.fit(self.X, self.MR_train, weights)

        # Get the model coeficients
        coefs = np.zeros(self.num_features + 1)
        coefs[0] = lr_model.intercept_
        coefs[np.sort(mostImpFeats[0:self.retain]) + 1] = lr_model.coef_
        
        # Get the prediction at this point
        prediction = lr_model.predict(x_p.reshape(1, -1))
        
        out = {}
        out["weights"] = weights
        out["coefs"] = coefs
        out["pred"] = prediction
        
        return out

    def predict(self, X):
        n = X.shape[0]
        pred = np.zeros(n)
        for i in range(n):
            exp = self.explain(X[i, :])
            pred[i] = exp["pred"][0]
        return pred

    # Make the predictions based on the forest ensemble (either random forest or gradient boosted regression tree) instead of MAPLE
    def predict_fe(self, X):
        return self.fe.predict(X)

    # Make the predictions based on SILO (no feature selection) instead of MAPLE
    def predict_silo(self, X):
        n = X.shape[0]
        pred = np.zeros(n)
        for i in range(n): #The contents of this inner loop are similar to explain(): doesn't use the features selected by MAPLE or return as much information
            x = X[i, :].reshape(1, -1)
        
            curr_leaf_ids = self.fe.apply(x)[0]
            weights = self.training_point_weights(curr_leaf_ids)
                    
            # Local linear model
            lr_model = Ridge(alpha = self.regularization)
            lr_model.fit(self.X_train, self.MR_train, weights)
                
            pred[i] = lr_model.predict(x)[0]
        
        return pred


