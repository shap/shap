import pandas as pd

def to_dataframe(X, shap_values, type="global"):
    """ Convert the shap values to a pandas DataFrame.

    Parameters
    ----------

    X : numpy.array or pandas.DataFrame
        A matrix of samples (# samples x # features).

    shap_values : numpy.array or list
        A matrix of shap values (# samples x # features).

    type : str
        "global" : (default) generate one dataframe for all observations.
        "local" : generates a list of dataframes.


    Returns
    -------
    Pandas DataFrame or list of Pandas DataFrame
        If type == "global", returns a DataFrame (# samples x # shap values) containing
        the shap values for each feature.
        If type == "local", returns a list with a DataFrame (# features x # shap values x # values) 
        for each sample.
    """
    if type == "global":
        return pd.DataFrame(shap_values, columns=X.columns)
    else:
        samples = []
        for i, sv in enumerate(shap_values):
            df = pd.DataFrame()
            samples.append(pd.DataFrame({
                'feature_name': X.columns,
                'shap_value': sv,
                'feature_value': X.iloc[i, :].values
            }))
        return samples