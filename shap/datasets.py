import pandas as pd
import numpy as np
import sklearn.datasets

def boston():
    """ Return the boston housing data in a nice package. """

    d = sklearn.datasets.load_boston()
    df = pd.DataFrame(data=d.data, columns=d.feature_names)
    return df,d.target,df

def adult():
    """ Return the Adult census data in a nice package. """
    dtypes = [
        ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
        ("Education", "category"), ("Education-Num", "float32"), ("Marital Status", "category"),
        ("Occupation", "category"), ("Relationship", "category"), ("Race", "category"),
        ("Sex", "category"), ("Capital Gain", "float32"), ("Capital Loss", "float32"),
        ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")
    ]
    raw_data = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        names=[d[0] for d in dtypes],
        na_values="?",
        dtype=dict(dtypes)
    )
    data = raw_data.drop(["Education"], axis=1) # redundant with Education-Num
    filt_dtypes = list(filter(lambda x: not (x[0] in ["Target", "Education"]), dtypes))
    data["Target"] = data["Target"] == " >50K"
    rcode = {
        "Not-in-family": 0,
        "Unmarried": 1,
        "Other-relative": 2,
        "Own-child": 3,
        "Husband": 4,
        "Wife": 5
    }
    for k,dtype in filt_dtypes:
        if dtype == "category":
            if k == "Relationship":
                data[k] = np.array([rcode[v.strip()] for v in data[k]])
            else:
                data[k] = data[k].cat.codes

    return data.drop(["Target"], axis=1),data["Target"],raw_data.drop(["Education", "Target"], axis=1)

def nhanesi():
    X = pd.read_csv("https://github.com/slundberg/shap/raw/master/notebooks/data/NHANESI_subset_X.csv")
    y = pd.read_csv("https://github.com/slundberg/shap/raw/master/notebooks/data/NHANESI_subset_y.csv")["y"]
    X_display = X.copy()
    X_display["Sex"] = ["Male" if v == 1 else "Female" for v in X["Sex"]]
    return X,np.array(y),X_display
