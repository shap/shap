import pandas as pd
import numpy as np
import sklearn.datasets
import os
try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

def boston(display=False):
    """ Return the boston housing data in a nice package. """

    d = sklearn.datasets.load_boston()
    df = pd.DataFrame(data=d.data, columns=d.feature_names)
    return df,d.target

def iris(display=False):
    """ Return the classic iris data in a nice package. """

    d = sklearn.datasets.load_iris()
    df = pd.DataFrame(data=d.data, columns=d.feature_names)
    if display:
        return df,[d.target_names[v] for v in d.target]
    else:
        return df,d.target

def adult(display=False):
    """ Return the Adult census data in a nice package. """
    dtypes = [
        ("Age", "float32"), ("Workclass", "category"), ("fnlwgt", "float32"),
        ("Education", "category"), ("Education-Num", "float32"), ("Marital Status", "category"),
        ("Occupation", "category"), ("Relationship", "category"), ("Race", "category"),
        ("Sex", "category"), ("Capital Gain", "float32"), ("Capital Loss", "float32"),
        ("Hours per week", "float32"), ("Country", "category"), ("Target", "category")
    ]
    raw_data = pd.read_csv(
        cache("https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data"),
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

    if display:
        return raw_data.drop(["Education", "Target"], axis=1),data["Target"].as_matrix()
    else:
        return data.drop(["Target"], axis=1),data["Target"].as_matrix()

def nhanesi(display=False):
    X = pd.read_csv(cache("https://github.com/slundberg/shap/raw/master/notebooks/data/NHANESI_subset_X.csv"))
    y = pd.read_csv(cache("https://github.com/slundberg/shap/raw/master/notebooks/data/NHANESI_subset_y.csv"))["y"]
    X_display = X.copy()
    X_display["Sex"] = ["Male" if v == 1 else "Female" for v in X["Sex"]]
    if display:
        return X_display,np.array(y)
    else:
        return X,np.array(y)

def cache(url, file_name=None):
    if file_name is None:
        file_name = os.path.basename(url)
    data_dir = os.path.join(os.path.dirname(__file__), "cached_data")
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)

    file_path = os.path.join(data_dir, file_name)
    if not os.path.isfile(file_path):
        urlretrieve(url, os.path.join(data_dir, file_name))

    return file_path
