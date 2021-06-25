import pandas as pd
import numpy as np
import sklearn.datasets
import os

try:
    from urllib.request import urlretrieve
except ImportError:
    from urllib import urlretrieve

github_data_url = "https://github.com/slundberg/shap/raw/master/data/"

def imagenet50(display=False, resolution=224):
    """ This is a set of 50 images representative of ImageNet images.

    This dataset was collected by randomly finding a working ImageNet link and then pasting the
    original ImageNet image into Google image search restricted to images licensed for reuse. A
    similar image (now with rights to reuse) was downloaded as a rough replacment for the original
    ImageNet image. The point is to have a random sample of ImageNet for use as a background
    distribution for explaining models trained on ImageNet data.

    Note that because the images are only rough replacements the labels might no longer be correct.
    """

    prefix = github_data_url + "imagenet50_"
    X = np.load(cache(prefix + "%sx%s.npy" % (resolution, resolution))).astype(np.float32)
    y = np.loadtxt(cache(prefix + "labels.csv"))
    return X, y

def boston(display=False):
    """ Return the boston housing data in a nice package. """

    d = sklearn.datasets.load_boston()
    df = pd.DataFrame(data=d.data, columns=d.feature_names) # pylint: disable=E1101
    return df, d.target # pylint: disable=E1101


def linnerud(display=False):
    """ Return the linnerud data in a nice package (multi-target regression). """

    d = sklearn.datasets.load_linnerud()
    X = pd.DataFrame(d.data, columns=d.feature_names) # pylint: disable=E1101
    y = pd.DataFrame(d.target, columns=d.target_names) # pylint: disable=E1101
    return X, y # pylint: disable=E1101


def imdb(display=False):
    """ Return the clssic IMDB sentiment analysis training data in a nice package.

    Full data is at: http://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz
    Paper to cite when using the data is: http://www.aclweb.org/anthology/P11-1015
    """

    with open(cache(github_data_url + "imdb_train.txt"), encoding="utf-8") as f:
        data = f.readlines()
    y = np.ones(25000, dtype=np.bool)
    y[:12500] = 0
    return data, y

def communitiesandcrime(display=False):
    """ Predict total number of non-violent crimes per 100K popuation.

    This dataset is from the classic UCI Machine Learning repository:
    https://archive.ics.uci.edu/ml/datasets/Communities+and+Crime+Unnormalized
    """

    raw_data = pd.read_csv(
        cache(github_data_url + "CommViolPredUnnormalizedData.txt"),
        na_values="?"
    )

    # find the indices where the total violent crimes are known
    valid_inds = np.where(np.invert(np.isnan(raw_data.iloc[:,-2])))[0]
    y = np.array(raw_data.iloc[valid_inds,-2], dtype=np.float)

    # extract the predictive features and remove columns with missing values
    X = raw_data.iloc[valid_inds,5:-18]
    valid_cols = np.where(np.isnan(X.values).sum(0) == 0)[0]
    X = X.iloc[:,valid_cols]

    return X, y

def diabetes(display=False):
    """ Return the diabetes data in a nice package. """

    d = sklearn.datasets.load_diabetes()
    df = pd.DataFrame(data=d.data, columns=d.feature_names) # pylint: disable=E1101
    return df, d.target # pylint: disable=E1101


def iris(display=False):
    """ Return the classic iris data in a nice package. """

    d = sklearn.datasets.load_iris()
    df = pd.DataFrame(data=d.data, columns=d.feature_names) # pylint: disable=E1101
    if display:
        return df, [d.target_names[v] for v in d.target] # pylint: disable=E1101
    else:
        return df, d.target # pylint: disable=E1101


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
        cache(github_data_url + "adult.data"),
        names=[d[0] for d in dtypes],
        na_values="?",
        dtype=dict(dtypes)
    )
    data = raw_data.drop(["Education"], axis=1)  # redundant with Education-Num
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
    for k, dtype in filt_dtypes:
        if dtype == "category":
            if k == "Relationship":
                data[k] = np.array([rcode[v.strip()] for v in data[k]])
            else:
                data[k] = data[k].cat.codes

    if display:
        return raw_data.drop(["Education", "Target", "fnlwgt"], axis=1), data["Target"].values
    else:
        return data.drop(["Target", "fnlwgt"], axis=1), data["Target"].values


def nhanesi(display=False):
    """ A nicely packaged version of NHANES I data with surivival times as labels.
    """
    X = pd.read_csv(cache(github_data_url + "NHANESI_X.csv"), index_col=0)
    y = pd.read_csv(cache(github_data_url + "NHANESI_y.csv"), index_col=0)["y"]
    if display:
        X_display = X.copy()
        #X_display["sex_isFemale"] = ["Female" if v else "Male" for v in X["sex_isFemale"]]
        return X_display, np.array(y)
    else:
        return X, np.array(y)


def corrgroups60(display=False):
    """ Correlated Groups 60
    
    A simulated dataset with tight correlations among distinct groups of features.
    """

    # set a constant seed
    old_seed = np.random.seed()
    np.random.seed(0)

    # generate dataset with known correlation
    N = 1000
    M = 60

    # set one coefficent from each group of 3 to 1
    beta = np.zeros(M)
    beta[0:30:3] = 1

    # build a correlation matrix with groups of 3 tightly correlated features
    C = np.eye(M)
    for i in range(0,30,3):
        C[i,i+1] = C[i+1,i] = 0.99
        C[i,i+2] = C[i+2,i] = 0.99
        C[i+1,i+2] = C[i+2,i+1] = 0.99
    f = lambda X: np.matmul(X, beta)

    # Make sure the sample correlation is a perfect match
    X_start = np.random.randn(N, M)
    X_centered = X_start - X_start.mean(0)
    Sigma = np.matmul(X_centered.T, X_centered) / X_centered.shape[0]
    W = np.linalg.cholesky(np.linalg.inv(Sigma)).T
    X_white = np.matmul(X_centered, W.T)
    assert np.linalg.norm(np.corrcoef(np.matmul(X_centered, W.T).T) - np.eye(M)) < 1e-6 # ensure this decorrelates the data

    # create the final data
    X_final = np.matmul(X_white, np.linalg.cholesky(C).T)
    X = X_final
    y = f(X) + np.random.randn(N) * 1e-2

    # restore the previous numpy random seed
    np.random.seed(old_seed)

    return pd.DataFrame(X), y


def independentlinear60(display=False):
    """ A simulated dataset with tight correlations among distinct groups of features.
    """

    # set a constant seed
    old_seed = np.random.seed()
    np.random.seed(0)

    # generate dataset with known correlation
    N = 1000
    M = 60

    # set one coefficent from each group of 3 to 1
    beta = np.zeros(M)
    beta[0:30:3] = 1
    f = lambda X: np.matmul(X, beta)

    # Make sure the sample correlation is a perfect match
    X_start = np.random.randn(N, M)
    X = X_start - X_start.mean(0)
    y = f(X) + np.random.randn(N) * 1e-2

    # restore the previous numpy random seed
    np.random.seed(old_seed)

    return pd.DataFrame(X), y


def a1a():
    """ A sparse dataset in scipy csr matrix format.
    """
    return sklearn.datasets.load_svmlight_file(cache(github_data_url + 'a1a.svmlight'))


def rank():
    """ Ranking datasets from lightgbm repository.
    """
    rank_data_url = 'https://raw.githubusercontent.com/Microsoft/LightGBM/master/examples/lambdarank/'
    x_train, y_train = sklearn.datasets.load_svmlight_file(cache(rank_data_url + 'rank.train'))
    x_test, y_test = sklearn.datasets.load_svmlight_file(cache(rank_data_url + 'rank.test'))
    q_train = np.loadtxt(cache(rank_data_url + 'rank.train.query'))
    q_test = np.loadtxt(cache(rank_data_url + 'rank.test.query'))
    return x_train, y_train, x_test, y_test, q_train, q_test


def cache(url, file_name=None):
    if file_name is None:
        file_name = os.path.basename(url)
    data_dir = os.path.join(os.path.dirname(__file__), "cached_data")
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
    
    file_path = os.path.join(data_dir, file_name)
    if not os.path.isfile(file_path):
        urlretrieve(url, file_path)

    return file_path
