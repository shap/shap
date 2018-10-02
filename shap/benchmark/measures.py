import numpy as np
from tqdm import tqdm
import gc


_remove_cache = {}
def remove(nmask, X_train, y_train, X_test, y_test, attr_test, model_generator, metric, trained_model):
    """ The model is retrained for each test sample with the important features set to a constant.

    If you want to know how important a set of features is you can ask how the model would be
    different if those features had never existed. To determine this we can mask those features
    across the entire training and test datasets, then retrain the model. If we apply compare the
    output of this retrained model to the original model we can see the effect produced by knowning
    the features we masked. Since for individualized explanation methods each test sample has a
    different set of most important features we need to retrain the model for every test sample
    to get the change in model performance when a specified fraction of the most important features
    are withheld.
    """

    # see if we match the last cached call
    global _remove_cache
    args = (X_train, y_train, X_test, y_test, model_generator, metric)
    cache_match = False
    if "args" in _remove_cache:
        if all(a is b for a,b in zip(_remove_cache["args"], args)) and np.all(_remove_cache["attr_test"] == attr_test):
            cache_match = True

    X_train, X_test = to_array(X_train, X_test)

    # how many features to mask
    assert X_train.shape[1] == X_test.shape[1]

    # this is the model we will retrain many times
    model_masked = model_generator()

    # mask nmask top features and re-train the model for each test explanation
    X_train_tmp = np.zeros(X_train.shape)
    X_test_tmp = np.zeros(X_test.shape)
    yp_masked_test = np.zeros(y_test.shape)
    tie_breaking_noise = const_rand(X_train.shape[1]) * 1e-6
    last_nmask = _remove_cache.get("nmask", None)
    last_yp_masked_test = _remove_cache.get("yp_masked_test", None)
    for i in tqdm(range(len(y_test)), "Retraining for the 'remove' metric"):
        if cache_match and last_nmask[i] == nmask[i]:
            yp_masked_test[i] = last_yp_masked_test[i]
        elif nmask[i] == 0:
            yp_masked_test[i] = get_predictions(trained_model, X_test[i:i+1])[0]
        else:
            # mask out the most important features for this test instance
            X_train_tmp[:] = X_train
            X_test_tmp[:] = X_test
            ordering = np.argsort(-attr_test[i,:] + tie_breaking_noise)
            X_train_tmp[:,ordering[:nmask[i]]] = X_train[:,ordering[:nmask[i]]].mean()
            X_test_tmp[i,ordering[:nmask[i]]] = X_train[:,ordering[:nmask[i]]].mean()

            # retrain the model and make a prediction
            model_masked.fit(X_train_tmp, y_train)
            yp_masked_test[i] = get_predictions(model_masked, X_test_tmp[i:i+1])[0]

    # save our results so the next call to us can be faster when there is redundancy
    _remove_cache["nmask"] = nmask
    _remove_cache["yp_masked_test"] = yp_masked_test
    _remove_cache["attr_test"] = attr_test
    _remove_cache["args"] = args

    return metric(y_test, yp_masked_test)

def mask_remove(nmask, X_train, y_train, X_test, y_test, attr_test, model_generator, metric, trained_model):
    """ Each test sample is masked by setting the important features to a constant.
    """

    X_train, X_test = to_array(X_train, X_test)

    # how many features to mask
    assert X_train.shape[1] == X_test.shape[1]

    # mask nmask top features for each test explanation
    X_test_tmp = X_test.copy()
    tie_breaking_noise = const_rand(X_train.shape[1]) * 1e-6
    mean_vals = X_train.mean(0)
    for i in range(len(y_test)):
        if nmask[i] > 0:
            ordering = np.argsort(-attr_test[i,:] + tie_breaking_noise)
            X_test_tmp[i,ordering[:nmask[i]]] = mean_vals[ordering[:nmask[i]]]
    
    yp_masked_test = get_predictions(trained_model, X_test_tmp)

    return metric(y_test, yp_masked_test)

def batch_remove(nmask_train, nmask_test, X_train, y_train, X_test, y_test, attr_train, attr_test, model_generator, metric):
    """ An approximation of holdout that only retraines the model once.

    This is alse called ROAR (RemOve And Retrain) in work by Google. It is much more computationally
    efficient that the holdout method because it masks the most important features in every sample
    and then retrains the model once, instead of retraining the model for every test sample like
    the holdout metric.
    """

    X_train, X_test = to_array(X_train, X_test)

    # how many features to mask
    assert X_train.shape[1] == X_test.shape[1]

    # mask nmask top features for each explanation
    X_train_tmp = X_train.copy()
    X_train_mean = X_train.mean(0)
    tie_breaking_noise = const_rand(X_train.shape[1]) * 1e-6
    for i in range(len(y_train)):
        if nmask_train[i] > 0:
            ordering = np.argsort(-attr_train[i, :] + tie_breaking_noise)
            X_train_tmp[i, ordering[:nmask_train[i]]] = X_train_mean[ordering[:nmask_train[i]]]
    X_test_tmp = X_test.copy()
    for i in range(len(y_test)):
        if nmask_test[i] > 0:
            ordering = np.argsort(-attr_test[i, :] + tie_breaking_noise)
            X_test_tmp[i, ordering[:nmask_test[i]]] = X_train_mean[ordering[:nmask_test[i]]]

    # train the model with all the given features masked
    model_masked = model_generator()
    model_masked.fit(X_train_tmp, y_train)
    yp_test_masked = get_predictions(model_masked, X_test_tmp)

    return metric(y_test, yp_test_masked)

_keep_cache = {}
def keep(nkeep, X_train, y_train, X_test, y_test, attr_test, model_generator, metric, trained_model):
    """ The model is retrained for each test sample with the non-important features set to a constant.

    If you want to know how important a set of features is you can ask how the model would be
    different if only those features had existed. To determine this we can mask the other features
    across the entire training and test datasets, then retrain the model. If we apply compare the
    output of this retrained model to the original model we can see the effect produced by only
    knowning the important features. Since for individualized explanation methods each test sample
    has a different set of most important features we need to retrain the model for every test sample
    to get the change in model performance when a specified fraction of the most important features
    are retained.
    """

    # see if we match the last cached call
    global _keep_cache
    args = (X_train, y_train, X_test, y_test, model_generator, metric)
    cache_match = False
    if "args" in _keep_cache:
        if all(a is b for a,b in zip(_keep_cache["args"], args)) and np.all(_keep_cache["attr_test"] == attr_test):
            cache_match = True

    X_train, X_test = to_array(X_train, X_test)

    # how many features to mask
    assert X_train.shape[1] == X_test.shape[1]

    # this is the model we will retrain many times
    model_masked = model_generator()

    # keep nkeep top features and re-train the model for each test explanation
    X_train_tmp = np.zeros(X_train.shape)
    X_test_tmp = np.zeros(X_test.shape)
    yp_masked_test = np.zeros(y_test.shape)
    tie_breaking_noise = const_rand(X_train.shape[1]) * 1e-6
    last_nkeep = _keep_cache.get("nkeep", None)
    last_yp_masked_test = _keep_cache.get("yp_masked_test", None)
    for i in tqdm(range(len(y_test)), "Retraining for the 'keep' metric"):
        if cache_match and last_nkeep[i] == nkeep[i]:
            yp_masked_test[i] = last_yp_masked_test[i]
        elif nkeep[i] == attr_test.shape[1]:
            yp_masked_test[i] = get_predictions(trained_model, X_test[i:i+1])[0]
        else:

            # mask out the most important features for this test instance
            X_train_tmp[:] = X_train
            X_test_tmp[:] = X_test
            ordering = np.argsort(-attr_test[i,:] + tie_breaking_noise)
            X_train_tmp[:,ordering[nkeep[i]:]] = X_train[:,ordering[nkeep[i]:]].mean()
            X_test_tmp[i,ordering[nkeep[i]:]] = X_train[:,ordering[nkeep[i]:]].mean()

            # retrain the model and make a prediction
            model_masked.fit(X_train_tmp, y_train)
            yp_masked_test[i] = get_predictions(model_masked, X_test_tmp[i:i+1])[0]

    # save our results so the next call to us can be faster when there is redundancy
    _keep_cache["nkeep"] = nkeep
    _keep_cache["yp_masked_test"] = yp_masked_test
    _keep_cache["attr_test"] = attr_test
    _keep_cache["args"] = args

    return metric(y_test, yp_masked_test)

def mask_keep(nkeep, X_train, y_train, X_test, y_test, attr_test, model_generator, metric, trained_model):
    """ The model is retrained for each test sample with the non-important features set to a constant.
    """

    X_train, X_test = to_array(X_train, X_test)

    # how many features to mask
    assert X_train.shape[1] == X_test.shape[1]

    # keep nkeep top features for each test explanation
    X_test_tmp = X_test.copy()
    yp_masked_test = np.zeros(y_test.shape)
    tie_breaking_noise = const_rand(X_train.shape[1]) * 1e-6
    mean_vals = X_train.mean(0)
    for i in range(len(y_test)):
        if nkeep[i] < X_test.shape[1]:
            ordering = np.argsort(-attr_test[i,:] + tie_breaking_noise)
            X_test_tmp[i,ordering[nkeep[i]:]] = mean_vals[ordering[nkeep[i]:]]

    yp_masked_test = get_predictions(trained_model, X_test_tmp)

    return metric(y_test, yp_masked_test)

def batch_keep(nkeep_train, nkeep_test, X_train, y_train, X_test, y_test, attr_train, attr_test, model_generator, metric):
    """ An approximation of keep that only retraines the model once.

    This is alse called KAR (Keep And Retrain) in work by Google. It is much more computationally
    efficient that the keep method because it masks the unimportant features in every sample
    and then retrains the model once, instead of retraining the model for every test sample like
    the keep metric.
    """

    X_train, X_test = to_array(X_train, X_test)

    # how many features to mask
    assert X_train.shape[1] == X_test.shape[1]

    # mask nkeep top features for each explanation
    X_train_tmp = X_train.copy()
    X_train_mean = X_train.mean(0)
    tie_breaking_noise = const_rand(X_train.shape[1]) * 1e-6
    for i in range(len(y_train)):
        if nkeep_train[i] < X_train.shape[1]:
            ordering = np.argsort(-attr_train[i, :] + tie_breaking_noise)
            X_train_tmp[i, ordering[nkeep_train[i]:]] = X_train_mean[ordering[nkeep_train[i]:]]
    X_test_tmp = X_test.copy()
    for i in range(len(y_test)):
        if nkeep_test[i] < X_test.shape[1]:
            ordering = np.argsort(-attr_test[i, :] + tie_breaking_noise)
            X_test_tmp[i, ordering[nkeep_test[i]:]] = X_train_mean[ordering[nkeep_test[i]:]]

    # train the model with all the features not given masked
    model_masked = model_generator()
    model_masked.fit(X_train_tmp, y_train)
    yp_test_masked = get_predictions(model_masked, X_test_tmp)

    return metric(y_test, yp_test_masked)

def local_accuracy(X_train, y_train, X_test, y_test, attr_test, model_generator, metric, trained_model):
    """ The how well do the features plus a constant base rate sum up to the model output.
    """

    X_train, X_test = to_array(X_train, X_test)

    # how many features to mask
    assert X_train.shape[1] == X_test.shape[1]

    # keep nkeep top features and re-train the model for each test explanation
    yp_test = get_predictions(trained_model, X_test)

    return metric(yp_test, strip_list(attr_test).sum(1))

def to_array(*args):
    return [a.values if str(type(a)).endswith("'pandas.core.frame.DataFrame'>") else a for a in args]

def const_rand(size, seed=23980):
    """ Generate a random array with a fixed seed.
    """
    old_seed = np.random.seed()
    np.random.seed(seed)
    out = np.random.rand(size)
    np.random.seed(old_seed)
    return out

def strip_list(attrs):
    """ This assumes that if you have a list of outputs you just want the second one (the second class).
    """
    if isinstance(attrs, list):
        return attrs[1]
    else:
        return attrs

def get_predictions(model, X):
    if hasattr(model, "predict_proba"):
        return model.predict_proba(X)[:,1]
        # v = model.predict_log_proba(X)
        # return v[:,1] - v[:,0] # return the log odds (can't because some model have -inf)
    else:
        return model.predict(X)