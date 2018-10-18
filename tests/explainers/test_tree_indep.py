import numpy as np
import shap

def test_single_tree_compare_with_kernel_shap():
    """ Compare with Kernel SHAP, which makes the same independence assumptions
    as Independent Tree SHAP.  Namely, they both assume independence between the 
    set being conditioned on, and the remainder set.
    """
    import xgboost as xgb
    np.random.seed(10)

    n = 1000
    X = np.random.normal(size=(n,7))
    b = np.array([-2,1,3,5,2,20,-5])
    y = np.matmul(X,b)
    max_depth = 6

    # train a model with single tree
    Xd = xgb.DMatrix(X, label=y)
    model = xgb.train({'eta':1, 
                       'max_depth':max_depth, 
                       'base_score': 0, 
                       "lambda": 0}, 
                      Xd, 1)
    ypred = model.predict(Xd)

    # Compare for five random samples
    for i in range(5):
        x_ind = np.random.choice(X.shape[1]); x = X[x_ind,:]

        expl = shap.TreeExplainer(model, direct=True, ref_X=X)
        f = lambda inp : model.predict(xgb.DMatrix(inp))
        expl_kern = shap.KernelExplainer(f, X)

        itshap = expl.independent_treeshap(x).mean(0)
        kshap = expl_kern.shap_values(x, nsamples=150)
        assert np.allclose(itshap,kshap), \
        "Kernel SHAP doesn't match Independent Tree SHAP!"
        assert np.allclose(itshap.sum() + expl.expected_value, ypred[x_ind]), \
        "SHAP values don't sum to model output!"   
        
def test_several_trees():
    """ Make sure Independent Tree SHAP sums up to the correct value for
    larger models (20 trees).
    """
    import xgboost as xgb
    np.random.seed(10)

    n = 1000
    X = np.random.normal(size=(n,7))
    b = np.array([-2,1,3,5,2,20,-5])
    y = np.matmul(X,b)
    max_depth = 6

    # train a model with single tree
    Xd = xgb.DMatrix(X, label=y)
    model = xgb.train({'eta':1, 
                       'max_depth':max_depth, 
                       'base_score': 0, 
                       "lambda": 0}, 
                      Xd, 20)
    ypred = model.predict(Xd)

    # Compare for five random samples
    for i in range(5):
        x_ind = np.random.choice(X.shape[1]); x = X[x_ind,:]

        expl = shap.TreeExplainer(model, direct=True, ref_X=X)

        itshap = expl.independent_treeshap(x).mean(0)
        assert np.allclose(itshap.sum() + expl.expected_value, ypred[x_ind]), \
        "SHAP values don't sum to model output!"
        
def test_single_tree_nonlinear_transformations():
    """ Make sure Independent Tree SHAP single trees with non-linear
    transformations.
    """
    # Supported non-linear transforms
    def sigmoid(x):
        return(1/(1+np.exp(-x)))

    def log_loss(yt,yp):
        return(-(yt*np.log(yp) + (1 - yt)*np.log(1 - yp)))

    def mse(yt,yp):
        return(np.square(yt-yp))

    import xgboost as xgb
    np.random.seed(10)

    n = 1000
    X = np.random.normal(size=(n,7))
    b = np.array([-2,1,3,5,2,20,-5])
    y = np.matmul(X,b)
    y = y + abs(min(y))
    y = np.random.binomial(n=1,p=y/max(y))
    max_depth = 6

    # train a model with single tree
    Xd = xgb.DMatrix(X, label=y)
    model = xgb.train({'eta':1, 
                       'max_depth':max_depth, 
                       'base_score': y.mean(), 
                       "lambda": 0,
                       "objective": "binary:logistic"}, 
                      Xd, 1)
    pred = model.predict(Xd,output_margin=True) # In margin space (log odds)
    trans_pred = model.predict(Xd) # In probability space

    expl = shap.TreeExplainer(model, direct=True, ref_X=X)
    f = lambda inp : model.predict(xgb.DMatrix(inp), output_margin=True)
    expl_kern = shap.KernelExplainer(f, X)

    x_ind = 0; x = X[x_ind,:]
    perref_itshap_margin = expl.independent_treeshap(x)
    itshap = perref_itshap_margin.mean(0)
    kshap = expl_kern.shap_values(x, nsamples=300)
    assert np.allclose(itshap.sum() + expl.expected_value, pred[x_ind]), \
    "SHAP values don't sum to model output on explaining margin!"
    assert np.allclose(itshap, kshap), \
    "Independent Tree SHAP doesn't match Kernel SHAP on explaining margin!"

    expl = shap.TreeExplainer(model, direct=True, output_type="logistic", ref_X=X)
    itshap = expl.independent_treeshap(x).mean(0)
    assert np.allclose(itshap.sum() + expl.expected_value, trans_pred[x_ind]), \
    "SHAP values don't sum to model output on explaining logistic!"

    x_ind = 0; x = X[x_ind,:]
    expl = shap.TreeExplainer(model, direct=True, output_type="logloss", ref_X=X, ref_y=y)
    perref_itshap_logloss2 = expl.independent_treeshap(x,y[x_ind])
    itshap = expl.independent_treeshap(x,y[x_ind]).mean(0)
    margin_pred = model.predict(xgb.DMatrix(x[np.newaxis,:]),output_margin=True)
    currpred = log_loss(y[x_ind],sigmoid(margin_pred))
    assert np.allclose(itshap.sum(), currpred - expl.expected_value), \
    "SHAP values don't sum to model output on explaining logloss!"

    x_ind = 0; x = X[x_ind,:]
    expl = shap.TreeExplainer(model, direct=True, output_type="mse", ref_X=X, ref_y=y)
    perref_itshap_logloss2 = expl.independent_treeshap(x,y[x_ind])
    itshap = expl.independent_treeshap(x,y[x_ind]).mean(0)
    margin_pred = model.predict(xgb.DMatrix(x[np.newaxis,:]),output_margin=True)
    currpred = mse(y[x_ind],margin_pred)
    assert np.allclose(itshap.sum(), currpred - expl.expected_value), \
    "SHAP values don't sum to model output on explaining mse!"