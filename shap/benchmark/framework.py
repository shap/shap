import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 
import scipy as sp 
import sklearn 
from sklearn.model_selection import train_test_split
import xgboost
import torch 
import transformers
import time
import shap 
from shap.utils import safe_isinstance, MaskedModel
from perturbation import SequentialPerturbation

def update(model, X, y, explainer, masker, sort_order, score_function, perturbation, results, score=""):
    metric = perturbation + ' ' + sort_order + ' ' + score
    sp = SequentialPerturbation(model, masker, sort_order, score_function, perturbation)
    x, y = sp.score(explainer, X, y=y)
    results['metrics'].append(metric)
    results['output'][metric] = [x, y, sklearn.metrics.auc(x, y)] 

def benchmark(model, X, y, explainer, masker, metrics, *args):
    # convert dataframes
    if safe_isinstance(X, "pandas.core.series.Series") or safe_isinstance(X, "pandas.core.frame.DataFrame"):
        X = X.values
    if safe_isinstance(masker, "pandas.core.series.Series") or safe_isinstance(masker, "pandas.core.frame.DataFrame"):
        masker = masker.values
        
    results = {'metrics': list(), 'output': dict()}
    for sort_order in metrics['sort_order']:
        for perturbation in metrics['perturbation']:
            if sort_order == "positive" or sort_order == "negative": 
                score_function = lambda true, pred: np.mean(pred)
                update(model, X, y, explainer, masker, sort_order, score_function, perturbation, results, *args)

            if sort_order == "absolute": 
                score_function = sklearn.metrics.r2_score
                update(model, X, y, explainer, masker, sort_order, score_function, perturbation, results, score="r2", *args)

                score_function = sklearn.metrics.roc_auc_score
                update(model, X, y, explainer, masker, sort_order, score_function, perturbation, results, score="roc auc", *args)

    return results 

def dataframe(benchmarks, trend=False):
    '''
    results - dictionary of benchmark 
    '''
    explainers = list()
    for explainer in benchmarks: 
        metrics = benchmarks[explainer]['metrics']
        plt.clf()
        if trend: 
            for metric in benchmarks[explainer]['output']: 
                fcounts, scores, _ = benchmarks[explainer]['output'][metric]
                plt.plot(fcounts, scores, '--o')
                plt.show()
        else: 
            explainer_metric = [explainer]
            for metric in benchmarks[explainer]['output']:
                _, _, auc = benchmarks[explainer]['output'][metric]
                explainer_metric.append(auc)
            explainers.append(explainer_metric)

    df = pd.DataFrame(explainers, columns=['Explainers']+metrics)

    return df 

markers = dict()
def process_dataframe(df):
    explainers = df['Explainers']
    df = df.set_index('Explainers')

    metrics = df.columns
    for metric in metrics:
        if df[metric].dtype == bool:
            markers[df.columns.get_loc(metric)] = df[metric]

    bool_column = np.array([i/(len(explainers)-1) for i in range(0, len(explainers))]).reshape(-1, 1)
    for bool_index in markers:
        df[metrics[bool_index]] = bool_column

    min_per_metric = df.min(axis=0)
    df = df.sub(min_per_metric, axis=1)
    
    max_per_normalized = df.max(axis=0)
    percent_df = df.divide(max_per_normalized, axis=1)

    return explainers, percent_df

def singleplot():
    pass 

def multiplot(explainers, df):
    '''
    df - dataframe of benchmark results
    '''
    metrics = df.columns 
    ax = plt.gca()
    for explainer in explainers:
        plt.plot(df.loc[explainer], '--o')

    ax.tick_params(which='major', axis='both', labelsize=8)

    ax.set_yticks([i/(len(explainers)-1) for i in range(0, len(explainers))])
    ax.set_yticklabels(explainers, rotation=0)

    ax.set_xticks([i for i in range(len(metrics))])
    ax.set_xticklabels(metrics, rotation=45, ha='right')

    plt.grid(which='major', axis='x', linestyle='--')
    plt.tight_layout()
    plt.show()


metrics = {'sort_order': ['positive', 'negative'], 'perturbation': ['keep', 'remove']}

model_generator = lambda: xgboost.XGBRegressor(n_estimators=100, subsample=0.3)
X,y = shap.datasets.boston()
X = X.values

test_size = 0.3 
random_state = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

model = model_generator().fit(X_train, y_train)

permutation_explainer = shap.Explainer(model.predict, X, algorithm='permutation')
tree_explainer = shap.Explainer(model, X, algorithm='tree')
exact_explainer = shap.Explainer(model.predict, X, algorithm='exact')

benchmarks = dict()
masker = X_train
benchmarks[permutation_explainer.name] = benchmark(model.predict, X_train, y_train, permutation_explainer, masker, metrics)
benchmarks[tree_explainer.name] = benchmark(model.predict, X_train, y_train, tree_explainer, masker, metrics)

df = dataframe(benchmarks)
explainers, processed_df = process_dataframe(df)
multiplot(explainers, processed_df)