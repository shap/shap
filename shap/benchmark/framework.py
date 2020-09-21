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

metric_names = {
    'runtime': shap.benchmark.metrics.runtime,
    'efficiency': shap.benchmark.metrics.local_accuracy,
    'keep positive mask': shap.benchmark.metrics.keep_positive_mask, 
    'keep negative mask': shap.benchmark.metrics.keep_negative_mask, 
    'keep absolute mask r2': shap.benchmark.metrics.keep_absolute_mask__r2, 
    'keep absolute mask roc auc': shap.benchmark.metrics.keep_absolute_mask__roc_auc, 
    'remove positive mask': shap.benchmark.metrics.remove_positive_mask, 
    'remove negative mask': shap.benchmark.metrics.remove_negative_mask, 
    'remove absolute mask r2': shap.benchmark.metrics.remove_absolute_mask__r2, 
    'remove absolute mask roc auc': shap.benchmark.metrics.remove_absolute_mask__roc_auc, 
    'keep positive impute': shap.benchmark.metrics.keep_positive_impute, 
    'keep negative impute': shap.benchmark.metrics.keep_negative_impute, 
    'keep absolute impute r2': shap.benchmark.metrics.keep_absolute_impute__r2, 
    'keep absolute impute roc auc': shap.benchmark.metrics.keep_absolute_impute__roc_auc, 
    'remove positive impute': shap.benchmark.metrics.remove_positive_impute, 
    'remove negative impute': shap.benchmark.metrics.remove_negative_impute, 
    'remove absolute impute r2': shap.benchmark.metrics.remove_absolute_impute__r2, 
    'remove absolute impute roc auc': shap.benchmark.metrics.remove_absolute_impute__roc_auc, 
    'keep positive resample': shap.benchmark.metrics.keep_positive_resample, 
    'keep negative resample': shap.benchmark.metrics.keep_negative_resample, 
    'keep absolute resample r2': shap.benchmark.metrics.keep_absolute_resample__r2, 
    'keep absolute resample auc roc': shap.benchmark.metrics.keep_absolute_resample__roc_auc,
    'remove positive resample': shap.benchmark.metrics.remove_positive_resample, 
    'remove negative resample': shap.benchmark.metrics.remove_negative_resample, 
    'remove absolute resample r2': shap.benchmark.metrics.remove_absolute_resample__r2, 
    'remove absolute resample auc roc': shap.benchmark.metrics.remove_absolute_resample__roc_auc,
}

def benchmark(model, X, y, explainer, masker, metrics):
    '''
    model - trained model 
    X, y - data for explainer 
    explainer - explainer object 
    masker - masker object 
    metrics - list of string 
    '''
    results = {'metrics': metrics, 'output': dict()}
    for metric in metrics: 
        fcounts, scores = metric_names[metric](X, y, model, explainer)
        if metric == 'runtime': 
            runtime = scores 
            results['output'][metric] = [None, None, runtime]
        else: 
            auc = sklearn.metrics.auc(fcounts, scores) / fcounts[-1]
            results['output'][metric] = [fcounts, scores, auc]
    
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

def plot(explainers, df):
    '''
    df - dataframe of benchmark results
    '''
    ax = plt.gca()
    for explainer in explainers:
        plt.plot(df.loc[explainer], '--o')

    ax.tick_params(which='major', axis='both', labelsize=8)

    ax.set_yticks([i/(len(explainers)-1) for i in range(0, len(explainers))])
    ax.set_yticklabels(explainers, rotation=0)

    ax.set_xticks([i for i in range(len(df.columns))])
    ax.set_xticklabels(metrics, rotation=45, ha='right')

    plt.grid(which='major', axis='x', linestyle='--')
    plt.tight_layout()
    plt.show()


metrics = ['keep positive mask']

model_generator = lambda: xgboost.XGBRegressor(n_estimators=100, subsample=0.3)
X,y = shap.datasets.boston()
test_size = 0.3 
random_state = 0
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

model = model_generator().fit(X_train, y_train)

permutation_explainer = shap.Explainer(model.predict, X, algorithm='permutation')
tree_explainer = shap.Explainer(model, X, algorithm='tree')
exact_explainer = shap.Explainer(model.predict, X, algorithm='exact')

benchmarks = dict()
benchmarks[permutation_explainer.name] = benchmark(model_generator, X_train, y_train, permutation_explainer, X_train, metrics)
benchmarks[tree_explainer.name] = benchmark(model_generator, X_train, y_train, tree_explainer, X_train, metrics)
benchmarks[exact_explainer.name] = benchmark(model_generator, X_train, y_train, exact_explainer, X_train, metrics)

df = dataframe(benchmarks)
explainers, processed_df = process_dataframe(df)
plot(explainers, processed_df)


# model_generator = lambda: xgboost.XGBClassifier()
# X,y = shap.datasets.adult()
# test_size = 100 
# random_state = 0
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# model = xgboost.XGBClassifier().fit(X, y)

# masker = shap.maskers.Partition(X)
# permutation_explainer = shap.Explainer(model.predict, masker, algorithm='permutation')
# exact_explainer = shap.Explainer(model.predict, masker)

# benchmarks = dict()
# benchmarks[permutation_explainer.name] = benchmark(model_generator, X, y, permutation_explainer, masker, metrics)
# benchmarks[exact_explainer.name] = benchmark(model_generator, X, y, exact_explainer, masker, metrics)

# df = dataframe(benchmarks)
# explainers, processed_df = process_dataframe(df)
# plot(explainers, processed_df)


