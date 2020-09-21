import matplotlib.pyplot as plt 
import pandas as pd 
import numpy as np 

root = 'C:/Users/maggiewu/Desktop/'
model = 'Explainers'

df = pd.read_excel(root + '{}.xlsx'.format(model))
explainers = df['Explainers']

df = df.set_index('Explainers')
metrics = df.columns
model = model.replace('_', ' ')
model = 'Random Forest'
markers = dict()

def decide_markers(df):
    for metric in metrics:
        if df[metric].dtype == bool:
            markers[df.columns.get_loc(metric)] = df[metric]

def set_bool(df):
    bool_column = np.array([i/(len(explainers)-1) for i in range(0, len(explainers))]).reshape(-1, 1)
    for bool_index in markers:
        df[metrics[bool_index]] = bool_column

def normalize(df):
    min_per_metric = df.min(axis=0)
    df = df.sub(min_per_metric, axis=1)
    
    max_per_normalized = df.max(axis=0)
    percent_df = df.divide(max_per_normalized, axis=1)

    return percent_df

def plot_explainers(df):
    ax = plt.gca()
    for explainer in explainers:
        plt.plot(df.loc[explainer], '--o')

    ax.tick_params(which='major', axis='both', labelsize=8)

    ax.set_yticks([i/(len(explainers)-1) for i in range(0, len(explainers))])
    ax.set_yticklabels(explainers, rotation=0)

    ax.set_xticks([i for i in range(len(df.columns))])
    ax.set_xticklabels(metrics, rotation=45, ha='right')

    plt.title(model)
    plt.grid(which='major', axis='x', linestyle='--')
    plt.tight_layout()
    plt.show()

decide_markers(df)
set_bool(df)
df = normalize(df)
plot_explainers(df)