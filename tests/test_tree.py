import sklearn
import numpy as np
import shap

def test_sklearn_interaction():
    from sklearn.model_selection import train_test_split
    from sklearn.ensemble import RandomForestClassifier
    X,y = shap.datasets.iris()
    X_train,X_test,Y_train,Y_test = train_test_split(*shap.datasets.iris(), test_size=0.2, random_state=0)
    rforest = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0)
    model = rforest.fit(X_train, Y_train)
    interaction_vals = shap.TreeExplainer(model).shap_interaction_values(X)
    for i in range(len(interaction_vals)):
        for j in range(len(interaction_vals[i])):
            for k in range(len(interaction_vals[i][j])):
                for l in range(len(interaction_vals[i][j][k])):
                    assert abs(interaction_vals[i][j][k][l]-interaction_vals[i][j][l][k])<0.0000001
            if j<len(interaction_vals[i])-1:
                assert abs(interaction_vals[i][j][len(interaction_vals[i][j])-1][len(interaction_vals[i][j])-1]-interaction_vals[i][j+1][len(interaction_vals[i][j])-1][len(interaction_vals[i][j])-1])<0.0000001
