import matplotlib
import numpy as np
matplotlib.use('Agg')
import shap

def test_random_dependence():
    shap.dependence_plot(0, np.random.randn(20, 5), np.random.randn(20, 5), show=False)

def test_random_dependence_no_interaction():
    shap.dependence_plot(0, np.random.randn(20, 5), np.random.randn(20, 5), show=False, interaction_index=None)