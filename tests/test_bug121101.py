# ruff: noqa
# fmt: off
import time

import torch
import lightgbm
import numpy as np
# from sklearn.datasets import fetch_california_housing


def test_something():
    # X, y = fetch_california_housing(return_X_y=True)
    X = np.ones(shape=(200, 20))
    torch.tensor(X)
    time.sleep(3)
