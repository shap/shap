# ruff: noqa
# fmt: off
import time

import lightgbm
import torch
from sklearn.datasets import fetch_california_housing


def test_something():
    X, y = fetch_california_housing(return_X_y=True)
    torch.tensor(X)
    time.sleep(3)
