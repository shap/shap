import random as rand

import numpy
import pytest


@pytest.fixture(autouse=True)
def set_random_seed():
    rand.seed(0)
    numpy.random.seed(0)
