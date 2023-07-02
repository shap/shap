import os

import numpy as np
import pytest


@pytest.fixture()
def random_seed():
    """Provides a test-specific random seed for reproducible "fuzz testing".

    By default, each run of the test will use a different seed. Alternatively,
    the seed can be fixed by setting an environment varable TEST_RANDOM_SEED.

    If the test fails, the random seed used will be displayed in the pytest
    logs.

    Example use in a test:

        def test_thing(random_seed):

            # Numpy
            rng = np.random.default_rng(seed=random_seed)
            values = rng.integers(...)

            # Pytorch
            torch.manual_seed(random_seed)

            # Tensorflow
            tf.compat.v1.random.set_random_seed(random_seed)

    """
    try:
        # If set, use a seed from environment variable
        seed = int(os.environ['TEST_RANDOM_SEED'])
    except KeyError:
        # Otherwise, create a new seed for each test
        rng = np.random.default_rng()
        seed = rng.integers(0, 1000)
    return seed


@pytest.fixture(autouse=True)
def global_random_seed():
    """Set the global numpy random seed before each test

    Nb. Tests that use random numbers should instantiate a random number
    Generator with `np.random.default_rng` rather than use the global numpy
    random state.
    """
    np.random.seed(0)
