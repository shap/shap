import os

import numpy as np
import pytest


@pytest.fixture()
def random_seed():
    """Provides a reproducible random seed for tests that use randomness.

    If the test fails, the random seed used will be displayed in the pytest logs.

    By default, each run of the test will use a different seed. Alternatively,
    the seed can be fixed setting an environment varable TEST_RANDOM_SEED.

    Example use in a test:

        def test_thing(random_seed):
            # Numpy random values
            rng = np.random.default_rng(seed=random_seed)
            values = rng.integers(...)

            # Pytorch random values
            torch.manual_seed(random_seed)

    """
    try:
        # If set, use a seed from environment variable
        seed = int(os.environ['TEST_RANDOM_SEED'])
    except KeyError:
        # Otherwise, create a new seed for each test
        rng = np.random.default_rng()
        seed = rng.integers(0, 1000)
    return seed
