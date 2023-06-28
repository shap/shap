import os

import numpy as np
import pytest


@pytest.fixture()
def random_seed():
    """Provides a random seed for tests that require randomness, for reproducibility.

    The seed can be overwritten by setting an environment varable TEST_RANDOM_SEED.

    Example use in a test:
        rng = np.random.default_rng(seed=random_seed)

    """
    try:
        # If set, use a seed from environment variable
        seed = int(os.environ['TEST_RANDOM_SEED'])
    except KeyError:
        # Otherwise, create a new seed for each test
        rng = np.random.default_rng()
        seed = rng.integers(0, 1000)

    # Ensure the seed is printed to the pytest logs for failing tests
    print("RANDOM SEED: ", seed)
    return seed
