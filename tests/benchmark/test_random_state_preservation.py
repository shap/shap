"""benchmark helpers must restore the global numpy RNG after temporary seeding."""

import numpy as np

from shap.benchmark import measures


def test_const_rand_preserves_global_rng():
    np.random.seed(424242)
    expected = np.random.rand(7)
    np.random.seed(424242)
    out = measures.const_rand(20, seed=999)
    assert out.shape == (20,)
    assert np.all(np.random.rand(7) == expected)


def test_const_shuffle_preserves_global_rng():
    np.random.seed(424242)
    expected = np.random.rand(7)
    np.random.seed(424242)
    arr = np.arange(50)
    measures.const_shuffle(arr, seed=1001)
    assert np.all(np.random.rand(7) == expected)
