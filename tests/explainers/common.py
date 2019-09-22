import typing
import itertools

import numpy as np


def drop(x: np.ndarray, to_drop: typing.Iterable[int]):
    """
    Returns x with ``to_drop`` features set to their average value.
    """
    x = np.copy(x)
    for c in to_drop:
        x[:, c] = np.mean(x[:, c])
    return x


def assert_accuracy(phi, f, x, rtol=1e-7, atol=0):
    """
    Asserts that Shapley values phi are accurate in the sense of Eq (5) in
    "A Unified Approach to Interpreting Model Predictions"
    """
    empty_x = drop(x, list(range(x.shape[1])))
    predictions = f(empty_x) + np.sum(phi, axis=1)
    expected = f(x)
    np.testing.assert_allclose(predictions, expected, rtol=rtol, atol=atol)


def assert_monotonicity(x: np.ndarray, f1, phi1: np.ndarray, f2, phi2: np.ndarray, i: int):
    """
    Asserts that Shapley values phi1 and phi2 are consistent in the sense of Eq (7) in
    "A Unified Approach to Interpreting Model Predictions"
    """
    columns = list(range(x.shape[1]))
    f_set = set(columns)
    f_less_i = f_set - {i}

    diffs_plus = np.full(len(x), True)
    diffs_minus = np.full(len(x), True)
    for s in range(len(f_less_i) + 1):
        # s is the number of features other than `i`. We loop through all their combinations.
        for s_set in itertools.combinations(f_less_i, s):
            s_set = set(s_set)

            s_u_i_set = s_set.union({i})
            x_s_u_i = drop(x, f_set - s_u_i_set)
            x_s = drop(x, f_set - s_set)

            diff1 = f1(x_s_u_i) - f1(x_s)
            diff2 = f2(x_s_u_i) - f2(x_s)
            diffs_plus &= (diff1 >= diff2)
            diffs_minus &= (diff2 >= diff1)

    inconsistency = np.array(
        (
            diffs_plus &
            (phi1[:, i] <= phi2[:, i])
        ) | (
            diffs_minus &
            (phi2[:, i] <= phi1[:, i])
        )
    )

    assert (~inconsistency).all(), x[inconsistency]
