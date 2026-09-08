import numpy as np

from shap.benchmark import ExplanationError
from shap.maskers import Independent


def make_benchmark(x, num_permutations=2):
    masker = Independent(np.zeros((1, x.shape[1])))

    def model(v):
        return v.sum(axis=1)

    return ExplanationError(
        masker,
        model,
        x,
        num_permutations=num_permutations,
        seed=0,
    )


def test_aggregates_all_rows_not_just_last():
    x = np.array(
        [
            [1.0, 2.0],
            [3.0, 4.0],
            [5.0, 6.0],
        ]
    )

    attributions = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [5.0, 6.0],
        ]
    )

    benchmark = make_benchmark(x)
    result = benchmark(attributions, "test", silent=True)

    benchmark_last = make_benchmark(x[-1:])
    result_last = benchmark_last(attributions[-1:], "test", silent=True)

    assert result.value > result_last.value + 1e-8, (
        "ExplanationError should aggregate across all rows, not just use the last row"
    )
