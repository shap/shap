from shap.benchmark._result import BenchmarkResult


class ComputeTime:
    """
    Computes the average runtime per sample from an Explanation object.

    The Explanation object is expected to have:
    - `compute_time`: total runtime of the model
    - `shape`: tuple where shape[0] represents number of samples

    The returned value is:
        compute_time / number_of_samples
    """

    def __call__(self, explanation, name):
        return BenchmarkResult(
            "compute time",
            name,
            value=explanation.compute_time / explanation.shape[0],
        )
