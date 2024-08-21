from ._result import BenchmarkResult


class ComputeTime:
    """Extracts a runtime benchmark result from the passed Explanation."""

    def __call__(self, explanation, name):
        return BenchmarkResult("compute time", name, value=explanation.compute_time / explanation.shape[0])
