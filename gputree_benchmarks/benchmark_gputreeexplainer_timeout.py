"""Run CPU/GPU benchmarks with process timeouts.

Run from the repository root with the same environment as the original script:
    python gputree_benchmarks/benchmark_gputreeexplainer_timeout.py

Edit TIMEOUT_SECONDS and sizes below to change the workloads. Timings include
explainer construction and exp(X), but exclude process startup and data creation.
After a backend times out, subsequent workloads with more samples at the same
feature count are recorded as timeouts without starting a worker.
"""

import logging
import multiprocessing as mp
import platform
import re
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from multiprocessing.connection import wait
from pathlib import Path
from time import perf_counter

logger = logging.getLogger(__name__)
TIMEOUT_SECONDS = 1_200
SETUP_TIMEOUT_SECONDS = 60
benchmark_dict = {}
run_id = 0


def explain(model, explainer_class, data):
    exp = explainer_class(model, feature_perturbation="tree_path_dependent")
    return exp(data)


def explanation_worker(connection, model, class_name, n_samples, n_features):
    try:
        import numpy as np
        import shap

        # Identical nested batches for CPU/GPU, without sending large arrays
        # through multiprocessing or retaining them in the parent process.
        X = np.random.default_rng(1).standard_normal((n_samples, n_features))
        explainer_class = getattr(shap, class_name)
        connection.send({"status": "ready"})
        start = perf_counter()
        result = explain(model, explainer_class, X)
        elapsed = perf_counter() - start
        connection.send({
            "status": "success",
            "time": elapsed,
            # Only return a small subset for correctness comparison.
            "values": result.values[:100].tolist(),
        })
    except Exception as exc:
        connection.send({"status": "error", "error": f"{type(exc).__name__}: {exc}"})
    finally:
        connection.close()


def receive_message(connection, process, timeout):
    ready = wait([connection, process.sentinel], timeout=timeout)
    if connection in ready or connection.poll():
        try:
            return connection.recv()
        except EOFError:
            pass
    if process.sentinel in ready:
        process.join()
        return {"status": "error", "error": f"Worker exited with code {process.exitcode}"}
    return None


def run_worker(target, args, timeout_seconds=TIMEOUT_SECONDS, setup_timeout_seconds=SETUP_TIMEOUT_SECONDS):
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=target, args=(sender, *args))
    started = False
    try:
        process.start()
        started = True
        sender.close()
        message = receive_message(receiver, process, setup_timeout_seconds)
        if message is None:
            return {"status": "timeout", "phase": "setup", "timeout_seconds": setup_timeout_seconds}
        if message["status"] != "ready":
            return message
        message = receive_message(receiver, process, timeout_seconds)
        if message is None:
            return {"status": "timeout", "phase": "explanation", "timeout_seconds": timeout_seconds}
        return message
    finally:
        if started:
            process.join(timeout=0.1)
            if process.is_alive():
                process.terminate()
                process.join(timeout=5)
            if process.is_alive():
                process.kill()
                process.join()
            process.close()
        sender.close()
        receiver.close()


def benchmark_function(model, class_name, n_samples, n_features):
    global run_id
    run_id += 1
    logger.info("Starting run %d: %s with data shape (%d, %d)", run_id, class_name, n_samples, n_features)
    for previous_run_id, previous in benchmark_dict.items():
        if (
            previous["class"] == class_name
            and previous["status"] == "timeout"
            and n_samples > previous["data_shape"][0]
            and n_features == previous["data_shape"][1]
        ):
            result = {
                "status": "timeout",
                "phase": previous["phase"],
                "timeout_seconds": previous["timeout_seconds"],
                "skipped": True,
                "timeout_run_id": previous.get("timeout_run_id", previous_run_id),
            }
            break
    else:
        result = run_worker(explanation_worker, (model, class_name, n_samples, n_features))
    values = result.pop("values", None)
    result.update({"class": class_name, "data_shape": [n_samples, n_features]})
    benchmark_dict[run_id] = result
    if result["status"] == "success":
        logger.info("Run %d: %s took %.3f seconds", run_id, class_name, result["time"])
    elif result.get("skipped"):
        logger.warning(
            "Run %d: skipping %s with data shape (%d, %d) because run %d timed out",
            run_id, class_name, n_samples, n_features, result["timeout_run_id"],
        )
    elif result["status"] == "timeout":
        logger.warning("Run %d: %s timed out during %s after %s seconds", run_id, class_name, result["phase"], result["timeout_seconds"])
    else:
        logger.error("Run %d: %s failed: %s", run_id, class_name, result["error"])
    return values


def compare_by_size(model, n_samples, n_features):
    import numpy as np

    cpu_values = benchmark_function(model, "TreeExplainer", n_samples, n_features)
    gpu_values = benchmark_function(model, "GPUTreeExplainer", n_samples, n_features)
    if cpu_values is not None and gpu_values is not None:
        try:
            np.testing.assert_allclose(cpu_values, gpu_values, rtol=1e-4, atol=1e-4)
        except AssertionError as exc:
            logger.error("CPU/GPU correctness comparison failed: %s", exc)
            for result_id in (run_id - 1, run_id):
                benchmark_dict[result_id]["validation"] = "failed"
        else:
            for result_id in (run_id - 1, run_id):
                benchmark_dict[result_id]["validation"] = "passed"
    else:
        logger.info("Skipping this batch's correctness comparison: both backends must finish")


def markdown_cell(value):
    return str(value).replace("&", "&amp;").replace("<", "&lt;").replace(
        ">", "&gt;"
    ).replace("|", "&#124;").replace("\r", " ").replace("\n", "<br>")


def hardware_abbreviation(name):
    name = re.sub(r"\((?:R|TM)\)|[®™]", "", str(name), flags=re.IGNORECASE)
    name = name.split("@", 1)[0]
    name = re.sub(
        r"\b(?:Intel|AMD|NVIDIA|GeForce|CPU|Processor)\b|\b\d+-Core\b",
        "", name, flags=re.IGNORECASE,
    )
    return " ".join(name.split()) or "unknown"


def save_results(cpu, gpu, metadata=None):
    metadata = {
        "CPU": cpu,
        "GPU": gpu,
        "OS": platform.platform(),
        "Python": platform.python_version(),
        "Explanation timeout (seconds)": TIMEOUT_SECONDS,
        "Setup timeout (seconds)": SETUP_TIMEOUT_SECONDS,
        "Timing scope": "Explainer construction and exp(X); excludes process startup and data creation",
        "Skip rule": "More samples at the same feature count, for the same explainer type",
        **(metadata or {}),
    }
    gpu_names = [gpu] if isinstance(gpu, str) else gpu
    gpu_label = ", ".join(hardware_abbreviation(name) for name in gpu_names) or "unknown"
    title = (
        f"# Tree explainer benchmark — CPU: {markdown_cell(hardware_abbreviation(cpu))}"
        f" | GPU: {markdown_cell(gpu_label)}"
    )
    lines = [title, "", "## Metadata", ""]
    lines.extend(f"- **{key}:** {markdown_cell(value)}" for key, value in metadata.items())
    lines.extend([
        "", "## Results", "",
        "| run_id | explainer_type | n_samples | n_features | time (s) | status | validation | details |",
        "| ---: | --- | ---: | ---: | ---: | --- | --- | --- |",
    ])
    for result_id, result in benchmark_dict.items():
        details = ""
        if result.get("skipped"):
            details = f"Skipped because run {result['timeout_run_id']} timed out"
        elif result["status"] == "timeout":
            details = f"{result['phase']} exceeded {result['timeout_seconds']} s"
        elif result["status"] == "error":
            details = result["error"]
        cells = [
            result_id, result["class"], *result["data_shape"],
            f"{result['time']:.6f}" if result["status"] == "success" else "—",
            result["status"], result.get("validation", "—"), details,
        ]
        lines.append("| " + " | ".join(markdown_cell(cell) for cell in cells) + " |")
    output_path = Path(__file__).with_name("benchmarks_timeout.md")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved benchmark results to %s", output_path)


def run():
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor

    from benchmark_gputreeexplainer import hardware_names

    sizes = {
        10: [1_000, 10_000, 100_000, 1_000_000],
        50: [1_000, 10_000, 100_000, 1_000_000],
        100: [1_000, 10_000, 100_000, 1_000_000],
    }
    cpu, gpu = hardware_names()
    metadata = {
        "Started (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "Model": "RandomForestRegressor(n_estimators=100, random_state=0); other parameters use sklearn defaults",
        "Training data": "make_regression: 800 samples, 1 target, noise=0.0, random_state=0",
        "Explanation data": "Standard normal, numpy default_rng seed=1",
        "Feature perturbation": "tree_path_dependent",
        "Validation": "First 100 samples; CPU/GPU assert_allclose with rtol=1e-4, atol=1e-4",
        "Workloads (features → samples)": sizes,
        "Initial comparison samples": 100,
    }
    for package in ("shap", "numpy", "scikit-learn"):
        try:
            metadata[f"{package} version"] = version(package)
        except PackageNotFoundError:
            metadata[f"{package} version"] = "unavailable"
    for n_features, sample_sizes in sizes.items():
        logger.info("Training model with 800 samples and %d features", n_features)
        X_train, y_train = make_regression(
            n_samples=800, n_features=n_features, n_targets=1, noise=0.0, random_state=0
        )
        model = RandomForestRegressor(n_estimators=100, random_state=0)
        model.fit(X_train, y_train)
        # A small independent comparison remains available if larger CPU runs
        # time out. These two runs are also recorded in the output.
        compare_by_size(model, 100, n_features)
        save_results(cpu, gpu, metadata)
        for n_samples in sample_sizes:
            compare_by_size(model, n_samples, n_features)
            save_results(cpu, gpu, metadata)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    run()
