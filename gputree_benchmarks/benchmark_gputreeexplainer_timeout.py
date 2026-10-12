"""Run CPU/GPU benchmarks on trained random forests with process timeouts.

Run from the repository root with the same environment as the original script:
    python gputree_benchmarks/benchmark_gputreeexplainer_timeout.py

Edit TIMEOUT_SECONDS and the workload grid in run() to change the workloads. Timings include
explainer construction and exp(X), but exclude process startup and data creation.
After a backend times out, subsequent workloads with more samples at the same
feature count and model configuration are recorded as timeouts without starting a worker.
"""

import argparse
import logging
import multiprocessing as mp
import platform
import re
import subprocess
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from multiprocessing.connection import wait
from pathlib import Path
from time import perf_counter

logger = logging.getLogger(__name__)
TIMEOUT_SECONDS = 1_200
SETUP_TIMEOUT_SECONDS = 180
benchmark_dict = {}
run_id = 0


def hardware_names():
    """Detect hardware locally, tolerating unavailable GPU tooling."""
    cpu = platform.processor() or platform.machine()
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass

    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10,
        )
        gpu = result.stdout.strip().splitlines()
    except (OSError, subprocess.SubprocessError):
        logger.warning("Could not determine GPU names using nvidia-smi")
        gpu = []
    return cpu, gpu


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


def forest_statistics(model):
    """Summarize actual fitted tree sizes; depths count edges from the root."""
    trees = [estimator.tree_ for estimator in model.estimators_]
    depths = [tree.max_depth for tree in trees]
    return {
        "max_depth": model.max_depth,
        "n_trees": len(trees),
        "min_tree_depth": min(depths),
        "max_tree_depth": max(depths),
        "mean_tree_depth": sum(depths) / len(trees),
        "mean_leaves_per_tree": sum(tree.n_leaves for tree in trees) / len(trees),
        "total_nodes": sum(tree.node_count for tree in trees),
    }


def benchmark_function(model, class_name, n_samples, n_features):
    model_details = forest_statistics(model)
    global run_id
    run_id += 1
    context = (
        f"Run {run_id}: {class_name} "
        f"(n_trees={model_details['n_trees']}, max_depth={model_details['max_depth']}, "
        f"n_samples={n_samples}, n_features={n_features})"
    )
    logger.info("%s starting", context)
    for previous_run_id, previous in benchmark_dict.items():
        if (
            previous["class"] == class_name
            and previous["status"] == "timeout"
            and previous["phase"] == "explanation"
            and all(previous.get(key) == value for key, value in model_details.items())
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
    result.update({"class": class_name, "data_shape": [n_samples, n_features], **model_details})
    benchmark_dict[run_id] = result
    if result["status"] == "success":
        logger.info("%s took %.3f seconds", context, result["time"])
    elif result.get("skipped"):
        logger.warning(
            "%s skipped because run %d timed out",
            context, result["timeout_run_id"],
        )
    elif result["status"] == "timeout":
        logger.warning("%s timed out during %s after %s seconds", context, result["phase"], result["timeout_seconds"])
    else:
        logger.error("%s failed: %s", context, result["error"])
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


def save_results(cpu, gpu, metadata=None, output_path=None):
    metadata = {
        "CPU": cpu,
        "GPU": gpu,
        "OS": platform.platform(),
        "Python": platform.python_version(),
        "Explanation timeout (seconds)": TIMEOUT_SECONDS,
        "Setup timeout (seconds)": SETUP_TIMEOUT_SECONDS,
        "Timing scope": "Explainer construction and exp(X); excludes process startup and data creation",
        "Skip rule": "More samples at the same feature count and model configuration, for the same explainer type, after an explanation timeout",
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
        "| run_id | explainer_type | n_samples | n_features | max_depth | n_trees | min–max tree depth | mean tree depth | mean leaves/tree | total_nodes | time (s) | status | validation | details |",
        "| ---: | --- | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | ---: | --- | --- | --- |",
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
            result["max_depth"], result["n_trees"],
            f"{result['min_tree_depth']}–{result['max_tree_depth']}",
            f"{result['mean_tree_depth']:.2f}", f"{result['mean_leaves_per_tree']:.2f}",
            result["total_nodes"],
            f"{result['time']:.6f}" if result["status"] == "success" else "—",
            result["status"], result.get("validation", "—"), details,
        ]
        lines.append("| " + " | ".join(markdown_cell(cell) for cell in cells) + " |")
    output_path = Path(output_path) if output_path is not None else Path(__file__).with_name("benchmarks_timeout.md")
    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    logger.info("Saved benchmark results to %s", output_path)


def run(n_samples=(100, 1_000, 10_000), output_path=None):
    from sklearn.datasets import make_regression
    from sklearn.ensemble import RandomForestRegressor

    sample_sizes = sorted(set([n_samples] if isinstance(n_samples, int) else n_samples))
    logger.info("Benchmark grid: samples=%s, trees=%s, depths=%s", sample_sizes, [10, 50, 100], [12, 20, 28, 36])

    # Hold feature count fixed to isolate batch size, depth and ensemble size.
    n_features = 50
    n_training_samples = 20_000
    max_depths = [12, 20, 28, 36]
    tree_counts = [10, 50, 100]
    cpu, gpu = hardware_names()
    metadata = {
        "Started (UTC)": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "Model": "RandomForestRegressor(max_depth=grid, n_estimators=grid, random_state=0); other parameters use sklearn defaults",
        "Depth convention": "Root depth is 0; max_depth is a configured limit, actual depths are measured after fitting",
        "Tree depth statistics": "Min, max and mean of each tree's maximum path depth across the forest",
        "Training data": f"One shared make_regression invocation: {n_training_samples} samples, {n_features} features, 10 informative features, 1 target, noise=0.0, random_state=0",
        "Explanation data": "Standard normal, numpy default_rng seed=1",
        "Feature perturbation": "tree_path_dependent",
        "Validation": "First 100 samples; CPU/GPU assert_allclose with rtol=1e-4, atol=1e-4",
        "Max depth limits": max_depths,
        "Tree counts": tree_counts,
        "Explanation sample counts": sample_sizes,
        "Explanation feature count": n_features,
    }
    for package in ("shap", "numpy", "scikit-learn"):
        try:
            metadata[f"{package} version"] = version(package)
        except PackageNotFoundError:
            metadata[f"{package} version"] = "unavailable"
    X_train, y_train = make_regression(
        n_samples=n_training_samples, n_features=n_features, n_informative=10,
        n_targets=1, noise=0.0, random_state=0,
    )
    for max_depth in max_depths:
        for n_trees in tree_counts:
            logger.info("Training forest with max_depth=%d and n_trees=%d", max_depth, n_trees)
            model = RandomForestRegressor(
                max_depth=max_depth, n_estimators=n_trees, random_state=0
            )
            model.fit(X_train, y_train)
            for batch_samples in sample_sizes:
                compare_by_size(model, batch_samples, n_features)
                save_results(cpu, gpu, metadata, output_path=output_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-samples", type=int, nargs="+", default=[100, 1_000, 10_000],
                        help="Explanation batch sizes (default: 100 1000 10000)")
    parser.add_argument("--output", type=Path, default=Path(__file__).with_name("benchmarks_timeout.md"))
    args = parser.parse_args()
    if any(samples < 1 for samples in args.n_samples):
        parser.error("--n-samples must be positive")
    run(n_samples=args.n_samples, output_path=args.output)
