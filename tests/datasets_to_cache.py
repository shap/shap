"""
Configuration file listing all datasets used in tests.

This file is used by the CI pipeline to pre-download datasets before running tests.
"""

# Shap datasets that download from URLs
SHAP_DATASETS = [
    "imagenet50",
    "california",
    "imdb",
    "adult",
    "nhanesi",
    "a1a",
    "rank",
    "linnerud",
    "diabetes",
    "iris",
]

# Sklearn datasets that need to be fetched (download from internet)
SKLEARN_FETCH_DATASETS = [
    {
        "name": "fetch_california_housing",
        "kwargs": {},
    },
    {
        "name": "fetch_20newsgroups",
        "kwargs": {"subset": "train", "categories": ["alt.atheism", "talk.religion.misc"]},
    },
    {
        "name": "fetch_20newsgroups",
        "kwargs": {"subset": "test", "categories": ["alt.atheism", "talk.religion.misc"]},
    },
]

# Sklearn datasets that are bundled (no download, but we load them to ensure they work)
SKLEARN_LOAD_DATASETS = [
    "load_breast_cancer",
    "load_digits",
    "load_iris",
    "load_wine",
    "load_diabetes",
    "load_linnerud",
]


def download_all_datasets():
    """Download all datasets used in tests."""
    from sklearn import datasets as sklearn_datasets

    import shap

    print("Downloading shap datasets...")
    for dataset_name in SHAP_DATASETS:
        try:
            dataset_func = getattr(shap.datasets, dataset_name)
            print(f"  - {dataset_name}...", end=" ")
            dataset_func()
            print("✓")
        except Exception as e:
            print(f"✗ (Error: {e})")

    print("\nFetching sklearn datasets...")
    for dataset_config in SKLEARN_FETCH_DATASETS:
        try:
            dataset_func = getattr(sklearn_datasets, dataset_config["name"])
            kwargs_str = ", ".join(f"{k}={v}" for k, v in dataset_config["kwargs"].items())
            print(f"  - {dataset_config['name']}({kwargs_str})...", end=" ")
            dataset_func(**dataset_config["kwargs"])
            print("✓")
        except Exception as e:
            print(f"✗ (Error: {e})")

    print("\nLoading sklearn bundled datasets...")
    for dataset_name in SKLEARN_LOAD_DATASETS:
        try:
            dataset_func = getattr(sklearn_datasets, dataset_name)
            print(f"  - {dataset_name}...", end=" ")
            dataset_func()
            print("✓")
        except Exception as e:
            print(f"✗ (Error: {e})")

    print("\nAll datasets processed!")


if __name__ == "__main__":
    download_all_datasets()
