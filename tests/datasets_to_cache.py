"""
Configuration file listing all datasets used in tests.

This file is used by the CI pipeline to pre-download datasets before running tests.
"""

from __future__ import annotations

from typing import TypedDict


class DictKwargs(TypedDict, total=False):
    subset: str
    categories: list[str]


class Dataset(TypedDict):
    name: str
    kwargs: DictKwargs


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
SKLEARN_FETCH_DATASETS: list[Dataset] = [
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

    print("\nAll datasets processed!")


if __name__ == "__main__":
    download_all_datasets()
