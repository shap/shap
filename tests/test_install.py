import os
from pathlib import Path

import pytest


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") != "true", reason="Only enforced when running on CI")
def test_importing_from_installed_package_not_local_files():
    # If using an editable install (pip install --editable), then shap will
    # be imported directly from the repo. This is a common setup when developing.

    # However when running tests on CI, we want to test against the *installed* package.
    # This ensures that the library is packaged correctly.

    import shap

    assert "site-packages" in shap.__file__


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") != "true", reason="Only enforced when running on CI")
def test_installed_package_contains_typing_files():
    import shap

    shap_root = Path(shap.__file__).resolve().parent

    assert (shap_root / "py.typed").is_file()
    assert (shap_root / "__init__.pyi").is_file()
    assert (shap_root / "explainers" / "__init__.pyi").is_file()
    assert (shap_root / "plots" / "__init__.pyi").is_file()
