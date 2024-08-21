import os

import pytest


@pytest.mark.skipif(os.getenv("GITHUB_ACTIONS") != "true", reason="Only enforced when running on CI")
def test_importing_from_installed_package_not_local_files():
    # If using an editable install (pip install --editable), then shap will
    # be imported directly from the repo. This is a common setup when developing.

    # However when running tests on CI, we want to test against the *installed* package.
    # This ensures that the library is packaged correctly.

    import shap

    assert "site-packages" in shap.__file__
