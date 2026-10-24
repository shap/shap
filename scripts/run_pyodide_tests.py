"""Run the existing test suite in a Node-backed Pyodide environment."""

import os
import shutil
import sys
from pathlib import Path


def main():
    if sys.platform != "emscripten":
        raise SystemExit("Run this script with the Pyodide virtual environment's Python.")

    import pyodide_js
    from pyodide.ffi import run_sync

    # Enable sockets before importing SHAP, pytest, or the test modules.
    run_sync(pyodide_js.useNodeSockFS())

    import pytest

    import shap
    from shap import _cext, _cutils

    print(f"Testing installed SHAP: {shap.__file__}")
    print(f"WASM extensions: {_cext.__file__}, {_cutils.__file__}")

    # Reuse native CI downloads; socket support alone does not guarantee TLS support.
    cache = os.environ.get("SHAP_TEST_DATA_CACHE")
    if cache:
        cache = Path(cache)
        shutil.copytree(cache / "cached_data", Path(shap.__file__).parent / "cached_data", dirs_exist_ok=True)
        sklearn_cache = Path(os.environ.get("SCIKIT_LEARN_DATA", Path.home() / "scikit_learn_data"))
        shutil.copytree(cache / "scikit_learn_data", sklearn_cache, dirs_exist_ok=True)

    raise SystemExit(pytest.main(sys.argv[1:]))


if __name__ == "__main__":
    main()
