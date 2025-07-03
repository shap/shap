import importlib.util

import pytest


@pytest.mark.skipif(importlib.util.find_spec("cv2") is None, reason="cv2 (OpenCV) is not installed")
def test_import():
    # FIXME: Remove this test in the future once the follow-ups from #3076
    # are handled.
    import shap.benchmark  # noqa: F401
