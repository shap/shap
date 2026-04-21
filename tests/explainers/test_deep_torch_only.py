"""Regression guardrail for https://github.com/shap/shap/issues/3662.

The PyTorch code path of ``shap.DeepExplainer`` must not pull in tensorflow /
tf_keras / keras. This test runs the full PyTorch flow in a subprocess where
those modules are blocked in ``sys.modules`` so that a later regression
reintroducing an eager TF import would surface here.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap

import pytest

pytest.importorskip("torch")


BLOCKED = ("tensorflow", "tf_keras", "keras")

# Runs in a fresh Python process. The sentinel ``None`` in ``sys.modules`` makes
# ``import tensorflow`` (and the other names) raise ``ImportError`` as if the
# package were not installed, even if it is actually present in the test env.
_SUBPROCESS_SCRIPT = textwrap.dedent(
    """
    import sys

    BLOCKED = {blocked!r}
    for name in BLOCKED:
        sys.modules[name] = None

    import numpy as np
    import torch
    import torch.nn as nn

    import shap

    for name in BLOCKED:
        assert name not in sys.modules or sys.modules[name] is None, (
            f"{{name}} was loaded during 'import shap'"
        )

    class TinyNet(nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = nn.Linear(4, 2)

        def forward(self, x):
            return self.fc(x)

    torch.manual_seed(0)
    model = TinyNet().eval()
    background = torch.randn(4, 4)
    X = torch.randn(2, 4)

    explainer = shap.DeepExplainer(model, background)
    # check_additivity=False sidesteps an unrelated, model-specific additivity
    # assertion that is not what this test is guarding against.
    explainer.shap_values(X, check_additivity=False)

    for name in BLOCKED:
        assert name not in sys.modules or sys.modules[name] is None, (
            f"{{name}} was loaded during the PyTorch DeepExplainer flow"
        )

    print("OK")
    """
).format(blocked=BLOCKED)


def test_pytorch_deep_explainer_does_not_import_tensorflow():
    result = subprocess.run(
        [sys.executable, "-c", _SUBPROCESS_SCRIPT],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, (
        "PyTorch DeepExplainer should run without importing tensorflow/tf_keras/keras.\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
    assert result.stdout.strip().endswith("OK"), result.stdout
