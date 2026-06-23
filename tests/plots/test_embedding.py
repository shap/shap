import matplotlib.pyplot as plt
import numpy as np
import pytest

import shap


def test_embedding_uses_default_feature_name_with_int_index():
    shap_values = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.3, 0.1, 0.2],
            [0.2, 0.4, 0.1],
            [0.5, 0.2, 0.2],
            [0.4, 0.3, 0.1],
        ]
    )
    embedding_values = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ]
    )

    shap.plots.embedding(1, shap_values, feature_names=None, method=embedding_values, show=False)

    fig = plt.gcf()
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == "SHAP value for\nFeature 1"


def test_embedding_sum_selector_sets_expected_colorbar_label():
    shap_values = np.array(
        [
            [0.1, -0.2, 0.3],
            [0.3, 0.1, -0.2],
            [0.2, 0.4, 0.1],
            [0.5, -0.2, 0.2],
            [0.4, 0.3, -0.1],
        ]
    )
    embedding_values = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ]
    )

    shap.plots.embedding(
        "sum()",
        shap_values,
        feature_names=["f0", "f1", "f2"],
        method=embedding_values,
        show=False,
    )

    fig = plt.gcf()
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == "SHAP value for\nsum(SHAP values)"


def test_embedding_rank_selector_uses_highest_mean_abs_feature_name():
    shap_values = np.array(
        [
            [0.1, 0.2, 3.0],
            [0.3, 0.1, 2.0],
            [0.2, 0.4, 4.0],
            [0.5, 0.2, 5.0],
            [0.4, 0.3, 6.0],
        ]
    )
    embedding_values = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ]
    )

    shap.plots.embedding(
        "rank(0)",
        shap_values,
        feature_names=["f0", "f1", "f2"],
        method=embedding_values,
        show=False,
    )

    fig = plt.gcf()
    assert len(fig.axes) == 2
    assert fig.axes[1].get_ylabel() == "SHAP value for\nf2"


def test_embedding_pca_path_runs_and_creates_scatter():
    shap_values = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.3, 0.1, 0.2],
            [0.2, 0.4, 0.1],
            [0.5, 0.2, 0.2],
            [0.4, 0.3, 0.1],
        ]
    )

    shap.plots.embedding(0, shap_values, feature_names=["f0", "f1", "f2"], method="pca", show=False)

    fig = plt.gcf()
    main_ax = fig.axes[0]
    assert len(main_ax.collections) == 1
    assert len(fig.axes) == 2


def test_embedding_unsupported_method_prints_message_and_raises(capsys):
    shap_values = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.3, 0.1, 0.2],
            [0.2, 0.4, 0.1],
            [0.5, 0.2, 0.2],
            [0.4, 0.3, 0.1],
        ]
    )

    with pytest.raises(UnboundLocalError):
        shap.plots.embedding(0, shap_values, feature_names=["f0", "f1", "f2"], method="bad", show=False)

    captured = capsys.readouterr()
    assert "Unsupported embedding method: bad" in captured.out


def test_embedding_show_true_calls_plt_show(monkeypatch):
    shap_values = np.array(
        [
            [0.1, 0.2, 0.3],
            [0.3, 0.1, 0.2],
            [0.2, 0.4, 0.1],
            [0.5, 0.2, 0.2],
            [0.4, 0.3, 0.1],
        ]
    )
    embedding_values = np.array(
        [
            [0.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [3.0, 3.0],
            [4.0, 4.0],
        ]
    )

    calls = {"n": 0}

    def fake_show():
        calls["n"] += 1

    monkeypatch.setattr(plt, "show", fake_show)

    shap.plots.embedding(0, shap_values, feature_names=["f0", "f1", "f2"], method=embedding_values, show=True)
    assert calls["n"] == 1
