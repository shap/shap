import matplotlib.pyplot as plt
import numpy as np
import shap

# PR #4922: Ax support in partial_dependence_plot
# ---------------------------------------------------------------------------


def test_pdp_respects_custom_ax():
    """
    Verify that partial_dependence_plot draws onto the provided Axes
    instead of creating a new figure.
    """
    X, y = shap.datasets.adult(n_points=50)
    
    # Simple model function that returns the sum of features
    # This is more stable than using internal SHAP utility classes
    model_func = lambda x: np.array(x).sum(axis=1)

    fig, ax = plt.subplots()
    initial_artists = len(ax.get_children())

    # pdp internally might use different plot wrappers, we check if it populates the ax
    shap.plots.partial_dependence(
        "Age", model_func, X, ice=False, model_expected_value=True, feature_expected_value=True, show=False, ax=ax
    )

    # If artists were added to the axes, it means the plot was drawn on our supplied ax
    assert len(ax.get_children()) > initial_artists, "PDP failed to draw artists on the user-supplied Axes."
    plt.close(fig)
