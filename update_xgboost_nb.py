import re

import nbformat

nb_path = "notebooks/tabular_examples/tree_based_models/Census income classification with XGBoost.ipynb"
with open(nb_path) as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code":
        source = cell.source

        # 1. Update explainer.shap_values(X) -> explainer(X)
        source = source.replace(
            "shap_values = explainer.shap_values(X)",
            "shap_values = explainer(X)\nshap_values.display_data = X_display.values",
        )

        # 2. Update shap.force_plot(...)
        source = source.replace(
            "shap.force_plot(explainer.expected_value, shap_values[0, :], X_display.iloc[0, :])",
            "shap.plots.force(shap_values[0])",
        )
        source = source.replace(
            "shap.force_plot(explainer.expected_value, shap_values[:1000, :], X_display.iloc[:1000, :])",
            "shap.plots.force(shap_values[:1000])",
        )

        # 3. Update shap.summary_plot -> shap.plots.bar and shap.plots.beeswarm
        source = source.replace(
            'shap.summary_plot(shap_values, X_display, plot_type="bar")', "shap.plots.bar(shap_values)"
        )
        source = source.replace("shap.summary_plot(shap_values, X)", "shap.plots.beeswarm(shap_values)")

        # 4. Update shap.dependence_plot -> shap.plots.scatter
        # e.g., "shap.dependence_plot(name, shap_values, X, display_features=X_display)"
        source = re.sub(
            r"shap\.dependence_plot\(name, shap_values, X, display_features=X_display\)",
            r"shap.plots.scatter(shap_values[:, name])",
            source,
        )

        # 5. Update model_ind explainer
        # "shap_values_ind = shap.TreeExplainer(model_ind).shap_values(X)"
        if "shap_values_ind = shap.TreeExplainer(model_ind).shap_values(X)" in source:
            source = source.replace(
                "shap_values_ind = shap.TreeExplainer(model_ind).shap_values(X)",
                "explainer_ind = shap.Explainer(model_ind, X)\nshap_values_ind = explainer_ind(X)\nshap_values_ind.display_data = X_display.values",
            )

        # 6. Update shap.dependence_plot for model_ind
        # "shap.dependence_plot(name, shap_values_ind, X, display_features=X_display)"
        source = re.sub(
            r"shap\.dependence_plot\(name, shap_values_ind, X, display_features=X_display\)",
            r"shap.plots.scatter(shap_values_ind[:, name])",
            source,
        )

        cell.source = source

with open(nb_path, "w") as f:
    nbformat.write(nb, f)
