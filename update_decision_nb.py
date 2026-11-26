import nbformat

nb_path = "notebooks/api_examples/plots/decision_plot.ipynb"
with open(nb_path) as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code":
        # Replace explainer.shap_values(X)[1] with explainer(X)
        if "explainer.shap_values" in cell.source:
            cell.source = cell.source.replace(
                "shap_values = explainer.shap_values(features)[1]", "shap_values = explainer(features)"
            )
            cell.source = cell.source.replace(
                "hypothetical_shap_values = explainer.shap_values(R)[1]", "hypothetical_shap_values = explainer(R)"
            )
            cell.source = cell.source.replace("sh = explainer.shap_values(T)[1]", "sh = explainer(T)")

        # Replace shap.decision_plot(...) with the modern API
        if "shap.decision_plot" in cell.source:
            cell.source = cell.source.replace(
                "shap.decision_plot(expected_value, shap_values, features_display)",
                "shap.plots.decision(shap_values, features=features_display)",
            )
            cell.source = cell.source.replace(
                "shap.decision_plot(expected_value, shap_values, features_display, link='logit')",
                "shap.plots.decision(shap_values, features=features_display, link='logit')",
            )
            cell.source = cell.source.replace(
                "shap.decision_plot(expected_value, shap_values, features_display, highlight=misclassified)",
                "shap.plots.decision(shap_values, features=features_display, highlight=misclassified)",
            )
            cell.source = cell.source.replace(
                "shap.decision_plot(expected_value, shap_values, features_display, return_objects=True)",
                "shap.plots.decision(shap_values, features=features_display, return_objects=True)",
            )
            cell.source = cell.source.replace(
                "shap.decision_plot(expected_value, shap_values, features_display, feature_order='hclust')",
                "shap.plots.decision(shap_values, features=features_display, feature_order='hclust')",
            )
            cell.source = cell.source.replace(
                "shap.decision_plot(expected_value, shap_values, features_display, feature_order='hclust', feature_display_range=slice(None, -11, -1))",
                "shap.plots.decision(shap_values, features=features_display, feature_order='hclust', feature_display_range=slice(None, -11, -1))",
            )
            cell.source = cell.source.replace(
                "shap.decision_plot(expected_value, shap_values[m], features_display[m], highlight=0)",
                "shap.plots.decision(shap_values[m], features=features_display[m], highlight=0)",
            )
            cell.source = cell.source.replace(
                "shap.decision_plot(expected_value, shap_values[m], features_display[m])",
                "shap.plots.decision(shap_values[m], features=features_display[m])",
            )
            cell.source = cell.source.replace(
                "shap.decision_plot(expected_value, hypothetical_shap_values, R.iloc[:,:-1])",
                "shap.plots.decision(hypothetical_shap_values, features=R.iloc[:,:-1])",
            )
            cell.source = cell.source.replace(
                "shap.decision_plot(expected_value, sh, T, feature_order='hclust', return_objects=True)",
                "shap.plots.decision(sh, features=T, feature_order='hclust', return_objects=True)",
            )
            cell.source = cell.source.replace(
                "shap.decision_plot(expected_value, sh, T, feature_order=r.feature_idx)",
                "shap.plots.decision(sh, features=T, feature_order=r.feature_idx)",
            )
            cell.source = cell.source.replace(
                "shap.decision_plot(expected_value, sh, T, feature_order=r.feature_idx, feature_display_range=slice(None, -21, -1))",
                "shap.plots.decision(sh, features=T, feature_order=r.feature_idx, feature_display_range=slice(None, -21, -1))",
            )
            cell.source = cell.source.replace(
                "shap.decision_plot(expected_value, sh, T, feature_order=r.feature_idx, feature_display_range=slice(None, -11, -1))",
                "shap.plots.decision(sh, features=T, feature_order=r.feature_idx, feature_display_range=slice(None, -11, -1))",
            )

with open(nb_path, "w") as f:
    nbformat.write(nb, f)
