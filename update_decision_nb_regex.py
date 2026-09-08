import re

import nbformat

nb_path = "notebooks/api_examples/plots/decision_plot.ipynb"
with open(nb_path) as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code":
        # Many calls span multiple lines. Replace `shap.decision_plot(\n    expected_value, `
        # with `shap.plots.decision(\n    `

        # Replace `shap.decision_plot(` with `shap.plots.decision(`
        cell.source = cell.source.replace("shap.decision_plot(", "shap.plots.decision(")

        # Remove expected_value from arguments in cases where we pass shap_values as an Explanation
        cell.source = re.sub(
            r"shap\.plots\.decision\(\s*expected_value,\s*shap_values", "shap.plots.decision(shap_values", cell.source
        )
        cell.source = re.sub(
            r"shap\.plots\.decision\(\s*expected_value,\s*hypothetical_shap_values",
            "shap.plots.decision(hypothetical_shap_values",
            cell.source,
        )
        cell.source = re.sub(r"shap\.plots\.decision\(\s*expected_value,\s*sh", "shap.plots.decision(sh", cell.source)
        cell.source = re.sub(
            r"shap\.plots\.decision\(\s*expected_value,\s*shap_interaction_values",
            "shap.plots.decision(shap_interaction_values",
            cell.source,
        )

        # For legacy cells where expected_value is not an explanation (e.g. shap_interaction_values)
        # we can just pass them as before. Wait, if it's shap.plots.decision(expected_value, shap_interaction_values) that's actually fine because the signature supports it!
        # But if it was replaced to shap.plots.decision(shap_interaction_values), shap_interaction_values is an ndarray, so base_value=ndarray, and shap_values=None. It would fail!
        # Let's revert shap_interaction_values to use expected_value.
        cell.source = cell.source.replace(
            "shap.plots.decision(shap_interaction_values", "shap.plots.decision(expected_value, shap_interaction_values"
        )

        # In cases where we pass 'features' but forgot the kwargs:
        # e.g. shap.plots.decision(shap_values, features_display) -> shap.plots.decision(shap_values, features=features_display)
        # We can just leave them as positional args if they are the 3rd argument? NO.
        # Wait, the signature is:
        # def decision(base_value, shap_values=None, features=None)
        # If we do shap.plots.decision(explanation, features_display),
        # base_value = explanation, shap_values=features_display (WRONG!)
        # So we MUST pass features as a kwarg if we pass explanation!

        # We need to make sure any `shap.plots.decision(sh, T)` becomes `shap.plots.decision(sh, features=T)`
        cell.source = re.sub(
            r"shap\.plots\.decision\(shap_values, features_display",
            "shap.plots.decision(shap_values, features=features_display",
            cell.source,
        )
        cell.source = re.sub(r"shap\.plots\.decision\(sh, T", "shap.plots.decision(sh, features=T", cell.source)
        cell.source = re.sub(
            r"shap\.plots\.decision\(hypothetical_shap_values, R\.iloc\[:,:-1\]",
            "shap.plots.decision(hypothetical_shap_values, features=R.iloc[:,:-1]",
            cell.source,
        )
        cell.source = re.sub(
            r"shap\.plots\.decision\(shap_values\[m\], features_display\[m\]",
            "shap.plots.decision(shap_values[m], features=features_display[m]",
            cell.source,
        )

with open(nb_path, "w") as f:
    nbformat.write(nb, f)
