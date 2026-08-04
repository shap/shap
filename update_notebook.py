import nbformat

nb_path = "notebooks/tabular_examples/model_agnostic/Simple Kernel SHAP.ipynb"
with open(nb_path) as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code":
        if "explainer.shap_values(x)" in cell.source:
            cell.source = cell.source.replace("explainer.shap_values(x)", "explainer(x)")
            cell.source = cell.source.replace(
                'print("shap_values =", shap_values)\nprint("base value =", explainer.expected_value)',
                "print(shap_values)",
            )

with open(nb_path, "w") as f:
    nbformat.write(nb, f)
