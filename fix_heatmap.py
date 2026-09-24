import nbformat
from nbclient import NotebookClient

nb_path = "notebooks/api_examples/plots/heatmap.ipynb"
with open(nb_path) as f:
    nb = nbformat.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == "code":
        if "xgboost.XGBClassifier" in cell.source:
            if "random_state" not in cell.source:
                cell.source = cell.source.replace("max_depth=2", "max_depth=2, random_state=0")

client = NotebookClient(nb, timeout=600, kernel_name="python3")
try:
    client.execute()
    with open(nb_path, "w") as f:
        nbformat.write(nb, f)
    print("Successfully updated and executed notebook")
except Exception as e:
    print(f"Error executing notebook: {e}")
