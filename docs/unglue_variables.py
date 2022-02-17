#%%
import ast
import json
from glob import glob

import nbformat as nbf

# Collect a list of all notebooks in the content folder
loc = "bilateral-connectome/docs/**/*.ipynb"

notebooks = glob(loc, recursive=True)
# HACK what is the globby way to do this?
notebooks = [n for n in notebooks if "_build" not in n]

data_key = "application/papermill.record/text/plain"
image_key = "application/papermill.record/image/png"

variables = {}
for notebook_path in notebooks:
    notebook = nbf.read(notebook_path, nbf.NO_CONVERT)
    for cell in notebook.cells:
        if cell.get("cell_type") == "code":
            outputs = cell.get("outputs")
            for output in outputs:
                if "data" in output:
                    if (image_key not in output["data"]) and (
                        "image/svg+xml" not in output["data"]
                    ):
                        value = output["data"][data_key]
                        try:
                            value = ast.literal_eval(value)
                        except:
                            pass
                        name = output["metadata"]["scrapbook"]["name"]
                        variables[name] = value

with open("bilateral-connectome/docs/glued_variables.json", "w") as f:
    json.dump(variables, f, indent=4)
