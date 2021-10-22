jupytext --to notebook --output bilateral-connectome/docs/$1.ipynb bilateral-connectome/scripts/$1.py
jupyter nbconvert --to notebook --inplace --execute --ExecutePreprocessor.timeout=-1 bilateral-connectome/docs/$1.ipynb 
python bilateral-connectome/docs/add_cell_tags.py bilateral-connectome/docs/$1.ipynb
# {'metadata': {'path': run_path}}
# https://github.com/jupyter/nbconvert/blob/7ee82983a580464b0f07c68e35efbd5a0175ff4e/nbconvert/preprocessors/execute.py#L63
# --ExecutePreprocessor.record_timing=True