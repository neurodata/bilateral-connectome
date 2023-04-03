#%%
from pkg.data import DATA_PATH
import pandas as pd


adj = pd.read_csv(
    DATA_PATH / "science" / "Supplementary-Data-S1" / "all-all_connectivity_matrix.csv", index_col=0
)
adj = adj.values

adj.sum() / adj.shape[0]

#%%
adj.sum()