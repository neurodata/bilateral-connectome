#%% [markdown]
# # this is a test
#%%
from pkg.io import get_environment_variables

_, RERUN_SIMS, DISPLAY_FIGS = get_environment_variables()

print(RERUN_SIMS)
print(DISPLAY_FIGS)