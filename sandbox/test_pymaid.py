#%%
from pkg.data import load_unmatched
from pkg.plot.neuron import simple_plot_neurons
from pkg.pymaid import start_instance
import pymaid

start_instance()


left_adj, left_nodes = load_unmatched("left")

print(pymaid.get_names(left_nodes.index.values))
