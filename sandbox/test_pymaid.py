#%%
from pkg.data import load_unmatched
from pkg.plot.neuron import simple_plot_neurons
from pkg.pymaid import start_instance
import pymaid

start_instance()


left_adj, left_nodes = load_unmatched("left")

pymaid.get_names(left_nodes.index.values)

#%%

pickle_path = "bilateral-connectome/data/2021-05-24-v2/neurons.pickle"

import pickle

with open(pickle_path, "rb") as f:
    neurons = pickle.load(f)

#%%
simple_plot_neurons(neurons[1])
