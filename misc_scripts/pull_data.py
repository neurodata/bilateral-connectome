#%%
import json
import pprint
import sys
import time
from collections import defaultdict
from datetime import timedelta
from pathlib import Path
from pickle import dump

import matplotlib.pyplot as plt
import navis
import networkx as nx
import numpy as np
import pandas as pd
import pymaid
from pkg.pymaid import start_instance
from requests.exceptions import ChunkedEncodingError

t0 = time.time()

start_instance()

OUTPUT_PATH = Path("bilateral-connectome/data/2022-09-25")
pair_path = Path("bilateral-connectome/data/2022-09-25/pairs-2022-02-14.csv")


class Logger(object):
    # REF: https://stackoverflow.com/questions/4675728/redirect-stdout-to-a-file-in-python
    def __init__(self):
        self.terminal = sys.stdout
        self.log = open(OUTPUT_PATH / "log.txt", "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass

    def close(self):
        self.log.close()


sys.stdout = Logger()

#%%


def get_single_class(classes):
    single_class = classes[0]
    for c in classes[1:]:
        single_class += ";" + c
    return str(single_class)


def get_classes(meta, class_cols, fill_unk=False, priority_map=None):
    all_class = []
    single_class = []
    n_class = []
    for index, row in meta.iterrows():
        classes = class_cols[row[class_cols].astype(bool)]
        all_class.append(str(classes))
        n_class.append(int(len(classes)))
        if len(classes) > 0:
            if priority_map is not None:
                priorities = np.vectorize(priority_map.__getitem__)(classes)
                inds = np.where(priorities == priorities.min())[0]
                sc = get_single_class(classes[inds])
            else:
                sc = get_single_class(classes)
        else:
            if fill_unk:
                sc = "unk"
            else:
                sc = ""
        single_class.append(sc)
    return single_class, all_class, n_class


def get_indicator_from_annotation(annot_name, filt=None):
    ids = pymaid.get_skids_by_annotation(annot_name.replace("*", "\*"))
    if filt is not None:
        name = filt(annot_name)
    else:
        name = annot_name
    indicator = pd.Series(
        index=ids, data=np.ones(len(ids), dtype=bool), name=name, dtype=bool
    )
    return indicator


def df_from_meta_annotation(key, filt=None):
    print(f"Applying annotations under meta annotation {key}...")
    annot_df = pymaid.get_annotated(key)

    series_ids = []

    for annot_name in annot_df["name"]:
        print("\t" + annot_name)
        indicator = get_indicator_from_annotation(annot_name, filt=filt)
        series_ids.append(indicator)
    return pd.concat(series_ids, axis=1, ignore_index=False)


def apply_annotation(key, meta, new_key=None, filt=None):
    ids = pymaid.get_skids_by_annotation(key.replace("*", "\*"))
    if filt is not None:
        name = filt(key)
    else:
        name = key

    ids = np.intersect1d(ids, meta.index)
    meta.loc[ids, name] = True
    print(f"{len(ids)} neurons annotated by {key}.\n")


def apply_any_from_meta_annotation(key, meta, new_key=None, filt=None):
    if new_key is None:
        new_key = key
    group_meta = df_from_meta_annotation(key, filt=filt)
    group_meta = group_meta[group_meta.index.isin(meta.index)]
    is_in_group = group_meta.any(axis=1)
    meta.loc[is_in_group.index, new_key] = is_in_group
    print(f"{is_in_group.sum()} neurons annotated under meta annotation {key}.\n")


def filt(name):
    name = str(name)
    name = name.replace("mw brain ", "")
    name = name.replace("mw ", "")
    name = name.replace(" ", "_")
    return name


# %%
#%%
# get all of the neurons we may ever have to consider
all_neurons = get_indicator_from_annotation("mw brain paper all neurons").index
meta = pd.DataFrame(index=all_neurons)
print(f"\n{len(meta)} neurons in 'brain paper all neurons'.\n")

name_map = pymaid.get_names(meta.index.values)
meta["name"] = meta.index.map(lambda name: name_map[str(name)])

#%%

single_annotations = [
    "mw brain neurons",
    "mw brain paper clustered neurons",
    "mw left",
    "mw right",
    "mw center",
    "mw sink",
    "mw partially differentiated",
    "mw unsplittable",
    "mw ipsilateral axon",
    "mw contralateral axon",
    "mw bilateral axon",
    "mw brain incomplete",
    "mw MBON special-cases",
    "mw brain accessory neurons",
    "mw duplicated neurons",
    "mw developmental defect",
    "mw missing axon",
    "mw missing dendrite",
    "mw duplicated neurons to delete",
]


for annotation in single_annotations:
    print(f"Applying single annotation {annotation}...")
    n = apply_annotation(annotation, meta, filt=filt)
meta.fillna(False, inplace=True)

#%%
# pymaid.get_skids_by_annotation('mw brain celltypes discrete')

#%%
print("Adding axon laterality category...")

n_axon_lat = meta[["ipsilateral_axon", "contralateral_axon", "bilateral_axon"]].sum(
    axis=1
)
if n_axon_lat.max() > 1:
    print("Some neurons have more than 1 axon laterality category:")
    print(meta[n_axon_lat > 1])

meta["axon_lat"] = "unk"
meta.loc[meta[meta["ipsilateral_axon"]].index, "axon_lat"] = "ipsi"
meta.loc[meta[meta["contralateral_axon"]].index, "axon_lat"] = "contra"
meta.loc[meta[meta["bilateral_axon"]].index, "axon_lat"] = "bi"
print(f"{(meta['axon_lat'] == 'unk').sum()} neurons have no axon laterality category.")
print()

#%%

apply_any_from_meta_annotation("mw brain outputs", meta, new_key="outputs", filt=filt)
apply_any_from_meta_annotation("mw brain inputs", meta, new_key="inputs", filt=filt)
apply_any_from_meta_annotation(
    "mw brain A1 ascending", meta, new_key="a1_ascending", filt=filt
)
apply_any_from_meta_annotation("mw brain sensories", meta, new_key="sensory", filt=filt)
# TODO this one is erroring
# apply_any_from_meta_annotation(
#     "mw A1 neurons paired", meta, new_key="a1_paired", filt=filt
# )
apply_any_from_meta_annotation("mw A1 sensories", meta, new_key="a1_sensory", filt=filt)
apply_any_from_meta_annotation(
    "mw brain paper reconstruction incomplete",
    meta,
    new_key="reconstruction_incomplete",
    filt=filt,
)
meta.fillna(False, inplace=True)

print("Adding io category...")
meta["io"] = "inter"
meta.loc[meta[meta["inputs"]].index, "io"] = "input"
meta.loc[meta[meta["outputs"]].index, "io"] = "output"
print()

print("Pulling priority map...")
priority_map = defaultdict(lambda: np.inf)
priority_df = pymaid.get_annotated("mw brain simple priorities")
for meta_annotation in priority_df["name"].values:
    priority = int(meta_annotation.split(" ")[-1])
    annotations = list(pymaid.get_annotated(meta_annotation)["name"].values)
    for annotation in annotations:
        priority_map[filt(annotation)] = priority

print("Priority mapping;")
pprint.pprint(priority_map)
print("\n")


# simple groups
annot_df = pymaid.get_annotated("mw brain simple groups")
meta_annotations = annot_df["name"].values
for meta_annotation in meta_annotations:
    apply_any_from_meta_annotation(
        meta_annotation, meta, new_key=filt(meta_annotation), filt=filt
    )
    print("\n")


#%%

df = df_from_meta_annotation("mw brain celltypes discrete", filt).fillna(False)
row_inds, column_inds = np.nonzero(df.values)
column_names = df.columns.values
column_names = np.array([s[4:] for s in column_names])
celltype_discrete = column_names[column_inds]
celltype_discrete = pd.Series(
    index=df.index.values, data=celltype_discrete, name="celltype_discrete"
)
celltype_discrete = celltype_discrete.reindex(meta.index).fillna("other")
meta = pd.concat((meta, celltype_discrete), axis=1)

#%%
print("Condensing simple labels...")
meta.fillna(False, inplace=True)

simple_group_cols = np.array(list(map(filt, annot_df["name"].values)))
simple_groups, all_simple_groups, n_simple_groups = get_classes(
    meta, simple_group_cols, fill_unk=True, priority_map=priority_map
)

meta["simple_group"] = simple_groups
meta["all_simple_groups"] = all_simple_groups
meta["n_simple_groups"] = n_simple_groups
print()

#%%
print("Applying colors...")
meta["color"] = "#ACAAC8"  # grey for unk
vec_filter = np.vectorize(filt)
colors = pymaid.get_annotated("mw brain simple colors")["name"].values
color_map = {"unk": "#ACAAC8"}
for color in colors:
    groups = vec_filter(pymaid.get_annotated(color)["name"])
    group_idx = meta[meta["simple_group"].isin(groups)].index
    meta.loc[group_idx, "color"] = color
    for g in groups:
        color_map[g] = color
print("Found color map:")
pprint.pprint(color_map)
print()

# %% [markdown]
# ##
print("Simple class unique values before priority mapping:")
pprint.pprint(dict(zip(*np.unique(all_simple_groups, return_counts=True))))
print()

print("Simple class unique values after priority mapping:")
pprint.pprint(dict(zip(*np.unique(simple_groups, return_counts=True))))
print()

# %% [markdown]
# ## Hemisphere
meta["hemisphere"] = "unk"  # default is unknown
left_meta = meta[meta["left"]]
meta.loc[left_meta.index, "hemisphere"] = "L"
right_meta = meta[meta["right"]]
meta.loc[right_meta.index, "hemisphere"] = "R"
right_meta = meta[meta["center"]]
meta.loc[right_meta.index, "hemisphere"] = "C"

print("\n")
print("Center neurons:")
missing_hemisphere = meta[meta["hemisphere"] == "unk"][
    ["name", "simple_group", "hemisphere"]
]
missing_hemisphere.to_csv(OUTPUT_PATH / "no_hemisphere.csv")
print(missing_hemisphere)
print("\n")


# %% [markdown]
# # Pairs

# Pairs (NOTE this file has some issues where some ids are repeated in multiple pairs)
pair_df = pd.read_csv(pair_path, usecols=range(2))
pair_df["pair_id"] = range(len(pair_df))
pair_df["pair_id"] += 2  # to match original file

meta["pair"] = -1
meta["pair_id"] = -1

uni_left, left_counts = np.unique(pair_df["leftid"], return_counts=True)
uni_right, right_counts = np.unique(pair_df["rightid"], return_counts=True)

dup_left_inds = np.where(left_counts != 1)[0]
dup_right_inds = np.where(right_counts != 1)[0]
dup_left_ids = uni_left[dup_left_inds]
dup_right_ids = uni_right[dup_right_inds]

print("\n")
if len(dup_left_inds) > 0:
    print("Duplicate pairs left:")
    print(dup_left_ids)
if len(dup_right_inds) > 0:
    print("Duplicate pairs right:")
    print(dup_right_ids)
print("\n")

duplicate_df = pair_df[
    pair_df["leftid"].isin(dup_left_ids) | pair_df["rightid"].isin(dup_right_ids)
]

ok_duplicates = meta[meta["duplicated_neurons"]].index
delete_duplicates = meta[meta["duplicated_neurons_to_delete"]].index

drop_pair_ids = []
for idx, row in duplicate_df.iterrows():
    left_id = row["leftid"]
    right_id = row["rightid"]
    if left_id not in ok_duplicates and right_id not in ok_duplicates:
        print(f"{list(row)} is an invalid duplicate pairing.")

    if left_id in delete_duplicates or right_id in delete_duplicates:
        drop_pair_ids.append(idx)


drop_df = pair_df[pair_df.index.isin(drop_pair_ids)]

pair_df.drop(drop_pair_ids, axis=0, inplace=True)

pair_ids = np.concatenate((pair_df["leftid"].values, pair_df["rightid"].values))
meta_ids = meta.index.values
in_meta_ids = np.isin(pair_ids, meta_ids)
drop_ids = pair_ids[~in_meta_ids]
print("\n")
print("Pairs not in meta:")
print(drop_ids)
print()
pair_df = pair_df[~pair_df["leftid"].isin(drop_ids)]
pair_df = pair_df[~pair_df["rightid"].isin(drop_ids)]

left_to_right_df = pair_df.set_index("leftid")
right_to_left_df = pair_df.set_index("rightid")

meta.loc[left_to_right_df.index, "pair"] = left_to_right_df["rightid"]
meta.loc[right_to_left_df.index, "pair"] = right_to_left_df["leftid"]

meta.loc[left_to_right_df.index, "pair_id"] = left_to_right_df["pair_id"]
meta.loc[right_to_left_df.index, "pair_id"] = right_to_left_df["pair_id"]

for idx, row in drop_df.iterrows():
    left_id = row["leftid"]
    right_id = row["rightid"]
    pair_id = row["pair_id"]
    if left_id in delete_duplicates:
        meta.loc[left_id, "pair"] = -right_id
        meta.loc[left_id, "pair_id"] = -pair_id
    if right_id in delete_duplicates:
        meta.loc[right_id, "pair"] = -left_id
        meta.loc[right_id, "pair_id"] = -pair_id

#%% Fix places where L/R labels are not the same
print("\nFinding asymmetric L/R labels")
for i in range(len(meta)):
    my_id = meta.index[i]
    my_class = meta.loc[my_id, "simple_group"]
    partner_id = meta.loc[my_id, "pair"]
    if partner_id > 1:
        partner_class = meta.loc[partner_id, "simple_group"]
        if partner_class != "unk" and my_class == "unk":
            print(f"{my_id} had asymmetric class label {partner_class}, fixed")
            meta.loc[my_id, "simple_group"] = partner_class
        elif (partner_class != my_class) and (partner_class != "unk"):
            msg = (
                f"{meta.index[i]} and partner {partner_id} have different labels"
                + f", labels are {my_class}, {partner_class}"
            )
            print(msg)
print()


# %% [markdown]
# ## Load lineages


def filt(string):
    string = string.replace("akira", "")
    string = string.replace("Lineage", "")
    string = string.replace("lineage", "")
    string = string.replace("*", "")
    string = string.strip("_")
    string = string.strip(" ")
    # string = string.replace("_r", "")
    # string = string.replace("_l", "")
    string = string.replace("right", "")
    string = string.replace("left", "")
    string = string.replace("unknown", "unk")
    return string


lineage_df = df_from_meta_annotation("Volker", filt=filt)

lineage_df = lineage_df.fillna(False)
data = lineage_df.values
row_sums = data.sum(axis=1)
lineage_df.loc[row_sums > 1, :] = False
check_row_sums = lineage_df.values.sum(axis=1)
assert check_row_sums.max() == 1

columns = lineage_df.columns
lineages = []
for index, row in lineage_df.iterrows():
    lineage = columns[row].values
    if len(lineage) < 1:
        lineage = "unk"
    else:
        lineage = lineage[0]
    lineages.append(lineage)
lineage_series = pd.Series(index=lineage_df.index, data=lineages)
lineage_series = lineage_series[lineage_series.index.isin(meta.index)]
meta["lineage"] = "unk"
meta.loc[lineage_series.index, "lineage"] = lineage_series.values


#%%

print("\nChecking for rows in node metadata with Nan values...")
missing_na = []
nan_df = meta[meta.isna().any(axis=1)]
for row in nan_df.index:
    na_ind = nan_df.loc[row].isna()
    print(nan_df.loc[row][na_ind])
    missing_na.append(row)
print()
print("These skeletons have missing values in the metadata:")
print(missing_na)
print("\n")

#%%
print("Saving metadata as csv...")
meta.to_csv(OUTPUT_PATH / "meta_data.csv")
meta.to_csv(OUTPUT_PATH / "meta_data_unmodified.csv")

#%%
print("Pulling neurons...\n")

ids = meta.index.values
ids = [int(i) for i in ids]

batch_size = 100
max_tries = 5
n_batches = int(np.floor(len(ids) / batch_size))
if len(ids) % n_batches > 0:
    n_batches += 1
print(f"Batch size: {batch_size}")
print(f"Number of batches: {n_batches}")
print(f"Number of neurons: {len(ids)}")
print(f"Batch product: {n_batches * batch_size}\n")

i = 0
currtime = time.time()
nl = pymaid.get_neuron(
    ids[i * batch_size : (i + 1) * batch_size], with_connectors=False
)
print(f"{time.time() - currtime:.3f} seconds elapsed for batch {i}.")
for i in range(1, n_batches):
    currtime = time.time()
    n_tries = 0
    success = False
    while not success and n_tries < max_tries:
        try:
            nl += pymaid.get_neuron(
                ids[i * batch_size : (i + 1) * batch_size], with_connectors=False
            )
            success = True
        except ChunkedEncodingError:
            print(f"Failed pull on batch {i}, trying again...")
            n_tries += 1
    print(f"{time.time() - currtime:.3f} seconds elapsed for batch {i}.")

print("\nPulled all neurons.\b")

#%%
print("\nPickling neurons...")
currtime = time.time()

with open(OUTPUT_PATH / "neurons.pickle", "wb") as f:
    dump(nl, f)
print(f"{time.time() - currtime:.3f} seconds elapsed to pickle.")


#%%


def get_connectors(nl):
    connectors = pymaid.get_connectors(nl)
    connectors.set_index("connector_id", inplace=True)
    connectors.drop(
        [
            "confidence",
            "creation_time",
            "edition_time",
            "tags",
            "creator",
            "editor",
            "type",
        ],
        inplace=True,
        axis=1,
    )
    details = pymaid.get_connector_details(connectors.index.values)
    details.set_index("connector_id", inplace=True)
    connectors = pd.concat((connectors, details), ignore_index=False, axis=1)
    connectors.reset_index(inplace=True)
    return connectors


#%%
print("Pulling split points and special split neuron ids...")
currtime = time.time()

splits = pymaid.find_nodes(tags="mw axon split")
splits = splits.set_index("skeleton_id")["node_id"].squeeze()

# find all of the neurons under "mw MBON special-cases"
# split the neuron based on node-tags "mw axon start" and "mw axon end"
# axon is anything that is in between "mw axon start" and "mw axon end"
special_ids = [
    lst[0]
    for lst in pymaid.get_annotated("mw MBON special-cases")["skeleton_ids"].values
]

print(f"{time.time() - currtime:.3f} elapsed.\n")

# any neuron that is not brain incomplete, unsplittable, or partially differentiated
# and does not have a split tag should throw an error

# get the neurons that SHOULD have splits
should_split_meta = meta[
    ~meta["unsplittable"]
    & ~meta["partially_differentiated"]
    & ~meta["incomplete"]
    & ~meta["MBON_special-cases"]
]
didnt_split_meta = should_split_meta[~should_split_meta.index.isin(splits.index)]
if len(didnt_split_meta) > 0:
    print(
        f"WARNING: {len(didnt_split_meta)} neurons should have had split tag and didn't:"
    )
    print(didnt_split_meta.index.values)


#%%


def _append_labeled_nodes(add_list, nodes, name):
    for node in nodes:
        add_list.append({"node_id": node, "node_type": name})


def _standard_split(n, treenode_info, splits):
    skid = int(n.skeleton_id)
    split_node = splits[skid]
    # order of output is axon, dendrite
    fragments = navis.cut_skeleton(n, split_node)

    # axon(s)
    for f in fragments[:-1]:
        axon_treenodes = f.nodes.node_id.values
        _append_labeled_nodes(treenode_info, axon_treenodes, "axon")

    # dendrite
    dendrite = fragments[-1]
    tags = dendrite.tags
    if "mw periphery" not in tags and "soma" not in tags:
        msg = f"WARNING: when splitting neuron {skid} ({n.name}), no soma or periphery tag was found on dendrite fragment."
        raise UserWarning(msg)
        print("Whole neuron tags:")
        pprint.pprint(n.tags)
        print("Axon fragment tags:")
        for f in fragments[:-1]:
            pprint.pprint(f.tags)
    dend_treenodes = fragments[-1].nodes.node_id.values
    _append_labeled_nodes(treenode_info, dend_treenodes, "dendrite")


def _special_mbon_split(n, treenode_info):
    skid = int(n.skeleton_id)
    axon_starts = list(
        pymaid.find_nodes(tags="mw axon start", skeleton_ids=skid)["node_id"]
    )
    axon_ends = list(
        pymaid.find_nodes(tags="mw axon end", skeleton_ids=skid)["node_id"]
    )
    axon_splits = axon_starts + axon_ends

    fragments = navis.cut_skeleton(n, axon_splits)
    axons = []
    dendrites = []
    for fragment in fragments:
        root = fragment.root
        if "mw axon start" in fragment.tags and root in fragment.tags["mw axon start"]:
            axons.append(fragment)
        elif "mw axon end" in fragment.tags and root in fragment.tags["mw axon end"]:
            dendrites.append(fragment)
        elif "soma" in fragment.tags and root in fragment.tags["soma"]:
            dendrites.append(fragment)
        else:
            raise UserWarning(
                f"Something weird happened when splitting special neuron {skid}"
            )

    for a in axons:
        axon_treenodes = a.nodes.node_id.values
        _append_labeled_nodes(treenode_info, axon_treenodes, "axon")

    for d in dendrites:
        dendrite_treenodes = d.nodes.node_id.values
        _append_labeled_nodes(treenode_info, dendrite_treenodes, "dendrite")

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    navis.plot2d(axons, color="red", ax=ax)
    navis.plot2d(dendrites, color="blue", ax=ax)
    plt.savefig(
        OUTPUT_PATH / f"weird-mbon-{skid}.png",
        facecolor="w",
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.3,
        dpi=300,
    )

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection="3d")
    navis.plot2d(axons, color="red", ax=ax)
    navis.plot2d(dendrites, color="blue", ax=ax)
    ax.azim = -90
    ax.elev = 0
    plt.savefig(
        OUTPUT_PATH / f"weird-mbon-{skid}-top.png",
        facecolor="w",
        transparent=False,
        bbox_inches="tight",
        pad_inches=0.3,
        dpi=300,
    )


def get_treenode_types(nl, splits, special_ids):
    treenode_info = []
    print("Cutting neurons...")
    for i, n in enumerate(nl):
        skid = int(n.skeleton_id)

        if skid in special_ids:
            _special_mbon_split(n, treenode_info)
        elif skid in splits.index:
            _standard_split(n, treenode_info, splits)
        else:  # unsplittable neuron
            # TODO explicitly check that these are unsplittable
            unsplit_treenodes = n.nodes.node_id.values
            _append_labeled_nodes(treenode_info, unsplit_treenodes, "unsplit")

    treenode_df = pd.DataFrame(treenode_info)
    # a split node is included in pre and post synaptic fragments
    # here i am just removing, i hope there is never a synapse on that node...
    # NOTE: would probably throw an error later if there was
    treenode_df = treenode_df[~treenode_df["node_id"].duplicated(keep=False)]
    treenode_series = treenode_df.set_index("node_id")["node_type"]
    return treenode_series


print("Getting treenode compartment types...")
currtime = time.time()
treenode_types = get_treenode_types(nl, splits, special_ids)
print(f"{time.time() - currtime:.3f} elapsed.\n")

#%%
print("Pulling connectors...\n")
currtime = time.time()
connectors = get_connectors(nl)
print(f"{time.time() - currtime:.3f} elapsed.\n")

#%%
explode_cols = ["postsynaptic_to", "postsynaptic_to_node"]
index_cols = np.setdiff1d(connectors.columns, explode_cols)

print("Exploding connector DataFrame...")
# explode the lists within the connectors dataframe
connectors = (
    connectors.set_index(list(index_cols)).apply(pd.Series.explode).reset_index()
)
# TODO figure out these nans
bad_connectors = connectors[connectors.isnull().any(axis=1)]
bad_connectors.to_csv(OUTPUT_PATH / "bad_connectors.csv")
# connectors = connectors[~connectors.isnull().any(axis=1)]
#%%
print(f"Connectors with errors: {len(bad_connectors)}")
connectors = connectors.astype(
    {
        "presynaptic_to": "Int64",
        "presynaptic_to_node": "Int64",
        "postsynaptic_to": "Int64",
        "postsynaptic_to_node": "Int64",
    }
)

#%%
print("Applying treenode types to connectors...")
currtime = time.time()
connectors["presynaptic_type"] = connectors["presynaptic_to_node"].map(treenode_types)
connectors["postsynaptic_type"] = connectors["postsynaptic_to_node"].map(treenode_types)

connectors["in_subgraph"] = connectors["presynaptic_to"].isin(ids) & connectors[
    "postsynaptic_to"
].isin(ids)
print(f"{time.time() - currtime:.3f} elapsed.\n")

#%%
print("Calculating neuron total inputs and outputs...")
axon_output_map = (
    connectors[connectors["presynaptic_type"] == "axon"]
    .groupby("presynaptic_to")
    .size()
)
axon_input_map = (
    connectors[connectors["postsynaptic_type"] == "axon"]
    .groupby("postsynaptic_to")
    .size()
)

dendrite_output_map = (
    connectors[connectors["presynaptic_type"].isin(["dendrite", "unsplit"])]
    .groupby("presynaptic_to")
    .size()
)
dendrite_input_map = (
    connectors[connectors["postsynaptic_type"].isin(["dendrite", "unsplit"])]
    .groupby("postsynaptic_to")
    .size()
)
meta["axon_output"] = meta.index.map(axon_output_map).fillna(0.0)
meta["axon_input"] = meta.index.map(axon_input_map).fillna(0.0)
meta["dendrite_output"] = meta.index.map(dendrite_output_map).fillna(0.0)
meta["dendrite_input"] = meta.index.map(dendrite_input_map).fillna(0.0)
print()

#%%
# remap the true compartment type mappings to the 4 that we usually use


def flatten_compartment_types(synaptic_type):
    if synaptic_type == "axon":
        return "a"
    elif synaptic_type == "dendrite" or synaptic_type == "unsplit":
        return "d"
    else:
        return "-"


def flatten(series):
    f = np.vectorize(flatten_compartment_types)
    arr = f(series)
    new_series = pd.Series(data=arr, index=series.index)
    return new_series


connectors["compartment_type"] = flatten(connectors["presynaptic_type"]) + flatten(
    connectors["postsynaptic_type"]
)


#%%
subgraph_connectors = connectors[connectors["in_subgraph"]]
meta_data_dict = meta.to_dict(orient="index")


def connectors_to_nx_multi(connectors, meta_data_dict):
    g = nx.from_pandas_edgelist(
        connectors,
        source="presynaptic_to",
        target="postsynaptic_to",
        edge_attr=True,
        create_using=nx.MultiDiGraph,
    )
    nx.set_node_attributes(g, meta_data_dict)
    return g


def flatten_muligraph(multigraph, meta_data_dict):
    # REF: https://stackoverflow.com/questions/15590812/networkx-convert-multigraph-into-simple-graph-with-weighted-edges
    g = nx.DiGraph()
    for node in multigraph.nodes():
        g.add_node(node)
    for i, j, data in multigraph.edges(data=True):
        w = data["weight"] if "weight" in data else 1.0
        if g.has_edge(i, j):
            g[i][j]["weight"] += w
        else:
            g.add_edge(i, j, weight=w)
    nx.set_node_attributes(g, meta_data_dict)
    return g


full_g = connectors_to_nx_multi(subgraph_connectors, meta_data_dict)

graph_types = ["aa", "ad", "da", "dd"]
color_multigraphs = {}
color_flat_graphs = {}
for graph_type in graph_types:
    color_subgraph_connectors = subgraph_connectors[
        subgraph_connectors["compartment_type"] == graph_type
    ]
    color_g = connectors_to_nx_multi(color_subgraph_connectors, meta_data_dict)
    color_multigraphs[graph_type] = color_g
    flat_color_g = flatten_muligraph(color_g, meta_data_dict)
    color_flat_graphs[graph_type] = flat_color_g

flat_g = flatten_muligraph(full_g, meta_data_dict)


print("Saving metadata as csv...")
meta.to_csv(OUTPUT_PATH / "meta_data.csv")
meta.to_csv(OUTPUT_PATH / "meta_data_unmodified.csv")

print("Saving connectors as csv...")
connectors.to_csv(OUTPUT_PATH / "connectors.csv")

# print("Saving full multigraph as graphml...")
# nx.write_graphml(full_g, OUTPUT_PATH / "full_multigraph.graphml")

# print("Saving each flattened color graph as graphml...")
# for graph_type in graph_types:
#     nx.write_graphml(
#         color_flat_graphs[graph_type], OUTPUT_PATH / f"G{graph_type}.graphml"
#     )
# nx.write_graphml(flat_g, OUTPUT_PATH / "G.graphml")


print("Saving each flattened color graph as txt edgelist...")
for graph_type in graph_types:
    nx.write_weighted_edgelist(
        color_flat_graphs[graph_type], OUTPUT_PATH / f"G{graph_type}_edgelist.txt"
    )
nx.write_weighted_edgelist(flat_g, OUTPUT_PATH / "G_edgelist.txt")

print("Saving color map as json...")
with open(OUTPUT_PATH / "simple_color_map.json", "w") as f:
    json.dump(color_map, f)


#%%
print()
print()
print("Done!")

elapsed = time.time() - t0
delta = timedelta(seconds=elapsed)
print("----")
print(f"{delta} elapsed for whole script.")
print("----")

sys.stdout.close()
