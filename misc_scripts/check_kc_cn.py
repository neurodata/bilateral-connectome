#%%
left_nodes["inds"] = np.arange(len(left_nodes))
right_nodes["inds"] = np.arange(len(right_nodes))
left_kc_inds = left_nodes.query("simple_group == 'KCs'")["inds"]
right_kc_inds = right_nodes.query("simple_group == 'KCs'")["inds"]
left_cn_inds = left_nodes.query("simple_group == 'CNs'")["inds"]
right_cn_inds = right_nodes.query("simple_group == 'CNs'")["inds"]

left_kc_cn_adj = left_adj[left_kc_inds][:, left_cn_inds]
right_kc_cn_adj = right_adj[right_kc_inds][:, right_cn_inds]

#%%

fig, axs = plt.subplots(1, 2, figsize=(10, 5))
ax = axs[0]
heatmap_kws = dict(
    xticklabels=False,
    yticklabels=False,
    cbar=False,
    cmap="RdBu_r",
    center=0,
    square=True,
)
ax.set_title("Left")
sns.heatmap(left_kc_cn_adj, ax=ax, **heatmap_kws)
ax = axs[1]
ax.set_title("Right")
sns.heatmap(right_kc_cn_adj, ax=ax, **heatmap_kws)

for ax in axs.flat:
    ax.set(xlabel="CNs", ylabel="KCs")

fig.set_facecolor("w")

#%%
right_has_kc = right_kc_cn_adj.sum(axis=0) > 0
cn_right_nodes = right_nodes.query("simple_group == 'CNs'").copy()
cn_right_nodes["has_kc"] = right_has_kc
cn_right_nodes.query("has_kc").to_csv(OUT_PATH / "cn_right_nodes_from_kc.csv")

#%%
right_has_cn = right_kc_cn_adj.sum(axis=1) > 0
kc_right_nodes = right_nodes.query("simple_group == 'KCs'").copy()
kc_right_nodes["has_cn"] = right_has_cn
kc_right_nodes.query("has_cn").to_csv(OUT_PATH / "kc_right_nodes_to_cn.csv")
