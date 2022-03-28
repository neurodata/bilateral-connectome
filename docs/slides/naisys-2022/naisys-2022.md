---
marp: true
theme: poster
paginate: false
size: 44:33
---

<div class="header">
<div>

<!-- <div class=center_container> -->

![headerlogo](./images/../../../images/hopkins-logo.png)

<!-- <br>
<br>
<br>

### Summary -->

<!-- </div> -->

</div>
<div>

# Is a whole insect brain connectome bilaterally symmetric? <br> A case study on comparing two networks

## Benjamin D. Pedigo<span class=super>1*</span>, Mike Powell<span class=super>1</span>, Eric W. Bridgeford<span class=super>1</span>, Michael Winding<span class=super>2</span>, Carey E. Priebe<span class=super>1</span>, Joshua T. Vogelstein<span class=super>1</span>

<div class=underauthor> 

1 - Johns Hopkins University, 2 - University of Cambridge, $\ast$ - correspondence: ![icon](../../images/email.png) [_bpedigo@jhu.edu_](mailto:bpedigo@jhu.edu) ![icon](../../images/github.png) [_@bdpedigo (Github)_](https://github.com/bdpedigo) ![icon](../../images/twitter.png) [_@bpedigod (Twitter)_](https://twitter.com/bpedigod) ![icon](../../images/web.png) [https://bdpedigo.github.io/](https://bdpedigo.github.io/) 

</div>

</div>
<div>

![headerlogo](./images/../../../images/nd_logo.png)

<span style="text-align:center; margin:0; padding:0">

<!-- ### [neurodata.io](https://neurodata.io/) -->

</span>

</div>
</div>


<!-- # Towards statistical comparative connectomics:<br> A case study on the bilateral symmetry of an insect brain connectome -->


<span class='h3-noline'> Summary </span>

<div class='box'>
<div class="columns5">
<div>

<!-- #### Summary -->

- Aimed to define bilateral symmetry for a pair of networks, and formally test this hypothesis.

</div>
<div>

- Left and right hemispheres are significantly different under even the simplest model of a pair of networks

</div>
<div>

- Left and right differ significantly in cell type connection probabilities, even when adjusting for the difference in density

</div>
<div>

- Difference between hemispheres can be explained as combination of network-wide and cell type-specific effects

</div>
<div>

- Provided a definition of bilateral symmetry exhibited by this connectome, tools for future connectome comparisons

</div>
</div>
</div>

<div class="columns3">
<div>


### Motivation

- Connectomes are rich sources of inspiration for architectures in artificial intelligence.
- Comparing connectomes could help elucidate which structural features are necessary for yielding the capabilities animal intelligences.
- Bilateral symmetry for connectomes has been investigated, but not clearly defined as a network hypothesis.

<!-- - We explored statistically principled connectome comparison via a case study of a *Drosophila* larva connectome -->

### Larval *Drosophila* brain connectome

<!-- START subcolumns -->
<div class=columns2>
<div>

![center w:5.5in](./../../images/Figure1-brain-render.png)

**Fig 1A:** 3D rendering of larval *Drosophila* brain connectome [1]. Comprised of ~3k neurons and ~544k synapses.

</div>
<div>

![w:5.1in](./../../../results/figs/show_data/adjacencies.png)

**Fig 1B:** Adjacency matrix sorted by brain hemisphere. We focus on comparing $\color{#66c2a5} L \rightarrow L$ vs. $\color{#fc8d62} R \rightarrow R$ subgraphs.

</div>
</div>

<!-- - Connectome of a larval *Drosophila* [1] has xxx neurons and xxx synapses -->

<!-- END subcolumns -->

<!-- ![center](../../../results/figs/show_data/adj_and_layout.png) -->

## Are the <span style="color:var(--left)"> left </span> and <span style="color:var(--right)"> right </span> networks "different"?
<br>

<!-- - Two sample testing problem! But for networks -->
Requires that we define what we could mean by "different" for a pair of networks, and develop a test procedure for each definition.

### Density test (Model 1)

<div class=columns2>
<div>

![](../../../results/figs/er_unmatched_test/er_methods.svg)

</div>
<div>

![](../../../results/figs/er_unmatched_test/er_density.svg)

</div>
</div>

<div class=columns2>
<div>


**Fig 2A:** Testing symmetry under Erdos-Renyi (ER) model [2] amounts to comparing densities (here via Fisher's exact test).

</div>
<div>

**Fig 2B:** Densities are significantly different between hemispheres <br> ($p<10^{-23}$).

</div>
</div>

</div>
<div>


### Group connection test (Model 2)

<!-- #### A -->
![center w:11.5in](./../../../results/figs/sbm_unmatched_test/sbm_methods_explain.svg)
**Fig 3A:** Testing under stochastic block model (SBM) compares probabilities of connections between groups (here using cell types).

<!-- START subcolumns -->
<div class=columns2>
<div>

![](../../../results/figs/sbm_unmatched_test/sbm_uncorrected_pvalues.svg)

</div>
<div>

![center w:5in](../../../results/figs/sbm_unmatched_test/significant_p_comparison.svg)

</div>
</div>

<div class=columns2>
<div>

**Fig 3B:** Corrected p-values for each group connection. P-value test of equal group connections is $<10^{-8}$.

<!-- 5 connections are $<0.05$, shown with "X"s. -->

</div>
<div>

**Fig 3C:** Comparison of probabilities for significant connections: denser hemisphere probability always higher.
<!-- Probability is always higher on right side. -->

</div>
</div>

### Density-adjusted group connection test (Model 3)

<!-- ![](./../../../results/figs/adjusted_sbm_unmatched_test/adjusted_methods_explain.svg)

![](./../../../results/figs/adjusted_sbm_unmatched_test/sbm_pvalues.svg) -->

<!-- ![center w:14in](./../../../results/figs/adjusted_sbm_unmatched_test/adjusted_sbm_composite.svg) -->
<div class=columns2>
<div>

<br>

![center w:5in](./../../../results/figs/adjusted_sbm_unmatched_test/adjusted_methods_explain.svg)

</div>
<div>

![](./../../../results/figs/adjusted_sbm_unmatched_test/sbm_pvalues.svg)

</div>
</div>

<div class=columns2>
<div>

**Fig 4A:** Density-adjusted hypothesis, modified from Fig 3. 

</div>
<div>

**Fig 4B:** Corrected p-values for group connections w/ density adjustment. 

</div>
</div>


</div>
<div>


<!-- ### Removing Kenyon cells -->

<!-- - Density test: $p < 10^{-26}$
- Group connection test: $p < 10^{-2}$
- Density-adjusted group connection test: $p \approx 0.5$ -->

<!-- ### Removing Kenyon cells
Reran all tests after removing the asymmetric cell type (see below) -->

### Notions of bilateral symmetry

<!-- <style scoped>
table {
    font-size: 0.3in;
    /* text-align: center; */
    /* margin-bottom: 50px; */
}
</style> -->

<div class="columns2">
<div>

#### With Kenyon cells
| **Model** |                       $H_0$ (vs. $H_A \neq$)                       | p-value |
| :----: | :----------------------------------------------------------------: | :-----: |
| **1** |  $\color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$  |    x    |
| **2** | $\color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$  |         |
| **3** | $\color{#66c2a5}B^{(L)} \color{black}  = c \color{#fc8d62}B^{(R)}$ |         |


</div>
<div>

#### Without Kenyon cells
| Model |                       $H_0$ (vs. $H_A \neq$)                       | p-value |
| :----: | :----------------------------------------------------------------: | :-----: |
| **1** |  $\color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$  |    x    |
| **2** | $\color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$  |    d    |
| **3** | $\color{#66c2a5}B^{(L)} \color{black}  = c \color{#fc8d62}B^{(R)}$ |         |

</div>
</div>

<!-- #### With Kenyon cells
| Model  |                       $H_0$ (vs. $H_A \neq$)                       | p-value |
| :----- | :----------------------------------------------------------------: | :-----: |
| ER     |  $\color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$  |    x    |
| SBM    | $\color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$  |         |
| DA-SBM | $\color{#66c2a5}B^{(L)} \color{black}  = c \color{#fc8d62}B^{(R)}$ |         |

#### Without Kenyon cells
| Model  |                       $H_0$ (vs. $H_A \neq$)                       | p-value |
| :----- | :----------------------------------------------------------------: | :-----: |
| ER     |  $\color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$  |    x    |
| SBM    | $\color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$  |    d    |
| DA-SBM | $\color{#66c2a5}B^{(L)} \color{black}  = c \color{#fc8d62}B^{(R)}$ |         | --> |


### Edge weight thresholds

<!-- ![](../../../results/figs/thresholding_tests/edge_weight_dist_input_proportion.png) -->

<div class="columns2">
<div>

![](./results/thresholding_tests/../../../../../results/figs/thresholding_tests/thresholding_methods.svg)

</div>
<div>

![](../../../results/figs/thresholding_tests/input_threshold_pvalues_legend.svg)

</div>
</div>

<div class="columns2">
<div>

**Fig 5A:** Removed edges w/ weight (synapse count or proportion of input to downstream neuron) below some threshold, testing bilateral symmetry for each resulting pair of networks. 

</div>
<div>

**Fig 5B:** Higher edge weight thresholds generally make networks more symmetric. Less apparent when using synapse counts as edge weights (not shown). 

</div>
</div>

### Limitations and extensions
- Many other models to consider (e.g. random dot product graph [3])
- Many other potential neuron groupings for group connection testing
- Matched nodes between networks

### Code and references
<br>
<div class="columns2-np">
<div>

<div class="columns2-np">
<div>

![center h:1in](./../../images/graspologic_svg.svg)

</div>
<div>

[![h:.4in](https://pepy.tech/badge/graspologic)](https://pepy.tech/project/graspologic) 
[![h:.4in](https://img.shields.io/github/stars/microsoft/graspologic?style=social)](https://github.com/microsoft/graspologic)

</div>
</div>

[github.com/microsoft/graspologic](https://github.com/microsoft/graspologic)


</div>
<div>

[![h:0.4in](https://jupyterbook.org/badge.svg)](http://docs.neurodata.io/bilateral-connectome/)

[github.com/neurodata/bilateral-connectome](https://github.com/neurodata/bilateral-connectome) 


</div>
</div>

<footer>
[1]: Winding, Pedigo et al. *The complete connectome of an insect brain,* In prep. (2022)
<br>
[2]: Chung et al. *Statistical connectomics,* Ann. Rev. Statistics and its Application (2021)
<br>
[3]: Athreya et al. *Statistical inference on random dot product graphs: a survey,* JMLR (2017)
</footer>

</div>
</div>

