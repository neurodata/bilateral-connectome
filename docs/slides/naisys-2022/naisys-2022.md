---
marp: true
theme: poster
paginate: false
size: 44:33
---

<div class="header">
<div>

<!-- <div class=center_container> -->

![centerh](./images/../../../images/hopkins-logo.png)

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

![centerh](./images/../../../images/nd_logo.png)

<span style="text-align:center; margin:0; padding:0">

### [neurodata.io](https://neurodata.io/)

</span>

</div>
</div>


<!-- # Towards statistical comparative connectomics:<br> A case study on the bilateral symmetry of an insect brain connectome -->




<div class='box'>
<div class="columns5">
<div>

Aimed to define bilateral symmetry for a pair of networks, and formally test this hypothesis.

</div>
<div>

Found that left and right hemispheres are different under even the simplest model of a pair of networks

</div>
<div>

Left and right differ significantly in cell type connection probabilities, even when adjusting for the difference in density

</div>
<div>

Difference between hemispheres can be explained as combination of a network-wide and cell-type specific effects

</div>
<div>

Provided a definition of bilateral symmetry exhibited by this connectome, tools for future connectome comparisons

</div>
</div>
</div>

<div class="columns3">
<div>


### Motivation

- Connectomes are rich sources of inspiration for architectures in artificial intelligence, but unclear which structural features are necessary for yielding incredible capabilities animal intelligences. 
- Comparing connectomes 
<!-- - We explored statistically principled connectome comparison via a case study of a *Drosophila* larva connectome -->

### Larval *Drosophila* brain connectome

<!-- START subcolumns -->
<div class=columns2>
<div>

![center w:5.5in](./../../images/Figure1-brain-render.png)

**Fig 1A:** 3D rendering of larval *Drosophila* brain connectome 

</div>
<div>

![w:5.1in](./../../../results/figs/show_data/adjacencies.png)

**Fig 1B:** Adjacency matrix sorted by brain hemisphere


</div>
</div>

- Connectome of a larval *Drosophila* [1] has xxx neurons and xxx synapses

<!-- END subcolumns -->

<!-- ![center](../../../results/figs/show_data/adj_and_layout.png) -->

## Are <span style="color:var(--left)"> left </span> and the <span style="color:var(--right)"> right </span> networks "different"?
- Two sample testing problem! But for networks

### Density testing

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


**Fig 2A:** Comparison of densities via Fisher's exact test

</div>
<div>

**Fig 2B:** Densities are significantly different between hemispheres <br> ($p<10^{-23}$)

</div>
</div>


</div>
<div>



### Group connection testing 

<!-- #### A -->
![center w:13in](./../../../results/figs/sbm_unmatched_test/sbm_methods_explain.svg)
**Fig 3A:** Group connection testing fits SBMs using cell type partition. Group-to-group connection probabilities are compared (Fisher's exact test), p-values are combined (Tippett's method).

<!-- START subcolumns -->
<div class=columns2>
<div>

#### B
![](../../../results/figs/sbm_unmatched_test/sbm_uncorrected_pvalues.svg)

</div>
<div>

#### C
![center w:5in](../../../results/figs/sbm_unmatched_test/significant_p_comparison.svg)

</div>
</div>
<!-- END subcolumns -->

### Density-adjusted group connection testing

<!-- ![](./../../../results/figs/adjusted_sbm_unmatched_test/adjusted_methods_explain.svg)

![](./../../../results/figs/adjusted_sbm_unmatched_test/sbm_pvalues.svg) -->

![center w:14in](./../../../results/figs/adjusted_sbm_unmatched_test/adjusted_sbm_composite.svg)

**Figure x:** Adjusted the hypothesis from figure 

</div>
<div>



### Removing Kenyon cells
<!-- START subcolumns -->
<div class=columns2>
<div>

![](../../../results/figs/kc_minus/kc_minus_methods.svg)

</div>
<div>

- Density test: super small

- Group connection test: small

- Density adjusted group connection test: not small

</div>
</div>
<!-- END subcolumns -->

### Edge weight thresholds

<!-- ![](../../../results/figs/thresholding_tests/edge_weight_dist_input_proportion.png) -->

<!-- START columns -->
<div class="columns2-bl">
<div>

![](../../../results/figs/thresholding_tests/input_threshold_pvalues_p_removed_legend.svg)

</div>
<div>

- some stuff about it blah blah

</div>
</div>
<!-- END subcolumns -->

### Code
<div class="columns2">
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

### References

<footer>
[1]: Winding, Pedigo et al. *The complete connectome of an insect brain* In prep. (2022)
</footer>

</div>
</div>

