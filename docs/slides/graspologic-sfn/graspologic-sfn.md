---
marp: true
theme: poster
paginate: false
size: 44:33
---


<div class="header">
<div>

![headerlogo](./../../images/graspologic_svg.svg)

</div>
<div>

<!-- # Is a whole insect brain connectome bilaterally symmetric? <br> A case study on comparing two networks -->

<!-- # `graspologic` -->

# `graspologic`: network analysis in Python


<!-- # Towards statistical comparative connectomics:<br> A case study on the bilateral symmetry of an insect brain connectome -->


<!-- ## Benjamin D. Pedigo<span class=super>1*</span>, Mike Powell<span class=super>1</span>, Eric W. Bridgeford<span class=super>1</span>, Michael Winding<span class=super>2</span>, Carey E. Priebe<span class=super>1</span>, Joshua T. Vogelstein<span class=super>1</span> -->

<!-- ##### 1 - Johns Hopkins University, 2 - University of Cambridge, $\ast$ - correspondence: ![icon](../../images/email.png) [_bpedigo@jhu.edu_](mailto:bpedigo@jhu.edu) ![icon](../../images/github.png) [_@bdpedigo (Github)_](https://github.com/bdpedigo) ![icon](../../images/twitter.png) [_@bpedigod (Twitter)_](https://twitter.com/bpedigod) ![icon](../../images/web.png) [_bdpedigo.github.io_](https://bdpedigo.github.io/)  -->


<!-- [github.com/microsoft/graspologic](https://github.com/microsoft/graspologic) -->

## github.com/microsoft/graspologic   
## [![h:90](https://pepy.tech/badge/graspologic)](https://pepy.tech/project/graspologic)  [![h:90](https://img.shields.io/github/stars/microsoft/graspologic?style=social)](https://github.com/microsoft/graspologic)  [![h:90](https://img.shields.io/github/contributors/microsoft/graspologic)](https://github.com/microsoft/graspologic/graphs/contributors) [![h:90](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) ![h:90](./../../images/graspologic-qr.svg)

</div>
<div>

![headerlogo](./images/../../../images/hopkins-logo.png)

![headerlogo](./../../images/msr_logo.png)

<!-- ![headerlogo](./images/../../../images/nd_logo.png) -->

<span style="text-align:center; margin:0; padding:0">

<!-- ### [neurodata.io](https://neurodata.io/) -->

</span>

</div>
</div>

<div class="columns-main">

<div>

### Network dimensionality reduction

### Visualization

![center](../../images/background.png)

</div>

<div>

### Model estimation and sampling

![center](./../../images/graspologic-models.png)

### Clustering

<!-- ![center](../../images/bar-dendrogram-wide.svg) -->

![center](./../../images/adjacency-matrix-clustered.png)

</div>

<div>

### Statistical testing

![](./../../../results/figs/show_data/adj_and_layout.png)


<style scoped>
h2 {
    justify-content: center;
    text-align: center;
}
</style>

## Are the <span style="color: var(--left)"> left </span> and <span style="color: var(--right)"> right </span> sides of this connectome <p> </p> *different*?

- <span style='color: var(--left)'> $A^{(L)} \sim F^{(L)}$</span>, <span style='color: var(--right)'> $A^{(R)} \sim F^{(R)}$ </span>
- $H_0: \color{#66c2a5} F^{(L)} \color{black} = \color{#fc8d62}F^{(R)}$  
  $H_A: \color{#66c2a5} F^{(L)} \color{black} \neq  \color{#fc8d62} F^{(R)}$

<!-- ![](./show) -->

### Graph matching

![center](./../../images/network-matching-explanation.svg)

![center](./../../images/example_matched_morphologies_good.svg)

### Citation

## jmlr.org/papers/volume20/19-490/19-490.pdf

Chung, J.,* Pedigo, B. D.,* Bridgeford, E. W., Varjavand, B. K., Helm, H. S., & Vogelstein, J. T. (2019). GraSPy: Graph Statistics in Python. J. Mach. Learn. Res., 20(158), 1-7.

</div>

</div>
