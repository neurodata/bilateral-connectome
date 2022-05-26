---
marp: true
theme: slides
size: 16:9
paginate: true
---
<!-- _paginate: false -->

![bg center blur:3px opacity:30%](./../../../results/figs/background/background.svg)


<style scoped> 
/* h1 {
    font-size:40px;
} */
p {
    font-size: 24px;
}
</style>

<!-- # Generative network modeling reveals a first quantitative definition of bilateral symmetry exhibited by a whole insect brain connectome -->
<!-- ![icon](../../images/twitter.png) [_@bpedigod (Twitter)_](https://twitter.com/bpedigod) -->


<!-- # Generative network modeling reveals a quantitative definition of bilateral symmetry exhibited by a whole insect brain connectome -->
# Model-based comparison of connectomes: evaluating the bilateral symmetry of a whole insect brain

## Benjamin D. Pedigo
_(he/him) - ![icon](../../images/email.png) [_bpedigo@jhu.edu_](mailto:bpedigo@jhu.edu) 
[NeuroData lab](https://neurodata.io/)_
_Johns Hopkins University - Biomedical Engineering_

#### Team

<!-- Start people panels -->
<div class='minipanels'>

<div>

![person](./../../images/people/mike-powell.jpg)
Mike Powell

</div>

<div>

![person](./../../images/people/bridgeford.jpg)
Eric Bridgeford

</div>

<div>

![person](./../../images/people/michael_winding.png)
Michael Winding

</div>

<div>

![person](./../../images/people/marta_zlatic.jpeg)
Marta Zlatic

</div>

<div>

![person](./../../images/people/albert_cardona.jpeg)
Albert Cardona

</div>

<div>

![person](./../../images/people/priebe_carey.jpg)
Carey Priebe

</div>

<div>

![person](./../../images/people/vogelstein_joshua.jpg)
Joshua Vogelstein

</div>

<!-- End people panels -->
</div>


<!-- 

<!-- ![bg center blur:2.5px opacity:20%](./../../../results/figs/background/background.svg) -->

<!-- ![bg opacity:.6 95%](./../../../results/figs/plot_side_layouts/2_network_layout.png) -->


<!-- 
![icon](../../images/email.png) [_bpedigo@jhu.edu_](mailto:bpedigo@jhu.edu)
![icon](../../images/github.png) [_@bdpedigo (Github)_](https://github.com/bdpedigo)
![icon](../../images/twitter.png) [_@bpedigod (Twitter)_](https://twitter.com/bpedigod)
![icon](../../images/web.png) [https://bdpedigo.github.io/](https://bdpedigo.github.io/) -->


<!-- <!-- --- -->

<!-- ---
# Motivation
- This is why we need to compare connectomes to understand x,y,z
- Need methods etc. -->


<!-- ---
# What we do (notes)
- clustering
- model estimation
- flow
- testing for differences (e.g. left/right)
- matching (e.g. left/right)
- testing for stereotypy at the edge level (combo of the above two) -->

<!-- --- 
# Approaches for analyzing connectome data

<div class="columns">
<div>

- Model-based clustering of neurons by connectivity 
- Graph matching to estimate correspondence of neurons between datasets/brain hemispheres
- **Statistical testing for connectome comparison**

</div>
<div>

![center](./../../../results/figs/plot_side_layouts/2_network_layout.png)

</div>
</div>

<style scoped>
h2 {
    justify-content: center;
    text-align: center;
}
</style>

## Are the <span style="color: var(--left)"> left </span> and <span style="color: var(--right)"> right </span> sides of the larva brain connectome <p> </p> *different*? -->

--- 
# Why bilateral symmetry?
- Fundamental property of almost all animals
- Often (implicitly or explicitly) assumed in connectomics
- ...but many ways to write down what we mean for the networks

# Why else?
- As more connectomes are mapped, we'll want evaluate the *significance* and *nature* of differences between them

<style scoped>
h2 {
    justify-content: center;
    text-align: center;
}
</style>

## Are the <span style="color: var(--left)"> left </span> and <span style="color: var(--right)"> right </span> sides of the larva brain connectome <p> </p> *different*?

<!-- --- 
# Larval _Drosophila_ brain connectome

<div class="columns">
<div>

![center w:375](./images/../../../images/Figure1-brain-render.png)

</div>
<div>

![center](./../../../results/figs/plot_side_layouts/2_network_layout.png)

</div>
</div>

<style scoped>
h2 {
    justify-content: center;
    text-align: center;
}
</style>

## Are the <span style="color: var(--left)"> left </span> and <span style="color: var(--right)"> right </span> sides of this connectome <p> </p> *different*? -->

<!-- <footer>
Winding, Pedigo et al. “The complete connectome of an insect brain.” In prep. (2022)
</footer> -->
<!-- 
---
# Explain the statistical approach
- Decide what you want to quantify or compare -->

<!-- --- -->

<!-- # Are these populations different?

<div class="columns">
<div>

![center w:400](./../../../results/figs/two_sample_testing/2_sample_real_line.svg)

</div>
<div>

- Known as two-sample testing
- $\color{#66c2a5} Y^{(1)} \sim F^{(1)}$, $\color{#fc8d62} Y^{(2)} \sim F^{(2)}$
- $H_0: \color{#66c2a5} F^{(1)} \color{black} = \color{#fc8d62} F^{(2)}$  
  $H_A: \color{#66c2a5} F^{(1)} \color{black} \neq \color{#fc8d62} F^{(2)}$

</div>
</div> -->

<!-- --- 
# Are these two _networks_ different?

<div class="columns">
<div>

![center w:1000](./../../../results/figs/plot_side_layouts/2_network_layout.png)

</div>
<div>


- Want a **two-network-sample** test!
- <span style='color: var(--left)'> $A^{(L)} \sim F^{(L)}$</span>, <span style='color: var(--right)'> $A^{(R)} \sim F^{(R)}$ </span>
- $H_0: \color{#66c2a5} F^{(L)} \color{black} = \color{#fc8d62}F^{(R)}$  
  $H_A: \color{#66c2a5} F^{(L)} \color{black} \neq  \color{#fc8d62} F^{(R)}$
- What's $F$ for a network?
- We'll start with $A$ (the networks) being directed, unweighted (for now)

</div>
</div> -->

---
# Testing for differences

<div class="columns">
<div>

### Are these two populations different?

![center h:250](./../../../results/figs/two_sample_testing/2_sample_real_line_wide.svg)

<div class='center'>

&nbsp; &nbsp; $\color{#66c2a5} Y^{(1)} \sim F^{(1)}$ &nbsp; &nbsp; &nbsp;  $\color{#fc8d62} Y^{(2)} \sim F^{(2)}$

$H_0: \color{#66c2a5} F^{(1)} \color{black} = \color{#fc8d62} F^{(2)}$  
$H_A: \color{#66c2a5} F^{(1)} \color{black} \neq \color{#fc8d62} F^{(2)}$

</div>


</div>
<div>

### Are these two *networks* different?

![center h:250](./../../../results/figs/plot_side_layouts/2_network_layout.png)


<div class='center'>

<span style='color: var(--left)'> $A^{(L)} \sim F^{(L)}$</span> &nbsp; &nbsp; &nbsp; &nbsp; <span style='color: var(--right)'> $A^{(R)} \sim F^{(R)}$ </span>

$H_0: \color{#66c2a5} F^{(L)} \color{black} = \color{#fc8d62}F^{(R)}$  
$H_A: \color{#66c2a5} F^{(L)} \color{black} \neq  \color{#fc8d62} F^{(R)}$

</div>


</div>
</div>

---
# We reject even the simplest notion of symmetry

<div class="columns">
<div>

- Fit Erdos-Renyi models to the left and the right brain networks
![](./../../../results/figs/er_unmatched_test/er_explain.svg)
- Compare densities:
  $H_0: \color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$  
  $H_A: \color{#66c2a5} p^{(L)} \color{black} \neq  \color{#fc8d62} p^{(R)}$

<!-- ![center h:300](../../../results/figs/er_unmatched_test/er_methods.png) -->


</div>
<div>

![center h:350](./../../../results/figs/er_unmatched_test/er_density.png)

<br>

<div class='center'>

p-value: ${<}10^{-23}$

</div>

</div>
</div>


---
# Localizing differences to cell type connections

<div class="columns">
<div>

- Fit block models to both hemispheres
  ![](./../../../results/figs/sbm_unmatched_test/sbm_simple_methods.svg)
- Compare connection probabilities:
  $H_0: \color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$ 
  $H_A: \color{#66c2a5} B^{(L)} \color{black} \neq  \color{#fc8d62} B^{(R)}$

</div>
<div>

![center h:400](./../../../results/figs/sbm_unmatched_test/sbm_uncorrected_pvalues_unlabeled.svg)

<div class='center'>

Overall comparison: $p < 10^{-7}$

After adjusting for density: $p <0.01$

</div>

</div>
</div>




<!-- # Extensions (current and future)

- How does evaluation of symmetry depend on what you call a "cell type"?
  - Clustering neurons by connectivity
- Evaluating symmetry at the edge-level using neuron pairs
  - Testing for significant alignment with graph matching
- Power analysis: what differences could one even conceivably detect in a comparative connectomics experiment?
- **What about edge weights?** -->

<!-- <div class="columns">
<div>




</div>
<div>



</div>
</div> -->


<!-- ---
# Rescaling connection probabilities focuses remaining asymmetry on Kenyon cells

<div class="columns">
<div>

![center](../../../results/figs/adjusted_sbm_unmatched_test/adjusted_methods_explain.svg)

</div>
<div>

![center h:350](./../../../results/figs/adjusted_sbm_unmatched_test/sbm_pvalues_unlabeled.svg)

<div class='center'>

w/ Kenyon cells: $p < 0.05$
w/o Kenyon cells: $p \approx 0.51$

</div>


</div>
</div> -->


---
# Extentions: examining the effect of edge weights

<div class='columns'>
<div>

![](./../../../results/figs/thresholding_tests/thresholding_methods.svg)

</div>
<div>

![h:400](./../../../results/figs/thresholding_tests/input_threshold_pvalues_legend.svg)

*Only occurs when using input percentage as edge weight, not synapse number*

</div>
</div>

---
# Extensions: incorporating neuron/connection matching

<div class="columns-bl">
<div>

![center h:400](./../../_build/html/_images/left-pair-predictions.svg)

Neuron pair predictions from connectivity using improved graph matching tools

</div>
<div>

*Ongoing work*: combining matching and testing frameworks to evaluate stereotypy at the edge-level

</div>
</div>

<!-- *Ongoing work: testing for a significant matching/symmetry using tools from graph matching* -->

<!-- ---
<div class="columns">
<div>

## graspologic:

[github.com/microsoft/graspologic](https://github.com/microsoft/graspologic)

![w:450](./../../images/graspologic_svg.svg)

[![h:50](https://pepy.tech/badge/graspologic)](https://pepy.tech/project/graspologic)  [![h:50](https://img.shields.io/github/stars/microsoft/graspologic?style=social)](https://github.com/microsoft/graspologic)  [![h:50](https://img.shields.io/github/contributors/microsoft/graspologic)](https://github.com/microsoft/graspologic/graphs/contributors)  [![h:50](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

</div>
<div>


## This work:
[github.com/neurodata/bilateral-connectome](https://github.com/neurodata/bilateral-connectome) 
![center w:400](./../../images/jb_example.png)
[![h:50](https://jupyterbook.org/badge.svg)](http://docs.neurodata.io/bilateral-connectome/)

</div>
</div> -->

<!-- # Conclusions  -->

<!-- ---
![bg center blur:3px opacity:15%](./../../../results/figs/background/background.svg)

<div class='columns2-bl'>
<div>

## Conclusions

- Testing hypotheses in connectomics requires techniques for networks
    - We presented procedures for comparing connectomes
- Used to evaluate bilateral symmetry, finding how this brain is/is not bilaterally symmetric
- Poised to apply these tools to answer...
   - **{Your question here}**
   - Get in touch: 
     - ![icon](../../images/email.png) [_bpedigo@jhu.edu_](mailto:bpedigo@jhu.edu) 
     - ![icon](../../images/email.png) [_jovo@jhu.edu_](mailto:jovo@jhu.edu)

</div>
<div> -->


<!-- ## More info
- [![icon h:80](./../../images/graspologic_svg.svg)](https://github.com/microsoft/graspologic) [![icon](https://pepy.tech/badge/graspologic)](https://pepy.tech/project/graspologic)
- This work: [![icon](https://jupyterbook.org/badge.svg)](http://docs.neurodata.io/bilateral-connectome/)[github.com/neurodata/bilateral-connectome](https://github.com/neurodata/bilateral-connectome)
- Chung et al. *Statistical connectomics* (2021)
- Data: Winding, Pedigo et al. *In preparation* (2022) -->

<!-- --- 

![bg center blur:3px opacity:15%](./../../../results/figs/background/background.svg)


<div class="columns">
<div>

## Conclusions

- We focus on statistical analyses of connectome networks
- We developed tools for testing for differences between connectome networks
- Found ways in which the hemispheres of larval brain could or could not be considered "different" 

</div>
<div>

## Future work
- Testing for stereotypy/differences *at an *


</div>
</div> -->


---
# An invitation!

![bg center blur:3px opacity:15%](./../../../results/figs/background/background.svg)

<div class="columns">
<div>


- Want to use anything I talked about today?
- OR have another network question you want to test?
- **Let's chat!**

</div>
<div>

Code, slides, papers, contact info:
![center h:350](../../images/further-info-qr.svg)

![icon](../../images/email.png) [_bpedigo@jhu.edu_](mailto:bpedigo@jhu.edu)     ![icon](../../images/twitter.png)[_@bpedigod_](https://twitter.com/bpedigod)
![icon](../../images/github.png) [_@bdpedigo_](https://github.com/bdpedigo) ![icon](../../images/web.png)[bdpedigo.github.io](https://bdpedigo.github.io/)

</div>
</div>

<!-- <footer>Chung, Pedigo et al. JMLR (2019) <br> Winding, Pedigo et al. In prep. (2022) <br> Pedigo et al. In prep. (2022)</footer> -->

<!-- ---
# Questions?

![bg opacity:.6 95%](./../../../results/figs/plot_side_layouts/2_network_layout.png)

<span> </span>
<span> </span>
<span> </span>
<span> </span>
<span> </span>

<style scoped>
section {
    justify-content: center;
    text-align: center;
}
</style>

### Benjamin D. Pedigo
![icon](../../images/email.png) [_bpedigo@jhu.edu_](mailto:bpedigo@jhu.edu)
![icon](../../images/twitter.png) [_@bpedigod (Twitter)_](https://twitter.com/bpedigod)
![icon](../../images/github.png) [_@bdpedigo (Github)_](https://github.com/bdpedigo)
![icon](../../images/web.png) [https://bdpedigo.github.io/](https://bdpedigo.github.io/) -->



<!-- Ross: better summary of what the actual summary/conclusion is -->
<!-- Never say the method that we're using -->
<!-- Talk about it as a toolbox? -->
<!-- Sell the work a bit more in terms of novelty - nobody else has done x -->
<!-- Toolset vs. symmetry is complicated -->
<!--  -->