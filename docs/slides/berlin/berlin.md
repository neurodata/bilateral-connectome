---
marp: true
theme: slides
size: 16:9
paginate: true
---
<!-- _paginate: false -->

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)


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
# Tools for comparing connectomes: <br> evaluating the bilateral symmetry of a whole insect brain

<br>

## Benjamin D. Pedigo
_(he/him) - ![icon](../../images/email.png) [_bpedigo@jhu.edu_](mailto:bpedigo@jhu.edu) 
[NeuroData lab](https://neurodata.io/)_
_Johns Hopkins University - Biomedical Engineering_

#### Team

<!-- Start people panels -->
<div class='minipanels'>

<div>

![person](./../../images/people/michael_winding.png)
Michael Winding

</div>

<div>

![person](./../../images/people/mike-powell.jpg)
Mike Powell

</div>

<div>

![person](./../../images/people/bridgeford.jpg)
Eric Bridgeford

</div>

<div>

![person](./../../images/people/ali_saad_eldin.jpeg)
Ali <br> Saad-Eldin

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
# Comparative connectomics 
- Connectomes $\leftrightarrow$ {disease, evolution, development, experience, ...}
- As related connectomes are mapped, we'll want evaluate the *significance* and *nature* of differences between them
  
# Examples for today's talk
<style scoped>
h2 {
  padding-left: 100px
    /* justify-content: center; */
    /* text-align: center; */
}
</style>

 ## 1.  Are the <span style="color: var(--left)"> left </span> and <span style="color: var(--right)"> right </span> sides of a larva brain connectome *different*?

## 2. How can we *automatically* estimate neuron pairing between brain hemispheres?

---
# Testing for differences

<div class="columns">
<div>

### Are these two populations different?

![center h:250](./../../../results/figs/two_sample_testing/2_sample_real_line_wide.svg)

<div class='center'>

&nbsp; &nbsp; $\color{#66c2a5} Y^{(1)} \sim F^{(1)}$ &nbsp; &nbsp; &nbsp;  $\color{#fc8d62} Y^{(2)} \sim F^{(2)}$
$H_0: \color{#66c2a5} F^{(1)} \color{black} = \color{#fc8d62} F^{(2)}$ vs. $H_A: \color{#66c2a5} F^{(1)} \color{black} \neq \color{#fc8d62} F^{(2)}$

</div>


</div>
<div>

### Are these two *networks* different?

![center h:250](./../../../results/figs/plot_side_layouts/2_network_layout.png)

<div class='center'>

<span style='color: var(--left)'> $A^{(L)} \sim F^{(L)}$</span> &nbsp; &nbsp; &nbsp; &nbsp; <span style='color: var(--right)'> $A^{(R)} \sim F^{(R)}$ </span>
$H_0: \color{#66c2a5} F^{(L)} \color{black} = \color{#fc8d62}F^{(R)}$ vs. $H_A: \color{#66c2a5} F^{(L)} \color{black} \neq  \color{#fc8d62} F^{(R)}$

</div>
</div>
</div>
<div class='center'>

### Many ways to write what "symmetry" means! (different $F$, different statistics)

</div>

---
# Example: testing for differences in cell type connections

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
<!-- After adjusting for density: $p <0.01$ -->

</div>

</div>
</div>

<!-- ---
# We reject simple notions of symmetry

<div class="columns">
<div>

### Density test
Compares global connection probabilities
(Erdos-Renyi models)

<div class="columns2-bl">
<div>

![center h:350](./../../../results/figs/er_unmatched_test/er_density.png)

</div>
<div>


p-value: 
$p < 10^{-23}$

</div>
</div>


</div>
<div>

### Cell type connection test
Compares between-cell-type connection probabilities (stochastic block models)

<div class="columns2-bl">
<div>

![center h:350](./../../../results/figs/sbm_unmatched_test/sbm_uncorrected_pvalues_unlabeled.svg)


</div>
<div>

Overall comparison: $p < 10^{-7}$

Density-adjusted: $p <0.01$


</div>
</div>




</div>
</div>

 -->

---
# Examining the effect of edge weights

<div class='columns'>
<div>

![](./../../../results/figs/thresholding_tests/thresholding_methods.png$$)

</div>
<div>

![](./../../../results/figs/thresholding_tests/input_threshold_pvalues_legend.svg)

</div>
</div>

---
# Estimating neuron pairing using graph matching

<div class="columns-br">
<div>

![](../../images/explain-gm.png)

</div>
<div>

![center h:400](./../../_build/html/_images/left-pair-predictions.svg)

<div class='center'>
Morphologies of pairs predicted from connectivity.<br> ~80-85% agreement with an expert annotator.
</div>

</div>
</div>

<!-- <div class="columns-br">
<div>

- Neuron pair predictions from connectivity using improved graph matching tools

</div>
<div>




</div>
</div> -->

--- 
# Improving graph matching to suit connectomes

<div class="columns">
<div>

### Incorporating contralateral connections improves matching accuracy

![](./../../images/match_accuracy_comparison.svg)

</div>
<div>

### Improving accuracy and scalability

<!-- ![](./../../images/msr_corr.png) -->

![](../../images/goat-title.png)

Runs in ~1hr for 10k node networks

</div>
</div>

<!-- _footer: Pedigo et al. bioarxiv (2022), Saad-Eldin et al. arxiv (2021)-->

---
# Conclusions

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

<div class="columns-bl">
<div>

- Demonstrated novel tools for comparing connectomes, case study on symmetry in a *Drosophila* larva 
  - Model-based network comparison
  - Improved methods for matching neurons via connectivity
- Can be applied more generally to compare connectomes!
- *Ongoing work*: combining testing and matching frameworks to evaluate stereotypy at the edge-level
- **Have other network analysis questions? Let's chat!**

</div>
<div>

**Slides, code, papers, contact**
![center h:350](../../images/further-info-qr.svg)

![icon](../../images/email.png) [_bpedigo@jhu.edu_](mailto:bpedigo@jhu.edu)  
![icon](../../images/twitter.png)[_@bpedigod_](https://twitter.com/bpedigod)
![icon](../../images/web.png)[bdpedigo.github.io](https://bdpedigo.github.io/)


</div>
</div>