---
marp: true
theme: slides
size: 16:9
paginate: true
---
<!-- _paginate: false -->

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)


<style scoped> 
p {
    font-size: 24px;
}
</style>

# Tools for comparative connectomics: <br> case studies from two sides of a larval *Drosophila* brain

<br>

<div class="columns">
<div>

## Benjamin D. Pedigo
(he/him)
[NeuroData lab](https://neurodata.io/)
Johns Hopkins University - Biomedical Engineering


![icon](../../images/email.png) [bpedigo@jhu.edu](mailto:bpedigo@jhu.edu)
![icon](../../images/github.png) [@bdpedigo (Github)](https://github.com/bdpedigo)
![icon](../../images/twitter.png) [@bpedigod (Twitter)](https://twitter.com/bpedigod)
![icon](../../images/web.png) [bdpedigo.github.io](https://bdpedigo.github.io/)

</div>
<div>

### These slides at: 
### [tinyurl.com/princeton-bilarva](https://tinyurl.com/princeton-bilarva)


</div>
</div>

<!-- ---

# Connectomics is useful...

TODO: (5) plot of "connectome prevalence over time"

TODO: (5) highlight one example fly result (? maybe central complex) -->

---

# Many goals of connectomics involve linking connectome to other properties

![center h:500](./../../images/link_connectome.svg)

<!-- TODO: (4) finish this figure draw arrows -->

---

# Comparative connectomics as a potential solution? 

- Map connectomes from related individuals/organisms which may differ in feature $X$: 
  - Genome
  - Behavioral patterns/habits
  - Life experience
  - Developmental stage
- Compare connectomes
- Understand how $X$ {affects, is affected by, is associated with} connectome structure

---

# Connectome $\leftrightarrow$ memory

![center h:200](../../images/mind-of-a-mouse.png)

> ...the acquisition of wiring diagrams across multiple individuals will yield insights into how experiences shape neural connections.

<!-- *Emphasis added* -->


<!-- _footer: Abbott et al. Cell (2020) -->

---

# Connectome $\leftrightarrow$ disease

![center h:200](../../images/mind-of-a-mouse.png)

> The first step would be to learn what the normal wiring diagram is [...] it should be feasible to do many additional connectomes [...] of animal models of brain disorders

<!-- *Emphasis added* -->


<!-- _footer: Abbott et al. Cell (2020) -->


<!-- TODO: (3) diagram of linking connectome and memory -->


<!-- --- 
# Connectome $\leftrightarrow$ {development, genetics}

> ... we selectively altered the location or activity of [...] neurons and generated new EM volumes of the manipulated samples **to investigate the effects on connectivity**.

*Emphasis added* -->

---
# Connectome $\leftrightarrow$ development
![center h:475](./../../images/witvliet-fig1.png)

<!-- _footer: Witvliet et al. Nature (2021) -->

---

# Why is comparative connectomics hard?

## Collecting the data is still a large effort...

## But how do we even compare connectomes once we have them?
- ~~Data are networks~~
  - Data are networks with rich attributes
- Data will always have noise
  - "Experimental noise"
  - "Biological noise"
- Data are big (and getting bigger)
 
--- 

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

# Outline

- ### **Larval connectome dataset**
- ### Connectome comparison via network hypothesis testing
- ### Pairing neurons across connectomes via graph matching
- ### Ongoing extensions/applications

---
<!-- Timing: ~6 min -->

# Larval _Drosophila_ brain connectome 


<div class="columns">
<div>

<style scoped>
p {
  justify-content: center;
  text-align: center;
  padding-top: 0px;
  margin-top: 0px;
}
</style>

![center h:400](./../../images/Figure1-brain-render.png)
~3k neurons, ~550K synapses
**Both hemispheres**

</div>
<div>

<!-- ![center h:500](./../../../results/figs/plot_layouts/whole-network-layout.png) -->
![h:450](./../../../results/figs/show_data/adjacencies.png)

</div>
</div>

<!-- _footer: Winding, Pedigo et al. Submitted (2022) -->

---

# Bilateral symmetry

> "This brain is bilaterally symmetric."
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  -Neuroscientists

> "What does that even mean? And how would we know if it wasn't?"
> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  -Us

<!-- ![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg) -->
<!-- ![bg opacity:.6 95%](./../../../results/figs/plot_side_layouts/2_network_layout.png) -->

<style scoped>
h2 {
    justify-content: center;
    text-align: center;
}
</style>

## Are the <span style="color: var(--left)"> left </span> and <span style="color: var(--right)"> right </span> sides of this connectome <p> </p> *different*?

--- 

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

# Outline

- ### Larval connectome dataset
- ### **Connectome comparison via network hypothesis testing**
- ### Pairing neurons across connectomes via graph matching
- ### Ongoing extensions/applications

---

# Are these populations different?

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
</div>

--- 
# Are these _networks_ different?

<div class="columns">
<div>

![center w:1000](./../../../results/figs/plot_side_layouts/2_network_layout.png)

</div>
<div>


- Want a two-network-sample test!
- <span style='color: var(--left)'> $A^{(L)} \sim F^{(L)}$</span>, <span style='color: var(--right)'> $A^{(R)} \sim F^{(R)}$ </span>
- $H_0: \color{#66c2a5} F^{(L)} \color{black} = \color{#fc8d62}F^{(R)}$  
  $H_A: \color{#66c2a5} F^{(L)} \color{black} \neq  \color{#fc8d62} F^{(R)}$

</div>
</div>

---
# Assumptions
- Know the direction of synapses, so network is *directed*
- For simplicity (for now), consider networks to be *unweighted*
- For simplicity (for now), consider the <span style='color: var(--left)'> left $\rightarrow$ left </span> and <span style='color: var(--right)'> right $\rightarrow$ right </span> (*ipsilateral*) connections
- Not going to assume any nodes are matched

![center h:250](../../../results/figs/unmatched_vs_matched/unmatched_vs_matched.svg)

---
# Erdos-Renyi model
<!-- Timing: ~10 -->

- All edges are independent
- All edges generated with the same probability, $p$

![center](../../../results/figs/er_unmatched_test/er_explain.svg)

---
# Detect a difference in density

<div class="columns">
<div>

![center h:500](./../../../results/figs/er_unmatched_test/er_methods.svg)


</div>
<div>

![center h:400](../../../results/figs/er_unmatched_test/er_density.svg)

<style scoped>
p {
  justify-content: center;
  text-align: center;
}
</style>

p-value < $10^{-22}$


</div>
</div>

--- 
# Stochastic block model

Edge probabilities are a function of a neuron's group

![center h:450](./../../../results/figs/sbm_unmatched_test/sbm_explain.svg)

---
# Connection probabilities between groups

<style scoped>
    .columns {
        display: grid;
        grid-template-columns: 1fr 2fr;
        gap: 0rem;
    }
</style>


<div class="columns">
<div>

![center h:500](./../../images/Figure1-cell-classes-vertical.png)

</div>
<div>


![center w:700](../../../results/figs/sbm_unmatched_test/sbm_uncorrected.svg)


</div>
</div>

--- 
# Group connection test

![center](./../../../results/figs/sbm_unmatched_test/sbm_methods_explain.svg)


--- 
# Detect differences in group connection probabilities

<div class="columns">
<div>

![center h:450](sbm_unmatched_test/../../../../results/figs/sbm_unmatched_test/sbm_uncorrected_pvalues_unlabeled.svg)

</div>
<div>

- 5 group-to-group connections are significantly different (after multiple comparisons correction)
- Overall test (comparing all blocks):<br> p-value $<10^{-7}$

</div>
</div>

---
# Should we be surprised?
<div class="columns">
<div>

- Already saw that even the overall densities were different
- For all significant comparisons, probabilities on the right hemisphere were higher
- Maybe the right is just a "scaled up" version of the left?
   - $H_0: \color{#66c2a5}B^{(L)} \color{black}  = c \color{#fc8d62}B^{(R)}$  
  where $c$ is a density-adjusting constant, $\frac{\color{#66c2a5} p^{(L)}}{\color{#fc8d62} p^{(R)}}$

</div>
<div>

![center h:500](./../../../results/figs/sbm_unmatched_test/significant_p_comparison.svg)

</div>
</div>


---
# After adjusting for density, differences are in KCs

<div class="columns">
<div>

![center](./../../../results/figs/adjusted_sbm_unmatched_test/adjusted_methods_explain.svg)

</div>
<div>

![h:450](./../../../results/figs/adjusted_sbm_unmatched_test/sbm_pvalues_unlabeled.svg)

<style scoped>
p {
    justify-content: center;
    text-align: center;
}
</style>

Overall p-value: $<10^{-2}$

</div>
</div>

---
# When we remove KCs...

<div class="columns">
<div>

![center h:450](./../../../results/figs/kc_minus/kc_minus_methods.svg)


</div>
<div>

- Density test: 
  $p <10^{-26}$
- Group connection test:
  $p <10^{-2}$
- **Density-adjusted group connection test: 
  $p \approx 0.51$**

</div>
</div>

---
# To sum up...

> "This brain is bilaterally symmetric."
>   &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;  -Neuroscientists

Depends on what you mean...

<div class="columns">
<div>

#### With Kenyon cells
|   Model   |                       $H_0$ (vs. $H_A \neq$)                       |    p-value    |
| :-------: | :----------------------------------------------------------------: | :-----------: |
|  **ER**   |  $\color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$  | ${<}10^{-23}$ |
|  **SBM**  | $\color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$  | ${<}10^{-7}$  |
| **daSBM** | $\color{#66c2a5}B^{(L)} \color{black}  = c \color{#fc8d62}B^{(R)}$ | ${<}10^{-2}$  |


</div>
<div>

#### Without Kenyon cells
|   Model   |                       $H_0$ (vs. $H_A \neq$)                       |    p-value     |
| :-------: | :----------------------------------------------------------------: | :------------: |
|  **ER**   |  $\color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$  | ${<}10^{-26}$  |
|  **SBM**  | $\color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$  |  ${<}10^{-2}$  |
| **daSBM** | $\color{#66c2a5}B^{(L)} \color{black}  = c \color{#fc8d62}B^{(R)}$ | $\approx 0.51$ |

</div>
</div>


---
# Examining the effect of edge weights

![center h:500](./../../../results/figs/thresholding_tests/thresholding_methods.png)

<!-- ---
# What is an edge weight anyway?

![center](./../../../results/figs/thresholding_tests/weight_notions.svg) -->

---

# Highest edge weight networks show no asymmetry

<!-- <div class='columns'>
<div>

## Synapse count thresholding  -->

<!-- ![center h:400](./../../../results/figs/thresholding_tests/synapse_threshold_pvalues.svg) -->

<!-- </div>
<div>

## Input proportion thresholding -->
<!-- 
</div>
</div> -->

![center h:500](./../../../results/figs/thresholding_tests/input_threshold_pvalues_legend.svg)


--- 

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

# Outline

- ### Larval connectome dataset
- ### Connectome comparison via network hypothesis testing
- ### **Pairing neurons across connectomes via graph matching**
- ### Ongoing extensions/applications
---

<!-- Timing: 23:00  -->

# Bilaterally homologous neuron pairs 

We believe a matching exists!

![center](./../../images/mbon-expression.jpg)

<!-- _footer: Eschbach et al. eLife (2021) -->

---

# Can we use network structure to predict this pairing?

<div class="columns">
<div>

![](../../images/the-wire.png)


</div>
<div>

- Week 1: observe a network ($A$) of phone #s and the calls they make to each other
- Week 2: all of the #s change! But a (noisy) version of that network still exists, with different labels... ($B$)
- How to map nodes of network $A$ to those of network $B$?

</div>
</div>


<!-- _footer: The Wire Season 3 Episode 7, HBO -->

--- 
# What is graph matching?

![center](../../images/network-matching-explanation.svg)

---
# How do we measure network overlap?


<style scoped>
h2 {
    justify-content: center;
    text-align: center;
}
</style>

<div class="columns">
<div>

## $\min_{P \in \mathcal{P}} \underbrace{\|A - \overbrace{PBP^T}^{\text{reordered } B}\|_F^2}_{\text{distance between adj. mats.}}$

where $\mathcal{P}$ is the set of permutation matrices

<!-- TODO: (3) make a figure diagraming a permutation/matching of nodes -->

</div>
<div>

- Measures the number of edge disagreements for unweighted networks,
- Norm of edge disagreements for weighted networks

</div>
</div>

---
# How do we do graph matching?

- Relax the problem to a continuos space
  - Convex hull of permutation matrices
- Minimize a linear approximation of objective function (repeat)
- Project back to the closest permutation matrix

<!-- _footer: Vogelstein et al. PLOS One (2015) -->

---
# Matching (by connectivity only) performs fairly well

![center h:450](./../../_build/html/_images/left-pair-predictions.svg)


<style scoped>
p {
    justify-content: center;
    text-align: center;
}
</style>

With "vanilla" graph matching: ~80% correct (according to expert annotator)

---
# Many ways to try to improve on this...

<div class="columns">
<div>

- Edge types allow for "multilayer" graph matching
- Partial knowledge of the matching (seeds)
- Morphology (e.g. NBLAST)

</div>
<div>

![h:400](./../../images/fig2-connection-types.png)

*Summary of "edge types" based on neuron compartments*

</div>
</div>

<!-- _footer: Pantazis et al. Applied Network Science (2022), Fishkind et al. Pattern Recognition (2019), Winding, Pedigo et al. Submitted (2022) -->

---

<style scoped>
h1, h2 {
    padding-top: 140px;
    justify-content: center;
    text-align: center;
}
</style>

# Thus far, we've not used the contralateral connections

## These are about 1/3 of the edges in the brain!

---
# From graph matching to bisected graph matching

![](../../images/explain-bgm.svg)


<!-- _footer: Pedigo et al. bioRxiv (2022) -->

--- 
# Contralateral connections are helpful!

![center h:550](../../images/match_accuracy_comparison.svg)

<!-- _footer: Pedigo et al. bioRxiv (2022) -->

---
<!-- Timing: 31:00 -->
# Performance improvement on the full brain
![center](./../../images/matching_accuracy_upset.svg)

--- 

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

# Outline

- ### Larval connectome dataset
- ### Connectome comparison via network hypothesis testing
- ### Pairing neurons across connectomes via graph matching
- ### **Ongoing extensions/applications**
  
---
# Pairs facilitate more powerful tests

<div class="columns-br">
<div>

- Generate an Erdos-Renyi network ($A$)
- Perturb a copy of it ($B$) (add edges)
- Test for differences between $A$ and $B$

</div>
<div>

![](./../../../results/figs/matched_vs_unmatched_sims/er_power_comparison.svg)


</div>
</div>

---
# Testing for "stereotypy" in edge structure

Is matching stronger than expected under some model of independent networks?

<div class="columns">
<div>

![](../../images/kc-stereotypy-diagram.svg)

</div>
<div>

![](../../images/kc_alignment_dist.svg)

</div>
</div>

<!-- _footer: Eichler et al. Nature (2017), Fishkind et al. Applied Network Science (2021) -->

---

# Neurons clustered by connectivity using recursive spectral clustering

Where to stop splitting?

![center](../../images/bar-dendrogram-wide.svg)

![center w:700](../../images/cell-type-labels-legend.png)

<!-- _footer: Winding, Pedigo et al. Submitted (2022) -->

---
# Using *pairs* and *models* to evaluate cell type groupings
<!-- TODO: (2) diagram/describe SBM cross validation -->

<!-- ![center h:550](../../images/lik-by-n_params-blind.png) -->

<div class="columns">
<div>

- Clustering nodes corresponds with inferring groups in a stochastic block model (DCSBM)...
- How well do these models generalize to the other side of the brain (let alone the next maggot)?

</div>
<div>

![center h:550](../../images/dcsbm-swap-arrows.png)

</div>
</div>


--- 

# Summary 
<!-- 41 min -->
<div class="columns-br">
<div>

![](./../../../results/figs/draw_brain_comparisons/brain_approx_equals.png)

</div>
<div>

- Model-based network comparison enables testing (and refining) hypotheses about connectomes
  - We proposed a few tests, but just the beginning! 

</div>
</div>

<div class="columns-br">
<div>

![](./../../../results/figs/draw_brain_comparisons/brain_matching.png)

</div>
<div>

- Graph matching can pair neurons across datasets
  - Helpful to adapt off-the-shelf algos. to use biological info (e.g contralaterals, edge types)
</div>
</div>


**Aim to apply these (and other) tools to make inferences from connectome comparisons!**

---

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

# How to use these (and other) tools?

<div class="columns">
<div>

## graspologic

[github.com/microsoft/graspologic](https://github.com/microsoft/graspologic)

![w:450](./../../images/graspologic_svg.svg)

[![h:30](https://pepy.tech/badge/graspologic)](https://pepy.tech/project/graspologic)  [![h:30](https://img.shields.io/github/stars/microsoft/graspologic?style=social)](https://github.com/microsoft/graspologic)  [![h:30](https://img.shields.io/github/contributors/microsoft/graspologic)](https://github.com/microsoft/graspologic/graphs/contributors)

</div>
<div>

## Model-based testing
[github.com/neurodata/bilateral-connectome](github.com/neurodata/bilateral-connectome)
[![h:30](https://jupyterbook.org/badge.svg)](http://docs.neurodata.io/bilateral-connectome/)


## Improved matching
[github.com/neurodata/bgm](github.com/neurodata/bgm)
[![h:30](https://jupyterbook.org/badge.svg)](http://docs.neurodata.io/bilateral-connectome/)

(Or for WIP final implementation see
[github.com/microsoft/graspologic/pull/960](github.com/microsoft/graspologic/pull/960))

</div>
</div>

<!-- _footer: Chung, Pedigo et al. JMLR (2019) -->

--- 

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

# Acknowledgements

#### Team

<style scoped> 

p {
    font-size: 24px;
}
</style>


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

</div>

Tracers who contributed to larva connectome, Heather Patsolic, Youngser Park, NeuroData lab, Microsoft Research
Figures from Scidraw + Noun Project (Alexander Bates, Xuan Ma, Gil Costa, Vivek Kumar, Leslie Coonrod)

#### Funding
NSF Graduate Research Fellowship (B.D.P.), NSF CAREER Award (J.T.V.), NSF NeuroNex Award (J.T.V and C.E.P.), NIH BRAIN Initiative (J.T.V.)

---
# Questions?

![bg opacity:.6 95%](./../../../results/figs/plot_side_layouts/2_network_layout.png)

#### Slides: 
#### [tinyurl.com/princeton-bilarva](https://tinyurl.com/princeton-bilarva)


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
![icon](../../images/email.png) [bpedigo@jhu.edu](mailto:bpedigo@jhu.edu)
![icon](../../images/github.png) [@bdpedigo (Github)](https://github.com/bdpedigo)
![icon](../../images/twitter.png) [@bpedigod (Twitter)](https://twitter.com/bpedigod)
![icon](../../images/web.png) [bdpedigo.github.io](https://bdpedigo.github.io/)
