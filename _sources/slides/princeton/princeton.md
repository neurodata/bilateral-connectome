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
# Tools for comparative connectomics: <br> case studies from two sides of a larval Drosophila brain

<br>

<div class="columns">
<div>

## Benjamin D. Pedigo
(he/him)
[NeuroData lab](https://neurodata.io/)
Johns Hopkins University - Biomedical Engineering


![icon](../../images/email.png) [_bpedigo@jhu.edu_](mailto:bpedigo@jhu.edu)
![icon](../../images/github.png) [_@bdpedigo (Github)_](https://github.com/bdpedigo)
![icon](../../images/twitter.png) [_@bpedigod (Twitter)_](https://twitter.com/bpedigod)
![icon](../../images/web.png) [bdpedigo.github.io](https://bdpedigo.github.io/)

</div>
<div>

### These slides at: 
### [tinyurl.com/princeton-bilarva](https://tinyurl.com/princeton-bilarva)


</div>
</div>

---

# Connectomics is useful...

TODO: (3) plot of "connectome prevalence over time"

TODO: (4) highlight one example fly result (? maybe central complex)

---

# Many stated goals of connectomics are to link connectome to other properties

![center h:500](./../../images/link_connectome.svg)

<!-- TODO: (3) finish this figure draw arrows -->

<!-- /Users/bpedigo/JHU_code/pcc/pcc/results/figs/diagram/link_connectome.png -->

<!-- ![](../../../../../pcc/pcc/results/figs/diagram/link_connectome.svg) -->


<!-- ![](Users/bpedigo/JHU_code/pcc/pcc/results/figs/diagram/link_connectome.png) -->

---

# Connectome $\leftrightarrow$ memory

> ...the acquisition of wiring diagrams across multiple individuals will yield insights into **how experiences shape neural connections.**

*Emphasis added*

<!-- TODO: (2) diagram of linking connectome and memory -->

<!-- _footer: Abbott et al. Cell (2020) -->

---
# Connectome $\leftrightarrow$ evolution

> Comparative connectomics of [...] **species across the phylogenetic tree** can infer the archetypal neural architecture of each bauplan and identify any circuits that possibly converged onto a shared and potentially optimal, structure.

*Emphasis added*

<!-- TODO: (2) diagram of comparing connectomes across species -->

<!-- _footer: Barsotti + Correia et al. Curr. Opi. Neurobio. (2021)-->

---
# Connectome $\leftrightarrow$ development
![center h:475](./../../images/witvliet-fig1.png)

<!-- _footer: Witvliet et al. Nature (2021) -->

---

# But it is methodologically hard to compare connectomes!
- ~~Data are networks~~
  - Data are networks with rich attributes
- Data are noisy
  - "Developmental noise" 
  - "Experimental noise"
- Data are big (and getting bigger)
 
--- 

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

# Outline for today

- Describe larval connectome dataset
- Show how connectome comparison can be framed as network hypothesis testing
- Show how we can use automated tools for predicting the correspondence of neurons across datasets
- Mention extensions to use/combine/extend these tools

---

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

![center h:500](./../../../results/figs/plot_layouts/whole-network-layout.png)

</div>
</div>

<!-- _footer: Winding, Pedigo et al. Submitted (2022) -->

---

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)
<!-- ![bg opacity:.6 95%](./../../../results/figs/plot_side_layouts/2_network_layout.png) -->

<style scoped>
section {
    justify-content: center;
    text-align: center;
}
</style>

# Are the <span style="color: var(--left)"> left </span> and <span style="color: var(--right)"> right </span> sides of this connectome <p> </p> *different*?


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
# Are these two _networks_ different?

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
- We know the direction of synapses, so network is *directed*.
- For simplicity (for now), consider networks to be *unweighted*.
- For simplicity (for now), consider the <span style='color: var(--left)'> left $\rightarrow$ left </span> and <span style='color: var(--right)'> right $\rightarrow$ right </span> (*ipsilateral*) connections only.
- Not going to assume any nodes are matched

![center h:250](../../../results/figs/unmatched_vs_matched/unmatched_vs_matched.svg)

---
# Erdos-Renyi model
<!-- about 10 minutes to this point -->

- All edges are indepentent
- All edges generated with the same probability, $p$

![center](../../../results/figs/er_unmatched_test/er_explain.svg)

<!-- --- -->
<!-- # Density-based testing -->

<!-- <div class="columns">
<div> -->

<!-- - $P[i \rightarrow j] = p$ -->
<!-- - Compare probabilities:
  $H_0: \color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$  
  $H_A: \color{#66c2a5} p^{(L)} \color{black} \neq  \color{#fc8d62} p^{(R)}$ -->

<!-- </div>
<div> -->


<!-- </div>
</div> -->

---
# We detect a difference in density

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

- Edge probabilities are a function of a neuron's group

![center h:450](./../../../results/figs/sbm_unmatched_test/sbm_explain.svg)

---
# Connection probabilities between groups

<!-- ![center h:160](./../../images/Figure1-cell-classes.png)

![center h:350](../../../results/figs/sbm_unmatched_test/sbm_uncorrected.svg) -->
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
# Group-based testing

![center](./../../../results/figs/sbm_unmatched_test/sbm_methods_explain.svg)


--- 
# Detect a difference in group connection probabilities

<div class="columns">
<div>

![center h:450](sbm_unmatched_test/../../../../results/figs/sbm_unmatched_test/sbm_uncorrected_pvalues_unlabeled.svg)

</div>
<div>

- After multiple comparison, find 5 group-to-group connections which are significantly different
- Combine (uncorrected) p-values (like a meta-analysis), leads to p-value for overall test of $<10^{-7}$

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

<!-- ---
# Adjusting for a difference in density -->


<!-- ---
# Even with density adjustment, we detect a difference

<div class="columns">
<div>

![center](./../../../results/figs/adjusted_sbm_unmatched_test/resampled_pvalues_distribution.svg)

</div>
<div>

![center w:500](./../../../results/figs/adjusted_sbm_unmatched_test/sbm_pvalues_unlabeled.svg)

</div>
</div> -->

---
# After adjusting for density, differences are in KCs

<div class="columns">
<div>

![center](./../../../results/figs/adjusted_sbm_unmatched_test/adjusted_methods_explain.svg)

</div>
<div>

![h:500](./../../../results/figs/adjusted_sbm_unmatched_test/sbm_pvalues_unlabeled.svg)

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
  $p \approx 0.003$
- **Density-adjusted group connection test: 
  $p \approx 0.43$**

</div>
</div>

---
# To sum up...

<!-- <style scoped>
table {
    font-size: 24px;
    margin-bottom: 50px;
}
</style> -->

<!-- | Model | $H_0$ (vs. $H_A \neq$)                                             |  KCs  |     p-value     | Interpretation                                           |
| ----- | ------------------------------------------------------------------ | :---: | :-------------: | -------------------------------------------------------- |
| ER    | $\color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$   |   +   |   $<10^{-23}$   | Reject densities the same                                |
| SBM   | $\color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$  |   +   |   $< 10^{-7}$   | Reject group connection probabilities the same           |
| aSBM  | $\color{#66c2a5}B^{(L)} \color{black}  = c \color{#fc8d62}B^{(R)}$ |   +   | $\approx 0.002$ | Reject above even after accounting for density           |
| ER    | $\color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$   |   -   |   $<10^{-26}$   | Reject densities the same (w/o KCs)                      |
| SBM   | $\color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$  |   -   | $\approx 0.003$ | Reject group connection probabilities the same (w/o KCs) |
| aSBM  | $\color{#66c2a5}B^{(L)} \color{black}  = c \color{#fc8d62}B^{(R)}$ |   -   | $\approx 0.43$  | Don't reject above after density adjustment (w/o KCs)    | -->

<div class="columns">
<div>

#### With Kenyon cells
| Model |                       $H_0$ (vs. $H_A \neq$)                       |    p-value    |
| :---: | :----------------------------------------------------------------: | :-----------: |
| **ER** |  $\color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$  | ${<}10^{-23}$ |
| **SBM** | $\color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$  | ${<}10^{-7}$  |
| **daSBM** | $\color{#66c2a5}B^{(L)} \color{black}  = c \color{#fc8d62}B^{(R)}$ | ${<}10^{-2}$  |


</div>
<div>

#### Without Kenyon cells
| Model |                       $H_0$ (vs. $H_A \neq$)                       |    p-value    |
| :---: | :----------------------------------------------------------------: | :-----------: |
| **ER** |  $\color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$  | ${<}10^{-26}$ |
| **SBM** | $\color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$  | ${<}10^{-2}$  |
| **daSBM** | $\color{#66c2a5}B^{(L)} \color{black}  = c \color{#fc8d62}B^{(R)}$ |  $\approx 0.51$     |

</div>
</div>


---
# Examining the effect of edge weights

<div class='columns'>
<div>

![](./../../../results/figs/thresholding_tests/thresholding_methods.png)

</div>
<div>

![](./../../../results/figs/thresholding_tests/input_threshold_pvalues_legend.svg)

</div>
</div>


---

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

<style scoped>
section {
  justify-content: center;
  text-align: center;
}
</style>

# Matching neurons

---
# Bilaterally homologous neuron pairs 
![center](./../../images/mbon-expression.jpg)
<!-- Timing: 23:00

<!-- _footer: Eschbach et al. eLife (2021) -->

---

# Can we use networks structure to predict this pairing?

<div class="columns">
<div>

![](../../images/the-wire.png)


</div>
<div>

- Week 1: observe a network of cell phone #s and the calls they make to each other
- Week 2: all of the #s change! But a (noisy) version of that network still exists, with different labels...
- How to map network from week 1 ($A$) to network from week 2 ($B$)?

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

<!-- ---
# How do we do graph matching?

_footer: Vogelstein et al. PLOS One (2015) -->

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

<!-- _footer: Pantazis et al. Applied Network Science (2022), Fishkind et al. Pattern Recognition (2019) -->

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

<!-- ![](../../../results/figs/explain/explain.svg) -->

![](../../images/explain-bgm.svg)

<!-- --LL: #66c2a5;
    --RR: #fc8d62;
    --source: #8da0cb;
    --target: #e78ac3;
    --LR: #8da0cb;
    --RL: #e78ac3; -->

<!-- New gradient: 

$$\nabla f(P) = - \textcolor{#66c2a5}{A_{LL}} P \textcolor{#fc8d62}{A_{RR}}^T + \textcolor{#66c2a5}{A_{LL}}^TP\textcolor{#fc8d62}{A_{RR}} + \textcolor{#8da0cb}{A_{LR}} P^T \textcolor{#e78ac3}{A_{RL}}^T + \textcolor{#e78ac3}{A_{RL}}^T P^T \textcolor{#8da0cb}{A_{LR}}$$ -->

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

<style scoped>
section {
  justify-content: center;
  text-align: center;
}
</style>

# Extensions and ongoing work

---
# Pairs facilitate more powerful tests

<div class="columns-br">
<div>

- Generate an Erdos-Renyi network ($A$)
- Perturb a copy of it ($B$)
- Test for differences between $A$ and $B$

</div>
<div>

![](./../../../results/figs/matched_vs_unmatched_sims/er_power_comparison.svg)


</div>
</div>

---
# Testing for "stereotypy" at the edge level

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

<!-- # The value of pairs - looking at models -->

# Hierarchical clustering of neurons by connectivity

![center](../../images/bar-dendrogram-wide.svg)

![center w:700](../../images/cell-type-labels-legend.png)

<!-- _footer: Winding, Pedigo et al. Submitted (2022) -->

---
# Using *pairs* and *models* to evaluate cell type groupings
<!-- TODO: (3) fix up SBM cross validation curve  -->
<!-- TODO: (2) diagram/describe SBM cross validation -->

![center h:550](../../images/lik-by-n_params-blind.png)

--- 
<!-- ![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg) -->

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
  - Can incorporate biological info like morphology, partial matchings, contralateral connections, etc. 
</div>
</div>


- **Aim to apply these (and other) tools to understand similarities/differences between connectomes**

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
github.com/neurodata/bilateral-connectome
[![h:30](https://jupyterbook.org/badge.svg)](http://docs.neurodata.io/bilateral-connectome/)


## Improved matching
github.com/neurodata/bgm
[![h:30](https://jupyterbook.org/badge.svg)](http://docs.neurodata.io/bilateral-connectome/)

(Or for WIP final implementation see
github.com/microsoft/graspologic/pull/960)

</div>
</div>

<!-- ## graspologic
## bilateral repo 
## bgm 
## get in touch!  -->

--- 

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

# Acknowledgements

#### Team

<style scoped> 

p {
    font-size: 24px;
}
</style>


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

Tracers who contributed to larva connectome, NeuroData lab, Microsoft Research

#### Funding
NSF Graduate Research Fellowship (Grant no. DGE1746891) to B.D.P.
NSF CAREER Award (Grant no. 1942963) to J.T.V.
NSF NeuroNex Award (Grant no. 2014862) to J.T.V and C.E.P.
NIH BRAIN Initiative (Grant no. 1RF1MH123233-01) to J.T.V.

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
![icon](../../images/web.png) [bdpedigo.github.io](https://bdpedigo.github.io/) -->
