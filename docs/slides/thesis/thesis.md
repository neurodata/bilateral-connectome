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

# Analytical tools for comparative connectomics

## Thesis committee meeting

<br>

<div class="columns">
<div>

### Benjamin D. Pedigo
(he/him)
[NeuroData lab](https://neurodata.io/)
Johns Hopkins University
Department of Biomedical Engineering


![icon](../../images/email.png) [bpedigo@jhu.edu](mailto:bpedigo@jhu.edu)
![icon](../../images/github.png) [@bdpedigo (Github)](https://github.com/bdpedigo)
![icon](../../images/twitter.png) [@bpedigod (Twitter)](https://twitter.com/bpedigod)
![icon](../../images/web.png) [bdpedigo.github.io](https://bdpedigo.github.io/)

</div>
<div>

</div>
</div>

---

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

# Outline

- Prior work 
  - **Connectome of an insect brain**
  - Analysis of bilateral symmetry
  - Matching neurons between brain hemispheres 
  - graspologic
- Future work
  - Collaborative investigations
  - On the power of comparative connectomics
  - Plan towards graduation

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

<!-- ![center h:500](./../../../results/figs/plot_layouts/whole-network-layout.png) -->
![h:450](./../../../results/figs/show_data/adjacencies.png)

</div>
</div>

<!-- _footer: Winding, Pedigo et al. *Under review, Science* (2022) -->


---

# Neurons clustered by connectivity using recursive spectral clustering

Where to stop splitting?

![center](../../images/bar-dendrogram-wide.svg)

![center w:700](../../images/cell-type-labels-legend.png)

<!-- _footer: Winding, Pedigo et al. *Under review, Science*  (2022) -->

---
# Using *pairs* and *models* to evaluate cell type groupings

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
# TODO: show something else on the maggot brain

---

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

# Outline

- Prior work 
  - Connectome of an insect brain
  - **Analysis of bilateral symmetry**
  - Matching neurons between brain hemispheres 
  - graspologic
- Future work
  - Collaborative investigations
  - On the power of comparative connectomics
  - Plan towards graduation

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
# Stochastic block model

Edge probabilities are a function of a neuron's group

![center h:450](./../../../results/figs/sbm_unmatched_test/sbm_explain.svg)


--- 
# Group connection test

![center](./../../../results/figs/sbm_unmatched_test/sbm_methods_explain.svg)

<!-- _footer: Pedigo et al. *In preparation, eLife* -->

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

<!-- _footer: Pedigo et al. *In preparation, eLife* -->


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

<!-- _footer: Pedigo et al. *In preparation, eLife* -->

---

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

# Outline

- Prior work 
  - Connectome of an insect brain
  - Analysis of bilateral symmetry
  - **Matching neurons between brain hemispheres**
  - graspologic
- Future work
  - Collaborative investigations
  - On the power of comparative connectomics
  - Plan towards graduation

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

<!-- _footer: Vogelstein et al. *PLOS One* (2015) -->

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

We show how BGM can be solved via a simple modification of the FAQ algorithm

<!-- _footer: Pedigo et al. *bioRxiv, in revision at Network Neuro* (2022) -->

--- 
# Contralateral connections are helpful!

![center h:550](../../images/match_accuracy_comparison.svg)

<!-- _footer: Pedigo et al. *bioRxiv, in revision at Network Neuro* (2022) -->

---


![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

# Outline

- Prior work 
  - Connectome of an insect brain
  - Analysis of bilateral symmetry
  - Matching neurons between brain hemispheres
  - **graspologic**
- Future work
  - Collaborative investigations
  - On the power of comparative connectomics
  - Plan towards graduation


---

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

</div>
</div>

<!-- _footer: Chung, Pedigo et al. *JMLR* (2019) -->

---

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

# Outline

- Prior work 
  - Connectome of an insect brain
  - Analysis of bilateral symmetry
  - Matching neurons between brain hemispheres
  - graspologic
- Future work
  - **Collaborative investigations**
  - On the power of comparative connectomics
  - Plan towards graduation

---
# Male adult nerve cord connectome

<div class="columns">
<div>

![center h:400](./../../images/vnc.png)

</div>
<div>

- Collaboration with Jefferis' group
- Similar questions to maggot brain, e.g., 
  - Connectivity types
  - Matching neurons 
- Segmental homology, bilateral symmetry

</div>
</div>

<!-- _footer: Adapted from Phelps et al. *Cell* 2021 -->

--- 
# Linking behaviorally characterized neurons to a connectome

<div class="columns-bl">
<div>

![](../../lalanti-talk.png)

</div>
<div>

- Collab. w/ Zlatic group
- ~70 neuron pairs linked to connectome
- Goal: compare neighborhoods of each, detect signatures of "similar behavior neurons"

</div>
</div>

<!-- _footer: Adapted from Lalanti Venkatasubramanian -->

--- 
# Testing for edge-level stereotypy structure

---

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

# Outline

- Prior work 
  - Connectome of an insect brain
  - Analysis of bilateral symmetry
  - Matching neurons between brain hemispheres
  - graspologic
- Future work
  - Collaborative investigations
  - **On the power of comparative connectomics**
  - Plan towards graduation

---

# Many goals of connectomics involve linking connectome to other properties

![center h:500](./../../images/link_connectome.svg)

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

![center h:200](../../images/mind-of-a-mouse.png)

### Connectome $\leftrightarrow$ memory

> ...the acquisition of wiring diagrams across multiple individuals will yield insights into how experiences shape neural connections.

### Connectome $\leftrightarrow$ disease

> The first step would be to learn what the normal wiring diagram is [...] it should be feasible to do many additional connectomes [...] of animal models of brain disorders

<!-- _footer: Abbott et al. *Cell* (2020) -->

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
# A proposed experiment...

<div class="columns">
<div>

![](./../../images/mushroom-body-learning.png)

</div>
<div>

- Optogenetically activate mushroom body input neurons (DANs) which convey reward or punishment
- Map connectomes, look for changes in KC $\rightarrow$ MBON connectivity
- *How big does the {sample size, effect size} need to be such that we can detect a change 95% of the time?*

</div>
</div>

<!-- _footer: Modi, Shuai et al. *Ann. Rev. of Neuro.* (2020) -->

--- 
# On the power of comparative connectomics

- Still many extensions to simple network comparison tests we proposed in *Pedigo et al.* (bilateral symmetry manuscript)
  - Tests for weighted models
  - Tests for paired/matched networks

---

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

# Outline

- Prior work 
  - Connectome of an insect brain
  - Analysis of bilateral symmetry
  - Matching neurons between brain hemispheres
  - graspologic
- Future work
  - Collaborative investigations
  - On the power of comparative connectomics
  - **Plan towards graduation**

---
# Summary of work so far

<style scoped> 

ul,p {
    font-size: 26px;
}
</style>


<div class="columns">
<div>

#### (Co)-first papers/manuscripts
- Graspologic, JMLR (2019)
- Maggot brain, in review at *Science*
- Bilateral symmetry, submitting to *eLife*
- Bisected matching, resubmission at *Net. Neuro.* 

#### Other papers/manuscripts
- 3 published reviews
- 3 manuscripts in review

</div>
<div>

<!-- - Graph matching via optimal transport, under review at *Pattern Recognition Letters*
- Statistical connectomics -->

#### Conference presentations 
- Berlin Connectomics (x2)
- Neuromatch (x2)
- NAISys (x2)

#### Invited talks
- Drexel BME/Neuro seminar
- Princeton Murthy/Seung labs meeting

#### Awards
- NSF GRFP Fellowship
- BRAIN Initiative Trainee Highlight Award

</div>
</div>



---


<style scoped> 

ul,p {
    font-size: 26px;
}
</style>


# Summary of work to be done

<div class="columns">
<div>

#### Manuscripts
- 1-2 collaborations with experimentalists (e.g., MANC connectome, behavior + larva connectome)
- "On the power of comparative connectomics" 

#### Conferences 
SfN, COSYNE, OHBM, NetSci


</div>
<div>

#### Graduation
*June 2022*


</div>
</div>


---
---
---
---







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
