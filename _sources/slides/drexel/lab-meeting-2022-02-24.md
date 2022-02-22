---
marp: true
theme: default
paginate: true
style: |
    section {
        justify-content: flex-start;
        --orange: #ed7d31;
        --left: #66c2a5;
        --right: #fc8d62;
        --source: #8da0cb;
        --target: #e78ac3;
    }
    img[alt~="center"] {
        display: block;
        margin: 0 auto;
    }
    header {
        top: 0px;
        margin-top: auto;
    }
    .columns {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 1rem;
    }

---


<style scoped> 
p {
    font-size: 24px;
}
</style>

# Maggot brain, mirror image? A statistical analysis of bilateral symmetry in an insect brain connectome

### Benjamin D. Pedigo

_Johns Hopkins University_
_[NeuroData lab](https://neurodata.io/)_
[_@bdpedigo (Github)_](https://github.com/bdpedigo)
[_@bpedigod (Twitter)_](https://twitter.com/bpedigod)
[_bpedigo@jhu.edu_](mailto:bpedigo@jhu.edu)

![bg right:50% w:600](./../../images/temp-maggot-brain-umap-omni-hue_key=merge_class.png)

--- 
# My requests
- Feedback, feedback, feedback
   - Especially with figures 

---
# Electron microscopy connectomics

![center w:1000](./../../images/FigureS1-reconstruction.png)

<footer>
Winding, Pedigo et al. “The complete connectome of an insect brain.” In prep (2021)
</footer>


---

# Larval _Drosophila_ brain connectome


<div class="columns">
<div>

See [Michael Windings's talk](https://conference.neuromatch.io/abstract?edition=2021-4&submission_id=recVeh4RZFFRAQnIo)
- First whole-brain, single-cell connectome of any insect
- ~3000 neurons, ~544K synapses
- Both hemispheres of the brain reconstructed

</div>
<div>

![](./../../images/Figure1-brain-render.png)
<!-- ![w:600](./../../../results/figs/plot_layouts/whole-network-layout.png) -->

</div>
</div>

<footer>
Winding, Pedigo et al. “The complete connectome of an insect brain.” In prep (2021)
</footer>


---

<style scoped>
section {
    justify-content: center;
    text-align: center;
}
</style>

# Many connectomics questions require comparison

<!-- For instance,
- Understand connectomes across evolution [1]
- Understand connectomes across development [2]
- Understand links between genetics and connectivity [3] 

<footer>

[1] Bartsotti + Correia et al. *Curr. Op. Neurobiology* (2021)
[2] Witvliet et al. *Nature* (2021)
[3] Valdes-Aleman et al. *Neuron* (2021)

</footer> -->

---
# Connectomes across development
![center h:475](./../../images/witvliet-fig1.png)

<footer>

Witvliet et al. *Nature* (2021)

</footer>

---
# Connectomes across evolution, cortex
![center w:550](./../../images/bartsotti-fig2.png)

<footer>

Bartsotti + Correia et al. *Curr. Op. Neurobiology* (2021)

</footer>

---


<style scoped>
section {
    justify-content: center;
    text-align: center;
}
</style>

# Are the <span style="color: var(--left)"> left </span> and <span style="color: var(--right)"> right </span> sides of this connectome <p> </p> *the same*?

---

# Are these populations the same?

<div class="columns">
<div>

- Known as two-sample testing
- $\color{#66c2a5} Y^{(1)} \sim F^{(1)}$, $\color{#fc8d62} Y^{(2)} \sim F^{(2)}$
- $H_0: \color{#66c2a5} F^{(1)} \color{black} = \color{#fc8d62} F^{(2)}$  
  $H_A: \color{#66c2a5} F^{(1)} \color{black} \neq \color{#fc8d62} F^{(2)}$

</div>
<div>

![center w:400](./../../../results/figs/two_sample_testing/2_sample_real_line.svg)

</div>
</div>

--- 
# Are these two _networks_ the same?

<div class="columns">
<div>

- Want a two-network-sample test!

- <span style='color: var(--left)'> $A^{(L)} \sim F^{(L)}$</span>, <span style='color: var(--right)'> $A^{(R)} \sim F^{(R)}$ </span>
- $H_0: \color{#66c2a5} F^{(L)} \color{black} = \color{#fc8d62}F^{(R)}$  
  $H_A: \color{#66c2a5} F^{(L)} \color{black} \neq  \color{#fc8d62} F^{(R)}$

</div>
<div>

![center w:1000](./../../../results/figs/plot_side_layouts/2_network_layout.png)

</div>
</div>

---
# Assumptions
- We know the direction of synapses, so network is *directed*.
- For simplicity (for now), consider networks to be *unweighted*.
- For simplicity (for now), consider the <span style='color: var(--left)'> left $\rightarrow$ left </span> and <span style='color: var(--right)'> right $\rightarrow$ right </span> (*ipsilateral*) connections only.

---
# Density-based testing: Erdos-Renyi (ER) model

<!-- <div class="columns">
<div> -->

<!-- - $P[i \rightarrow j] = p$ -->
<!-- - Compare probabilities:
  $H_0: \color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$  
  $H_A: \color{#66c2a5} p^{(L)} \color{black} \neq  \color{#fc8d62} p^{(R)}$ -->

<!-- </div>
<div> -->

![center h:500](./../../../results/figs/er_unmatched_test/er_methods.png)

<!-- </div>
</div> -->

---
# We detect a difference in density

<div class="columns">
<div>

![](../../../results/figs/er_unmatched_test/er-density.png)

</div>
<div>

- p-value < $10^{-23}$


</div>
</div>

--- 
# Group-based testing: stochastic block model (SBM)

![center](./../../../results/figs/sbm_unmatched_test/sbm_methods_explain.svg)

---
# Connection probabilities between groups

![center h:200](./../../images/Figure1-cell-classes.png)

![center h:300](../../../results/figs/sbm_unmatched_test/sbm_uncorrected.svg)

--- 
# We detect a difference in group-to-group connection probabilities

<div class="columns">
<div>

![center h:450](sbm_unmatched_test/../../../../results/figs/sbm_unmatched_test/sbm_uncorrected_pvalues.svg)

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

---
# Adjusting for a difference in density

![center](./../../../results/figs/adjusted_sbm_unmatched_test/adjusted_methods_explain.svg)

---
# Even with an adjustment, still detect a difference

<div class="columns">
<div>

![center](./../../../overleaf/figs/adjusted_sbm_unmatched_test/resampled_pvalues_distribution.svg)

</div>
<div>

![center w:500](./../../../results/figs/adjusted_sbm_unmatched_test/sbm_uncorrected_pvalues.svg)

</div>
</div>

---
# So the Kenyon cells are the only group with remaining differences...

<div class="columns">
<div>

![center h:450](./../../../results/figs/kc_minus/kc_minus_methods.svg)


</div>
<div>

- ER test: $p <10^{-27}$
- SBM test: $p <...$
- Adjusted SBM test: $p \approx 0.43$

</div>
</div>

---

<style scoped>
section {
    justify-content: center;
    text-align: center;
}
</style>

# But wait, there's more (tests one could run)!

---
# To sum up...

<style scoped>
table {
    font-size: 24px;
    margin-bottom: 50px;
}
</style>

| Model | $H_0$ (vs. $H_A \neq$)                                             |   KC     | p-value | Interpretation                                     |
| ----- | ------------------------------------------------------------------ | :---: |:-----------: | -------------------------------------------------- |
| ER    | $\color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$   | + |  $<10^{-23}$  | Reject densities the same                          |
| SBM   | $\color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$  | + | $< 10^{-7}$  | Reject group connection probabilities the same |
| aSBM   | $\color{#66c2a5}B^{(L)} \color{black}  = c \color{#fc8d62}B^{(R)}$ | + | $\approx 0.0016$ | Reject above even after accounting for density  |
| ER    | $\color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$   | - |  $<10^{-27}$  | Reject densities the same (w/o KCs)                        |
| SBM   | $\color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$  | - | $< 10^{-4}$  | Reject group connection probabilities the same (w/o KCs) |
| aSBM   | $\color{#66c2a5}B^{(L)} \color{black}  = c \color{#fc8d62}B^{(R)}$ | - | $\approx 0.43$ | Don't reject after density adjustment (w/o KCs)  |

---
# More generally 
- We studied simple ways of framing a network two sample test
- We found that it can be important to "mod out" by other simple network statistics if you don't care about them (like density)
- We provide recommendations for what to run for you future connectome comparisons

--- 

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
![w:600](./../../images/jb_example.png)

</div>
</div>


<footer>Chung, Pedigo et al. JMLR (2019)</footer>

---
# Acknowledgements

<style scoped> 
p {
    font-size: 22px
}
h4 {font-size: 28px}
</style>

#### Johns Hopkins University
Mike Powell, Eric Bridgeford, Carey Priebe, Joshua Vogelstein, Kareef Ullah, Diane Lee, Sambit Panda, Jaewon Chung, Ali Saad-Eldin, NeuroData lab

#### University of Cambridge / MRC Laboratory of Molecular Biology 
Michael Winding, Albert Cardona, Marta Zlatic, Chris Barnes

#### Microsoft Research 
Hayden Helm, Dax Pryce, Nick Caurvina, Bryan Tower, Patrick Bourke, Jonathan McLean, Carolyn Buractaon, Amber Hoak

---
# Questions?

![bg opacity:.65 95%](./../../../results/figs/plot_side_layouts/2_network_layout.png)

---
<div class="columns">
<div>



</div>
<div>



</div>
</div>