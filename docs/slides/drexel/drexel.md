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
    img[alt~="icon"] {
        display: inline;
        margin: 0 0.125em;
        padding: 0;
        vertical-align: middle;
        height: 30px;
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
    .social_icons {
        padding: 0em;
        display: inline-block;
        vertical-align: top;
    }

---


<style scoped> 
h1 {
    font-size:40px;
}
p {
    font-size: 24px;
}
</style>

<!-- # Maggot brain, mirror image? A statistical analysis of bilateral symmetry in an insect brain connectome -->

<!-- # A statistical analysis of bilateral symmetry in an insect connectome -->

<!-- # Two-sample hypothesis testing for connectomes: 
## A case study on the bilateral symmetry of the *Drosophila* larva connectome -->

# Hypothesis testing for connectome comparisons: a statistical analysis of bilateral symmetry in an insect brain connectome

### Benjamin D. Pedigo

_Johns Hopkins University - Biomedical Engineering_
_[NeuroData lab](https://neurodata.io/)_

![icon](../../images/email.png) [_bpedigo@jhu.edu_](mailto:bpedigo@jhu.edu)
![icon](../../images/github.png) [_@bdpedigo (Github)_](https://github.com/bdpedigo)
![icon](../../images/twitter.png) [_@bpedigod (Twitter)_](https://twitter.com/bpedigod)
![icon](../../images/web.png) [https://bdpedigo.github.io/](https://bdpedigo.github.io/)

![bg right:45% w:600](./../../images/background.png)

--- 
# Outline
- What is electron microscopy connectomics 
- *Drosophila* larva brain connectome
- Why we should care about comparing connectomes
- Bilateral symmetry
- Extensions and other tools

<!-- What about weights? 
What about the partition?
What about matched pairs? -->
<!-- - Two sample testing
- Density test
- Group connection probability test
- Adjusting for a difference in density
- Removing -->
  
---

# Electron microscopy connectomics

![center w:1000](./../../images/FigureS1-reconstruction.png)

<footer>
Winding, Pedigo et al. “The complete connectome of an insect brain.” In prep. (2022)
</footer>


---

# _Drosophila_ larva (AKA a maggot) brain connectome


<div class="columns">
<div>

<!-- See [Michael Windings's talk](https://conference.neuromatch.io/abstract?edition=2021-4&submission_id=recVeh4RZFFRAQnIo) -->
- Collaboration with Marta Zlatic/Albert Cardona's groups - led by Michael Winding
- First whole-brain, single-cell connectome of any insect
- ~3000 neurons, ~550K synapses
- Both hemispheres of the brain reconstructed

</div>
<div>

![](./../../images/Figure1-brain-render.png)
<!-- ![w:600](./../../../results/figs/plot_layouts/whole-network-layout.png) -->

</div>
</div>

<footer>
Winding, Pedigo et al. “The complete connectome of an insect brain.” In prep. (2022)
</footer>

---
# We're just going to consider this to be a network

![center h:500](./../../../results/figs/show_data/Figure(1500x1500).png)

---
# Why bilateral symmetry?

> "We examined the connectivity of members of left–right homologous neuron pairs ... **to assess the amount of natural variability in connectivity.** ... **Differences between individual worms will be expected to be at least this large.**"

> "... the gustatory neuron ASEL (that is, the left neuron of the pair) has greater chemical connectivity than ASER (that is, the right neuron of the pair) to the olfactory neuron class AWC."



<footer>
Cook et al. Nature (2019)
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

<footer>Witvliet et al. Nature (2021)</footer>

---
# Connectomes across evolution, cortex
![center w:550](./../../images/bartsotti-fig2.png)

<footer>
Bartsotti + Correia et al. Curr. Op. Neurobiology (2021)

</footer>

---
# So, studying bilateral symmetry here lets us
- Try to formalize what we even mean by this property, and make claims about what we find in this connectome, and
- Test out methods for comparing networks for these future pursuits

---

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

- All edges are indepentent
- All edges generated with the same probability, $p$

![center](../../../results/figs/er_unmatched_test/er_explain.svg)

---
# Density-based testing

<!-- <div class="columns">
<div> -->

<!-- - $P[i \rightarrow j] = p$ -->
<!-- - Compare probabilities:
  $H_0: \color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$  
  $H_A: \color{#66c2a5} p^{(L)} \color{black} \neq  \color{#fc8d62} p^{(R)}$ -->

<!-- </div>
<div> -->

![center h:500](./../../../results/figs/er_unmatched_test/er_methods.svg)

<!-- </div>
</div> -->

---
# We detect a difference in density

<div class="columns">
<div>

![](../../../results/figs/er_unmatched_test/er_density.svg)

</div>
<div>

- p-value < $10^{-22}$


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
# We detect a difference in group-to-group connection probabilities

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

---
# Adjusting for a difference in density

![center](./../../../results/figs/adjusted_sbm_unmatched_test/adjusted_methods_explain.svg)

---
# Even with density adjustment, we detect a difference

<div class="columns">
<div>

![center](./../../../results/figs/adjusted_sbm_unmatched_test/resampled_pvalues_distribution.svg)

</div>
<div>

![center w:500](./../../../results/figs/adjusted_sbm_unmatched_test/sbm_pvalues_unlabeled.svg)

</div>
</div>

---
# So the Kenyon cells (KCs) are the only group where we detect remaining differences...

<div class="columns">
<div>

![center h:450](./../../../results/figs/kc_minus/kc_minus_methods.svg)


</div>
<div>

- ER test: $p <10^{-26}$
- SBM test: $p \approx 0.003$
- Adjusted SBM test: $p \approx 0.43$

</div>
</div>

---
# To sum up...

<style scoped>
table {
    font-size: 24px;
    margin-bottom: 50px;
}
</style>

| Model | $H_0$ (vs. $H_A \neq$)                                             |  KCs  |     p-value     | Interpretation                                           |
| ----- | ------------------------------------------------------------------ | :---: | :-------------: | -------------------------------------------------------- |
| ER    | $\color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$   |   +   |   $<10^{-23}$   | Reject densities the same                                |
| SBM   | $\color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$  |   +   |   $< 10^{-7}$   | Reject group connection probabilities the same           |
| aSBM  | $\color{#66c2a5}B^{(L)} \color{black}  = c \color{#fc8d62}B^{(R)}$ |   +   | $\approx 0.002$ | Reject above even after accounting for density           |
| ER    | $\color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$   |   -   |   $<10^{-26}$   | Reject densities the same (w/o KCs)                      |
| SBM   | $\color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$  |   -   | $\approx 0.003$ | Reject group connection probabilities the same (w/o KCs) |
| aSBM  | $\color{#66c2a5}B^{(L)} \color{black}  = c \color{#fc8d62}B^{(R)}$ |   -   | $\approx 0.43$  | Don't reject above after density adjustment (w/o KCs)    |

---

<style scoped>
section {
    justify-content: center;
    text-align: center;
}
</style>

# Extensions (and other tools)

---

<style scoped>
section {
    justify-content: center;
    text-align: center;
}
</style>

# But you threw out all of the edge weights!


---
# Thresholding at higher synapse counts reduces asymmetry

![center h:450](./../../../results/figs/thresholding_tests/integer_threshold_pvalues.svg)

---

<style scoped>
section {
    justify-content: center;
    text-align: center;
}
</style>

# What do we consider to be a "cell type"?

<!-- ---
# Something about methods for this stuff -->

---
# Hierarchical clustering of neurons based on observed connectivity

![center](../../images/bar-dendrogram-wide.svg)

![center w:800](../../images/cell-type-labels-legend.png)

<footer> Winding, Pedigo et al. “The complete connectome of an insect brain.” In prep. (2022) </footer>

<!-- TODO something to add some punch here - why care about the clusters -->

---
<style scoped>
section {
    justify-content: center;
    text-align: center;
}
</style>

# Are nodes/edges matched across hemispheres?

<!-- TODO something to show the idea of neuron pairs between hemispheres here before going on -->

---
# Bilateral neuron pairs 
![center](./../../images/mbon-expression.jpg)

<footer>Eschbach et al. eLife (2021)</footer>

---
# Graph matching
![center h:500](./../../images/network-matching-explanation.svg)

---
# Graph matching predicts single-neuron pairs between hemispheres
- ~86% of predicted pairs are confirmed by a human annotator

![center h:375](./../../../results/figs/match_examine/matched_adjacencies.png)

<footer> Winding, Pedigo et al. “The complete connectome of an insect brain.” In prep. (2022) </footer>

---
# Predicted pairs are morphologically similar

![center h:490](./match_examine/../../../../results/figs/match_examine/left-pair-predictions.svg)

<footer> Winding, Pedigo et al. “The complete connectome of an insect brain.” In prep. (2022) </footer>

---
# In summary...
- Studied statistical ways of framing "bilateral symmetry", proposing a test procedure for each
- All tests found the left and the right hemispheres significantly different, unless ignoring Kenyon cells and adjust for the difference in density
   <!-- - If there's a statistic (e.g. density) that you *don't* want in your definition of "different," it should be explicitly accounted for  -->
- Provided a foundation for future principled comparisons of connectomes
- Mentioned several other tools/analyses which could alter the definition of symmetry
   - Edge weights
   - Inferring neuron groups
   - Graph matching to find pairs

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
![center w:400](./../../images/jb_example.png)
[![h:50](https://jupyterbook.org/badge.svg)](http://docs.neurodata.io/bilateral-connectome/)


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
**Mike Powell**, **Eric Bridgeford**, **Carey Priebe**, **Joshua Vogelstein**, Kareef Ullah, Diane Lee, Sambit Panda, Jaewon Chung, Ali Saad-Eldin, NeuroData lab

#### University of Cambridge / MRC Laboratory of Molecular Biology 
**Michael Winding**, Albert Cardona, Marta Zlatic, Chris Barnes

#### Funding
![h:125](../../images/NSF_4-Color_bitmap_Logo.png)


<!-- #### Microsoft Research 
Hayden Helm, Dax Pryce, Nick Caurvina, Bryan Tower, Patrick Bourke, Jonathan McLean, Carolyn Buractaon, Amber Hoak -->

---
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
![icon](../../images/github.png) [_@bdpedigo (Github)_](https://github.com/bdpedigo)
![icon](../../images/twitter.png) [_@bpedigod (Twitter)_](https://twitter.com/bpedigod)
![icon](../../images/web.png) [https://bdpedigo.github.io/](https://bdpedigo.github.io/)

---

<style scoped>
section {
    justify-content: center;
    text-align: center;
}
</style>

# Appendix 

---
# There are so many other models!

Latent distribution test (random dot product graph)

<div class="columns">
<div>

![center w:500](../../../results/figs/rdpg_unmatched_test/screeplot.svg)

</div>
<div>

![center w:500](./../../../results/figs/rdpg_unmatched_test/pvalue_dimension_matrix.svg)

</div>
</div>

---
![center](../../../results/figs/sbm_unmatched_test/group_counts.svg)



---
# Combining p-values: nobody's perfect

![center h:475](./../../images/heard-fig1.png)

<footer>

Heard, Rubin-Delanchy *Biometrika* (2018)

</footer>

---
# Combining p-values: don't trust SciPy until 1.9.0

![center w:600](../../images/tippett-bug.png)

---

# Distribution under the null for combining p-values

![bg right h:600](./../../../results/figs/revamp_sbm_methods_sim/null_distributions.svg)

---

# Combining p-values: be careful with discreetness

![center w:800](./../../images/lancaster-title.png) 

<div class="columns">
<div>

![](./../../../overleaf/figs/plot_individual_pvalues/pvalue-dist-example2.svg)


</div>
<div>

$\leftarrow$ We are trying to approximate this null distribution with something continuous $Uniform(0,1)$

</div>
</div>


---
# Power for combining p-values

- We perturb:
  - Some # of them (x-axis)
  - By some amount (panels)

![bg center right:60% w:700](./../../../results/figs/revamp_sbm_methods_sim/perturbation_pvalues_lineplots.svg)

---

# Relative power (Fisher's vs Tippett's)

![center](./../../../results/figs/revamp_sbm_methods_sim/relative_power.svg)

---
# Plotting connection probabilities 

![center h:500](./../../../results/figs/sbm_unmatched_test/probs_scatter.svg)

<!-- ---
<div class="columns">
<div>



</div>
<div>



</div>
</div> -->