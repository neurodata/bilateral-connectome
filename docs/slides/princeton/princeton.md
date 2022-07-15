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

---

# Many of the stated goals of connectomics rely on linking connectome to other domains...

![center h:500](./../../images/link_connectome.svg)

<!-- /Users/bpedigo/JHU_code/pcc/pcc/results/figs/diagram/link_connectome.png -->

<!-- ![](../../../../../pcc/pcc/results/figs/diagram/link_connectome.svg) -->


<!-- ![](Users/bpedigo/JHU_code/pcc/pcc/results/figs/diagram/link_connectome.png) -->

---

# Connectome $\leftrightarrow$ memory

> ...the acquisition of wiring diagrams across multiple individuals will yield insights into **how experiences shape neural connections.**

*Emphasis added*

<!-- _footer: Mind of a mouse, Abbott et al. 2020 -->

---
# Connectome $\leftrightarrow$ evolution

> Comparative connectomics of [...] **species across the phylogenetic tree** can infer the archetypal neural architecture of each bauplan and identify any circuits that possibly converged onto a shared and potentially optimal, structure.

*Emphasis added*

<!-- _footer: Neural architectures in the light of comparative connectomics, Barsotti + Correia et al. 2021-->

---
# Connectomes across development
![center h:475](./../../images/witvliet-fig1.png)

<footer>Witvliet et al. Nature (2021)</footer>

---

# But it is methodologically hard to compare connectomes! this is a test

--- 

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

# Outline for today

- Describe a dataset that I'll use for these examples throughout
- Show how connectome comparison can be framed as network hypothesis testing
- Show how we can use automated tools for predicting the correspondence of neurons
  across datasets
- Mention some extensions to use/combine/extend these tools that we're working on

<!-- ---


<style scoped>
section {
    justify-content: center;
    text-align: center;
}
</style>

# Data -->


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
# Remaining differences are isolated to KCs
![h:500](./../../../results/figs/adjusted_sbm_unmatched_test/sbm_pvalues_unlabeled.svg)

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

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

<style scoped>
section {
  justify-content: center;
  text-align: center;
}
</style>

# Matching neurons

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
# matched versions of our tests
- we think greater power, basically

---
# testing for a significant matching
- evaluate stereotypy at a single neuron level, basically

---
# the value of pairs - looking at models
- show the hierarchical clustering
- SBM cross validation curve 

--- 
![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

# Summary 

---

![bg center blur:3px opacity:20%](./../../../results/figs/background/background.svg)

# How to use these tools?
## graspologic
## bilateral repo 
## bgm 
## get in touch! 

--- 
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

#### Funding

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
