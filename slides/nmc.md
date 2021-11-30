---
marp: true
theme: test-theme
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
    }

# <style>
#   :root {
#     --color-background: #fff !important;
#     --color-foreground: #333 !important;
#     --color-highlight: #f96 !important;
#     --color-dimmed: #888 !important;
#   }
# </style>
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

![bg right:50% w:600](./docs/../../docs/images/temp-maggot-brain-umap-omni-hue_key=merge_class.png)

---

# What is a connectome? (for *this* talk)

A **connectome** is a network model of brain structure consisting of <span style="color: black"> nodes which 
represent individual neurons </span> and <span style="color: black"> edges which represent the presence of a synaptic 
connection </span> between those neurons.

---
# Many connectomics questions require comparison
- Understand wiring substrate of learning/memory
- Understand links between genetics or disease and connectivity
- Understand how different neural architectures lead to different computational abilities

<p></p>
<p></p>
<p></p>

> "Understanding statistical regularities and learning which variations are stochastic and which are secondary to an animal’s life history will help define the substrate upon which individuality rests and *require comparisons between circuit maps within and between animals.*" [1] (emphasis added)

<footer>
[1] Abbott et al. "The mind of a mouse." Cell (2020)
</footer>

--- 

# Larval _Drosophila_ brain connectome
- ~2500 brain neurons + ~500 sensory neurons
- ~544K synapses
- Both hemispheres of the brain reconstructed

See [Michael Windings's talk](https://conference.neuromatch.io/abstract?edition=2021-4&submission_id=recVeh4RZFFRAQnIo), 11 AM (EST) Dec 2nd

![bg right:60% 95%](./results/figs/../../../results/figs/plot_layouts/whole-network-layout.png)
<!-- TODO say something about how we are treating as directed, unweighted, loopless -->

<footer>
Winding et al. “The complete connectome of an insect brain.” In prep (2021)
</footer>

---

<style scoped>
section {
    justify-content: center;
}
</style>

# Are the <span style="color: var(--left)"> left </span> and the <span style="color: var(--right)"> right </span> sides of this brain *the same*?

<!-- TODO some subtext here? or leave as is -->

--- 

# Are these populations the same?

<div class="twocols">

- Known as two-sample testing
- $\color{#66c2a5} Y^{(1)} \sim F^{(1)}$, $\color{#fc8d62} Y^{(2)} \sim F^{(2)}$
- $H_0: \color{#66c2a5} F^{(1)} \color{black} = \color{#fc8d62} F^{(2)}$  
  $H_A: \color{#66c2a5} F^{(1)} \color{black} \neq \color{#fc8d62} F^{(2)}$

<p class="break"></p>

![center w:400](./results/../../results/figs/two_sample_testing/2_sample_real_line.svg)

</div>

<!-- ![bg right vertical w:500](./em) -->

--- 
# Are these two _networks_ the same?
<div class="twocols">

- Left side network: $\color{#66c2a5} A^{(L)} \sim F^{(L)}$ 
- Right side network: $\color{#fc8d62} A^{(R)} \sim F^{(R)}$

- $H_0: \color{#66c2a5} F^{(L)} \color{black} = \color{#fc8d62}F^{(R)}$  
  $H_A: \color{#66c2a5} F^{(L)} \color{black} \neq  \color{#fc8d62} F^{(R)}$

<p class="break"></p>

<!-- TODO should this be simple diagram networks instead? -->
![center w:1000](./results/figs/../../../results/figs/plot_side_layouts/2_network_layout.png)

</div>


---
<!-- <style scoped>section { justify-content: start; }</style> -->
# The simplest thing: Erdos-Renyi (ER) model
<div class="twocols">

- Connections independent, same connection probability $p$ for all edges
-  $A_{ij} \sim Bernoulli(p)$
- Compare probabilities:
  $H_0: \color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$  
  $H_A: \color{#66c2a5} p^{(L)} \color{black} \neq  \color{#fc8d62} p^{(R)}$
<!-- TODO fix this centering -->
- **p-value $< 10^{-23}$**


<p class="break"></p>

![center width:500px](../results/figs/er_unmatched_test/er_density.svg)

</div>

<!-- <style scoped>
section {
  padding-right: -100;
}
</style> -->
<!-- ![bg right:45% w:500](../results/figs/er_unmatched_test/er-density.png "This is a caption") -->



--- 

# Testing under the stochastic block model (SBM)

<div class="twocols">

- Connections independent, with probability set by the <span style="color: var(--source)"> source node's group </span> and <span style="color: var(--target)"> target node's group </span>
- $A_{ij} \sim Bernoulli(B_{\color{#8da0cb}\tau_i, \color{#e78ac3}\tau_j})$
- Compare group-to-group connection probabilities:
  $H_0: \color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$  
  $H_A: \color{#66c2a5} B^{(L)} \color{black} \neq  \color{#fc8d62} B^{(R)}$
  (Many binomial tests)
- **p-value $< 10^{-4}$** 

<p class="break"></p>

![w:600](results/figs/../../../results/figs/sbm_unmatched_test/sbm_uncorrected_pvalues.svg) 

</div>

--- 

# Adjusting for a difference in density

<div class="twocols">

- Rejecting $\color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$ can be explained by the difference in density?
- New null hypothesis:
  $H_0: \color{#66c2a5}B^{(L)} \color{black}  = c \color{#fc8d62}B^{(R)}$  
  where $c$ is a density-adjusting constant, $\frac{\color{#66c2a5} p^{(L)}}{\color{#fc8d62} p^{(R)}}$
- Randomly subsample edges from denser network, rerun test
- **p-values $> 0.6$**

<p class="break"></p>

![w:500](../results/figs/sbm_unmatched_test/pvalues_corrected.png)

</div>

---
# More flexibility: Random dot product graph (RDPG)

<div class="twocols">

- Connections independent, probability from dot product of <span style='color: var(--source)'> source node's latent vector </span>, <span style="color: var(--target)"> target node's latent vector </span>.
- $A_{ij} \sim Bernoulli(\langle \color{#8da0cb} x_i, \color{#e78ac3} y_j \color{black} \rangle)$
- $\color{#66c2a5} x_i^{(L)} \sim F^{(L)}$,  $\color{#fc8d62} x_i^{(R)} \sim F^{(R)}$
- Compare distributions of latent vectors:
  $H_0: \color{#66c2a5} F^{(L)} \color{black} = \color{#fc8d62} F^{(R)}$  
  $H_A: \color{#66c2a5} F^{(L)} \color{black} \neq \color{#fc8d62} F^{(R)}$
- **p-value** $\approx 1$

<p class="break"></p>

![h:500](../results/figs/rdpg_unmatched_test/latents_d=3.png)

</div>

<footer>
Athreya et al. "Statistical inference on random dot product graphs: a survey." JMLR (2017)
</footer>

---
# To sum up so far...

<style scoped>
table {
    font-size: 28px;
    margin-bottom: 50px;
}
</style>

| Model | $H_0$ (vs. $H_A \neq$)                                             |    p-value    | Interpretation |
| ----- | ------------------------------------------------------------------ | :-----------: | -------------- |
| ER    | $\color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$   |  $<10^{-23}$  | Reject densities the same
| SBM   | $\color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$  |  $< 10^{-4}$  | Reject cell type connection probabilities the same
| SBM   | $\color{#66c2a5}B^{(L)} \color{black}  = c \color{#fc8d62}B^{(R)}$ | $\approx 0.7$ | Don't reject the above after density adjustment
| RDPG  | $\color{#66c2a5} F^{(L)} \color{black} = \color{#fc8d62} F^{(R)}$  |  $\approx 1$  | Don't reject latent distributions the same

**The answer to this very simple question totally depends on how you frame it!**

---
# How sensitive are these tests?

<div class="twocols">

- Make 2 copies of one hemisphere network
- Apply some perturbation: 

- Rerun a test for symmetry

<p class="break"></p>



</div>

---
# Summary
- Many different ways to write "are the left and right the same" as a statistical hypothesis
   - Each yields a different test procedure
   - Each test is sensitive to varying alternatives
- May not care about some differences (e.g. density) and any test will need to adjust
- Techniques apply anytime one wants to compare connectomes/networks

# Future work
- Many other tests one could run (e.g. compare subgraph counts)
- Many other alternatives one could be interested in

--- 
<!-- # ```graspologic``` -->

<div class="twocols">

## graspologic:

[github.com/microsoft/graspologic](https://github.com/microsoft/graspologic)

![w:450](graspologic_svg.svg)

[![h:50](https://pepy.tech/badge/graspologic)](https://pepy.tech/project/graspologic)  [![h:50](https://img.shields.io/github/stars/microsoft/graspologic?style=social)](https://github.com/microsoft/graspologic)  [![h:50](https://img.shields.io/github/contributors/microsoft/graspologic)](https://github.com/microsoft/graspologic/graphs/contributors)  [![h:50](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
<p class="break"></p>



## This work:
[github.com/neurodata/bilateral-connectome](https://github.com/neurodata/bilateral-connectome) 
![w:600](jb_example.png)

</div>

<footer>Chung, Pedigo et al. "Graspy: graph statistics in Python" JMLR (2019)</footer>

---
# Acknowledgements

<style scoped> 
p {
    font-size: 24px
}
h4 {font-size: 30px}
</style>

#### Johns Hopkins University
Joshua Vogelstein, Carey Priebe, Mike Powell, Eric Bridgeford, Kareef Ullah, Diane Lee, Sambit Panda, Jaewon Chung, Ali Saad-Eldin
#### University of Cambridge / Laboratory of Molecular Biology 
Michael Winding, Albert Cardona, Marta Zlatic, Chris Barnes
#### Microsoft Research 
Hayden Helm, Dax Pryce, Nick Caurvina, Bryan Tower, Patrick Bourke, Jonathan McLean, Carolyn Buractaon, Amber Hoak

---
# Questions?

![bg opacity:.7 95%](../results/figs/plot_side_layouts/2_network_layout.png)