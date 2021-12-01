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
        margin-top: 0 auto;
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

<!-- ---

# What is a connectome? (for *this* talk)

A **connectome** is a network model of brain structure consisting of <span style="color: black"> nodes which 
represent individual neurons </span> and <span style="color: black"> edges which represent the presence of a synaptic 
connection </span> between those neurons. -->

---
# Many connectomics questions require comparison
For instance,
- Understand connectomes across evolution [1]
- Understand connectomes across development [2]
- Understand links between genetics and connectivity [3] 


<p></p>
<p></p>
<p></p>

<!-- > "Understanding statistical regularities and learning which variations are stochastic and which are secondary to an animal’s life history will help define the substrate upon which individuality rests and *require comparisons between circuit maps within and between animals.*" [1] (emphasis added) -->

<footer>

[1] Bartsotti + Correia et al. *Curr. Op. Neurobiology* (2021)
[2] Witvliet et al. *Nature* (2021)
[3] Valdes-Aleman et al. *Neuron* (2021)

</footer>

--- 

# Larval _Drosophila_ brain connectome
See [Michael Windings's talk](https://conference.neuromatch.io/abstract?edition=2021-4&submission_id=recVeh4RZFFRAQnIo), 11 AM (EST) Dec 2nd
- ~3000 neurons, ~544K synapses
- Both hemispheres of the brain reconstructed

![bg right:60% 95%](./results/figs/../../../results/figs/plot_layouts/whole-network-layout.png)
<!-- TODO say something about how we are treating as directed, unweighted, loopless -->

<footer>
Winding et al. “The complete connectome of an insect brain.” In prep (2021)
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

- Want a two-network-sample test!
- For simplicity (for now), consider networks to be *directed*, *unweighted*.
- For simplicity (for now), consider the <span style='color: var(--left)'> left $\rightarrow$ left </span> and <span style='color: var(--right)'> right $\rightarrow$ right </span> (ipsilateral) connections only.
- <span style='color: var(--left)'> $A^{(L)} \sim F^{(L)}$</span>, <span style='color: var(--right)'> $A^{(R)} \sim F^{(R)}$ </span>
- $H_0: \color{#66c2a5} F^{(L)} \color{black} = \color{#fc8d62}F^{(R)}$  
  $H_A: \color{#66c2a5} F^{(L)} \color{black} \neq  \color{#fc8d62} F^{(R)}$


<p class="break"></p>

<!-- TODO should this be simple diagram networks instead? -->
![center w:1000](./results/figs/../../../results/figs/plot_side_layouts/2_network_layout.png)

</div>


---
<!-- <style scoped>section { justify-content: start; }</style> -->
# Density-based testing: Erdos-Renyi (ER) model
<div class="twocols">

- Connections independent, same connection probability $p$ for all edges
-  $A_{ij} \sim Bernoulli(p)$
- Compare probabilities:
  $H_0: \color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$  
  $H_A: \color{#66c2a5} p^{(L)} \color{black} \neq  \color{#fc8d62} p^{(R)}$
<!-- TODO fix this centering -->
- **p-value $< 10^{-23}$**
- Is this a difference we care about?

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

# Group-based testing: stochastic block model (SBM)

<div class="twocols">

- Connections independent, with probability set by the <span style="color: var(--source)"> source node's group </span> and <span style="color: var(--target)"> target node's group </span>
- $A_{ij} \sim Bernoulli(B_{\color{#8da0cb}\tau_i, \color{#e78ac3}\tau_j})$
- Compare group-to-group connection probabilities:
  $H_0: \color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$  
  $H_A: \color{#66c2a5} B^{(L)} \color{black} \neq  \color{#fc8d62} B^{(R)}$
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

- Connections independent, probability from dot product of <span style='color: var(--source)'> source node's latent vector</span>, <span style="color: var(--target)"> target node's latent vector</span>.
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
Athreya et al. JMLR (2017),
Tang et al. Bernoulli (2017)
</footer>



---
# Can we detect differences when we know they exist?

<div class="twocols">

- Make two copies of right hemisphere network
- Apply some perturbation to one of them: 
  - Ex: Shuffle edges incident to some number of nodes in some group
- Rerun the RDPG-based test for symmetry

<p class="break"></p>

![110%](../results/figs/perturbations_rdpg/perturbation_pvalues_rdpg_normalize=True.png)

</div>

---
# To sum up...

<style scoped>
table {
    font-size: 27px;
    margin-bottom: 50px;
}
</style>

| Model | $H_0$ (vs. $H_A \neq$)                                             |    p-value    | Interpretation                                     |
| ----- | ------------------------------------------------------------------ | :-----------: | -------------------------------------------------- |
| ER    | $\color{#66c2a5} p^{(L)} \color{black} = \color{#fc8d62}p^{(R)}$   |  $<10^{-23}$  | Reject densities the same                          |
| SBM   | $\color{#66c2a5} B^{(L)} \color{black} = \color{#fc8d62} B^{(R)}$  |  $< 10^{-4}$  | Reject cell type connection probabilities the same |
| SBM   | $\color{#66c2a5}B^{(L)} \color{black}  = c \color{#fc8d62}B^{(R)}$ | $\approx 0.7$ | Don't reject the above after density adjustment    |
| RDPG  | $\color{#66c2a5} F^{(L)} \color{black} = \color{#fc8d62} F^{(R)}$  |  $\approx 1$  | Don't reject latent distributions the same         |

**The answer to this very simple question totally depends on how you frame it!**

- Tests are sensitive to some alternatives and not others
- Difference you might not care about (e.g. density) need to be explicitly accounted for
  
---

# Future work
- Many other tests (e.g. compare subgraph counts)
- Studying the sets of alternatives each test is/is not sensitive to
- Roadmap for future principled comparisons of connectome networks!

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
Joshua Vogelstein, Carey Priebe, Mike Powell, Eric Bridgeford, Kareef Ullah, Diane Lee, Sambit Panda, Jaewon Chung, Ali Saad-Eldin
#### University of Cambridge / Laboratory of Molecular Biology 
Michael Winding, Albert Cardona, Marta Zlatic, Chris Barnes
#### Microsoft Research 
Hayden Helm, Dax Pryce, Nick Caurvina, Bryan Tower, Patrick Bourke, Jonathan McLean, Carolyn Buractaon, Amber Hoak
#### NMC organizers!

---
# Questions?

![bg opacity:.7 95%](../results/figs/plot_side_layouts/2_network_layout.png)