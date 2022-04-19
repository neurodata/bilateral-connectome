---
marp: true
theme: slides
size: 16:9
paginate: false
---


![bg center blur:3px opacity:15%](./../../../results/figs/background/background.svg)


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

# Model-based comparison of connectomes: applications in a whole insect brain

## Benjamin D. Pedigo
_(he/him) - ![icon](../../images/email.png) [_bpedigo@jhu.edu_](mailto:bpedigo@jhu.edu) 
[NeuroData lab](https://neurodata.io/)_
_Johns Hopkins University - Biomedical Engineering_

#### Acknowledgements
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

--- 
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

## Are the <span style="color: var(--left)"> left </span> and <span style="color: var(--right)"> right </span> sides of this connectome <p> </p> *different*?

<!-- <footer>
Winding, Pedigo et al. “The complete connectome of an insect brain.” In prep. (2022)
</footer> -->

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

![center h:400](./../../../results/figs/sbm_unmatched_test/sbm_uncorrected_pvalues.svg)

<div class='center'>

p-value: ${<}10^{-7}$

</div>

</div>
</div>

---
# Modified definitions of symmetry which ARE exhibited

<div class="columns">
<div>

#### Rescal connection probabilities AND remove Kenyon cells

![center h:350](../../../results/figs/adjusted_sbm_unmatched_test/adjusted_methods_explain.svg)

<div class='center'>

p-value: ~$0.51$

</div>

</div>
<div>

#### Threshold by edge weight (input proportion to postsynaptic neuron)

![center h:400](./../../../results/figs/thresholding_tests/input_threshold_pvalues_legend.svg)

</div>
</div>

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

---
<!-- # Conclusions  -->

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
<div>

## More info
- [![icon h:80](./../../images/graspologic_svg.svg)](https://github.com/microsoft/graspologic) [![icon](https://pepy.tech/badge/graspologic)](https://pepy.tech/project/graspologic)
- This work: [![icon](https://jupyterbook.org/badge.svg)](http://docs.neurodata.io/bilateral-connectome/)[github.com/neurodata/bilateral-connectome](https://github.com/neurodata/bilateral-connectome)
- Chung et al. *Statistical connectomics* (2021)
- Data: Winding, Pedigo et al. *In preparation* (2022)

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

<!-- fold this into one of the previous -->

<!-- refs at the end in like quesions or something -->

<!-- cite stat conn and/or connectal coding -->

<!-- email terry -->
<!-- cover letter -->
<!-- email marta alberta -->
<!-- check with Michael about timing -->
<!-- we have a draft cover letter somewhere -->