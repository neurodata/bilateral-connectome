# Outline 

# Preliminaries
- In [](define_data.ipynb) we explain what data we'll use for our comparisons. 

# Tests
- In (COMING SOON) [](er_test.ipynb) we run a test for whether the Erdos-Renyi model
fits between the two hemispheres are similar, finding that they are significantly different 
even under this simple model.
- In [](sbm_test.ipynb) we run a test for whether stochastic block model fits between the 
two hemispheres are similar. We also present a version for this test which accounts for
differences in edge density between the hemispheres. 
- In (COMING SOON) [](rdpg_test.ipynb) we run a test for whether random dot product graph model fits
between the two hemispheres are similar.
- In (COMING SOON) we run each of these tests over perturbed versions of the left and
right connectomes to analyze their power under a variety of alternatives to our null
hypothesis of symmetry.

# Appendix
- [](nhypergeom_sims.ipynb) describes a modified Fisher's exact test, supporting some of the work in 
[](sbm_test.ipynb).