# Outline 

# Preliminaries
- In [](define_data.ipynb) we explain what data we'll use for our comparisons. 

# Unmatched Tests
- In [](er_unmatched_test.ipynb), we run a test for whether the Erdos-Renyi model
fits between the two hemispheres are similar, finding that they are significantly different 
even under this simple model.
- In [](sbm_test.ipynb), we run a test for whether stochastic block model fits between the 
two hemispheres are similar. We also present a version for this test which accounts for
differences in edge density between the hemispheres. 
- In (COMING SOON) [](rdpg_test.ipynb), we run a test for whether random dot product graph model fits
between the two hemispheres are similar.
- In (COMING SOON), we run each of these tests over perturbed versions of the left and
right connectomes to analyze their power under a variety of alternatives to our null
hypothesis of symmetry.

# Matched Tests
- In [](er_matched_test.ipynb), we again test for bilateral symmetry under the 
  Erdos-Renyi model, this time using the matched/known-node-correspondence data. Again, 
  we find that the two hemispheres are significantly different under this model.
- In [](sbm_matched_test.ipynb), we perform a test for stochastick block model fit
  symmetry using the matched data. Again, we find that the stochastic block model fits
  are significantly different, but when adjusting to correct for a difference in
  density, this difference disappears.

# Appendix
- [](nhypergeom_sims.ipynb) describes a modified Fisher's exact test, supporting some of the work in 
[](sbm_test.ipynb).