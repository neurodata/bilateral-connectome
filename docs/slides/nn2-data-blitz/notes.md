# slide 1
- Introduction
- Tell you today about comparison of 
- weve been looking at developing methods for understanding the signifigance and nature of differences between connectomes
- as a case study, i'll tell you about one application of these tools to an insect brain connectome

# slide 2
- the dataset we've been working on is the connectome of the larval drosophila brain, mapped by our collaborators at the university of cambridge
- has around 3,000 neurons in its brain, and crucially, they've reconstructed all neurons on both hemispheres
- this gave us the opportunity to try to understand whether this brain was bilaterally symmetriy
- we can treat each hemisphere of the brain as its own network, and then we're going to ask a very simple question: are left right different
- to do so, we'll need to formalize what we mean by different

# slide 3
- the simplest model we started with is the erdos-renyi model, which only cares about one parameter: the density
- the density, p, sets the global connection probability in a network, how likely are any two nodes to be connected on average
- we can fit to the left and right, and then we're left with two parameters, p left and p right which we want to compare 
- if you run that test, we find that while the densities are not drastically different, the difference is highly significant. 
- we were surprised that even under this simplest model which only cares about this one high level network summary they are different, we next sought to localize 

# slide 4 
- to do so we use something called the stochastic block model
- treats each neuron as belonging to a group, here we use putative cell types
- models the probability of connections between groups, shown here by this blue matrix, B
- again, we can fit to left and right, and now we need to test whether these
matrices are different
- if you run the test we developed for this comparison, we get a p-value for each group-to-group connection, shown in this heatmap
- see there are 5 connections which pop out as significant, and the p-value comparing these entire *matrices* is also highly significant. 
- so, we again reject bilateral symmetry under this definition.

# slide 5
- briefly, i'll also add that we've extended these comparisons in a few ways
- can adjust the SBM to account for the difference in densities between hemispheres, that can yield bilateral symmetry depending on the subgraph
- can also remove low weight edges under some notion of weight, and find that 
if you use the right one can also get to "symmetry" under each test we looked at.

# slide 6
- to sum up, testing hypotheses in connectomics requires specialized procedures adapted to connectomes
- we presented several two-sample testing tools which correspond to different notions of what it means for networks to be different
- 
