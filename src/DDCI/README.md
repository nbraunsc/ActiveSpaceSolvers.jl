# ActiveSpaceSolvers

This package currently provides the following types:

### Types
- `DDCIAnsatz`

`RASVector` is initalized with a guess vector of the RASCI dimension defined 
by the `DDCIAnsatz` parameters. This is an ordered dictonary with keys being
the `RasBlocks` and entries are the ci coeffs in an array of alpha x beta x roots. 
This is the same `RASVector` as in `RASCI_2` just initalized with the `DDCI` parameters.

`DDCIAnsatz` is defined by number of orbs, nalpha, nbeta, number of orbs in each ras space,
maximum number of holes, and maximum number of particles.
This ansatz is used to initalize the `RASVector`, then vector is what is passed to 
sigma build methods.

### To do
- optimize how the lookup tables are used in the `LinearMap` method, these are very slow once the number of orbitals gets large and right now for the code to work, sigma1 and sigma2 builds need access to the next full RAS excitation lookup tables. For example, a DDCI (2x) calculation 
