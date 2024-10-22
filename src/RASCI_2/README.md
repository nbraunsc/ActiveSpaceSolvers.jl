# ActiveSpaceSolvers

This package currently provides the following types:

### Types
- `RasBlock`
- `RASVector`
- `RASCIAnsatz_2`
- `SubspaceDeterminantString`

`RasBlock` is the type that blocks the `RASVector` and holds the information
for number of electrons in (ras1, ras2, and ras3) for alpha and beta.

`RASVector` is initalized with a guess vector of the RASCI dimension defined 
by the `RASCIAnsatz_2` parameters. This is an ordered dictonary with keys being
the `RasBlocks` and entries are the ci coeffs in an array of alpha x beta x roots.

`RASCIAnsatz_2` is defined by number of orbs, nalpha, nbeta, number of orbs in each ras space,
maximum number of holes, and maximum number of particles.
This ansatz is used to initalize the `RASVector`, then vector is what is passed to 
sigma build methods.

`SubspaceDeterminantString` is very similar to the `DeterminantString` type of `FCI` module.
This creates a configuation in a single ras subspace.

### To do
- optimize lookup tables, these are very slow once the number of orbitals gets large
- optimize sigma3 build, possibly chance to looping over occupied/virtual orbs instead of all orbs
