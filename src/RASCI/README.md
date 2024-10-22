# ActiveSpaceSolvers

This package currently provides the following types:

### Types
- `RASCIAnsatz`
- `HP_Category`
- `RASCI_OlsenGraph`
- `SpinPairs`

`RASCIAnsatz` is defined by number of orbs, nalpha, nbeta, number of orbs in each ras space,
maximum number of holes, and maximum number of particles.

`HP_Category` and `SpinPairs` are used to help define the Hilbert space and the configurations. No RASVector blocking is used here.

`RASCI_OlsenGraph` using the graphical approach in the reverse lexicial ordering to find the configurations in the RASCI Hilbert space.  This is very slow because it uses a lot of dictonaries to do the depth-first search.


### To do
- nothing really since this is the slowest, unoptimized code, just use to check optimization of RASCI_2 module
