# ActiveSpaceSolvers

This package currently provides the following functions:

### Types
- `Ansatz`
- `Solution`
- `SolverSettings`

### Methods 
- `LinearMap`
- `build_H_matrix`
- `solve `
- `compute_1rdm`
- `compute_1rdm_2rdm`
- `compute_operator_ca_aa`
- `compute_operator_ca_bb`
- `compute_operator_ca_ab`
- `compute_operator_cc_aa`
- `compute_operator_cc_bb`
- `compute_operator_cc_ab`
- `compute_operator_cca_aaa`
- `compute_operator_cca_bbb`
- `compute_operator_cca_aba`
- `compute_operator_cca_abb`

### Modules
- `FCI`
- `RASCI`
- `RASCI_2`
- `DDCI`

RASCI was the first implementation of RASCI using OlsenGraphs.
This is slower and less optimized.
RASCI_2 was the second implementation of RASCI using blocking
in the RASVector. Each block corresponds to a specific Fock Space.
DDCI module is minimal code, but sets up a Difference-Dedicated CI
ansatz. DDCI just sets up the RASVector then uses sigma builds in the
RASCI_2 module.
