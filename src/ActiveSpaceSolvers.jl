module ActiveSpaceSolvers
using QCBase
using LinearMaps
using InCoreIntegrals 
using Printf
using BlockDavidson
using InteractiveUtils


# Interface Types
abstract type Ansatz end        
include("type_Solutions.jl");
include("type_SolverSettings.jl");
include("Helpers.jl");


# Interface Methods: extend each for a new `Ansatz`
function solve end     
function build_H_matrix end     
function build_S2_matrix end     
function apply_S2_matrix end     
function compute_1rdm end     
function compute_1rdm_2rdm end    
# operator functions
function compute_operator_c_a end     
function compute_operator_c_b end     
function compute_operator_ca_aa end     
function compute_operator_ca_bb end     
function compute_operator_ca_ab end     
function compute_operator_cc_aa end     
function compute_operator_cc_bb end     
function compute_operator_cc_ab end     
function compute_operator_cca_aaa end     
function compute_operator_cca_bbb end     
function compute_operator_cca_aba end     
function compute_operator_cca_abb end     
function compute_s2 end
function apply_sminus end
function apply_splus end
## analysis
function svd_state end     
# methods for getting info from Ansatze


# Exports from Interface Layer
# types
export Ansatz
export Solution 
export SolverSettings
# methods 
export LinearMap 
export build_H_matrix 
export build_S2_matrix 
export apply_S2_matrix 
export solve 
export compute_1rdm
export compute_1rdm_2rdm
export compute_operator_c_a
export compute_operator_c_b
export compute_operator_ca_aa      
export compute_operator_ca_bb      
export compute_operator_ca_ab      
export compute_operator_cc_aa      
export compute_operator_cc_bb      
export compute_operator_cc_ab      
export compute_operator_cca_aaa      
export compute_operator_cca_bbb      
export compute_operator_cca_aba      
export compute_operator_cca_abb      
export compute_s2
export svd_state
export apply_sminus
export apply_splus

export n_elec
export n_elec_a
export n_elec_b
export dim
#export compute_s2 

# include sub-modules and import/export Ansatz sub-types
include("FCI/FCI.jl");
import .FCI: FCIAnsatz
export FCIAnsatz

include("RASCI/RASCI.jl");
import .RASCI: RASCIAnsatz
export RASCIAnsatz

include("RASCI_2/RASCI_2.jl");
import .RASCI_2: RASCIAnsatz_2
export RASCIAnsatz_2


# some methods
n_elec(a::Ansatz) = a.na + a.nb 
n_elec_a(a::Ansatz) = a.na     
n_elec_b(a::Ansatz) = a.nb     
dim(a::Ansatz) = a.dim 

QCBase.n_orb(a::Ansatz) = a.no 
QCBase.n_orb(a::Solution) = n_orb(a.ansatz)
n_elec(a::Solution) = n_elec(a.ansatz)
n_elec_a(a::Solution) = n_elec_a(a.ansatz)
n_elec_b(a::Solution) = n_elec_b(a.ansatz)
dim(a::Solution) = dim(a.ansatz)

Base.size(a::Solution, i::Integer) = size(a.vectors,i)
end
