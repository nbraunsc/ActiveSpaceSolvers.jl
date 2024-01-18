module RASCI
using ActiveSpaceSolvers
using LinearMaps
using InCoreIntegrals 
using Printf

# includes
include("interface.jl");
include("type_SpinPairs.jl");
include("TDMs.jl");
include("rasci_inner.jl");

# import stuff so we can extend and export
import LinearMaps: LinearMap

#abstract type HP_Category end

# Exports
export RASCIAnsatz
export HP_Category
export Spin_Pair
export build_H_matrix 
export LinearMap 
export Spin_Categories


end
