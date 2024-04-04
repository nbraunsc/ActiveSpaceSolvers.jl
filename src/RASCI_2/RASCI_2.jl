module RASCI_2
using ActiveSpaceSolvers
using LinearMaps
using InCoreIntegrals 
using Printf

# includes
include("interface.jl");
include("type_SubspaceDeterminantString.jl");
include("type_RASVector.jl");
include("TDMs.jl");

# import stuff so we can extend and export
import LinearMaps: LinearMap

#abstract type HP_Category end

# Exports
export RASCIAnsatz_2
export RASVector
export LinearMap 

end
