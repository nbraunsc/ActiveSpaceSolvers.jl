using LinearAlgebra
using Printf
using NPZ
using StaticArrays
using JLD2
using BenchmarkTools
using LinearMaps
using TensorOperations

using QCBase
using InCoreIntegrals 
using BlockDavidson


"""
    compute_operator_c_a(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}


Compute representation of a operator between states `bra_v` and `ket_v` for alpha
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_c_a(bra::Solution{RASCIAnsatz_2,T}, 
                              ket::Solution{RASCIAnsatz_2,T}) where {T}

    bra.ansatz.na-1 == ket.ansatz.na     || throw(DimensionMismatch)#={{{=#
    bra.ansatz.nb == ket.ansatz.nb     || throw(DimensionMismatch) 
    bra_M = size(bra,2)
    ket_M = size(ket,2)
    #   TDM[p,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no)
    
    excitation_list = make_excitation_classes_c(ket.ansatz.ras_spaces)
    bra_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(bra.vectors, bra.ansatz)
    ket_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(ket.vectors, ket.ansatz)
    
    ras1, ras2, ras3 = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(ket.ansatz.ras_spaces)
    ras1_bra, ras2_bra, ras3_bra = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(bra.ansatz.ras_spaces)
    
    for (block1, vec) in ket_rasvec.data
        idx = 0
        det3 = SubspaceDeterminantString(prob.ras_spaces[3], block1.focka[3])
        for n in 1:det3.max
            det2 = SubspaceDeterminantString(prob.ras_spaces[2], block1.focka[2])
            for j in 1:det2.max
                det1 = SubspaceDeterminantString(prob.ras_spaces[1], block1.focka[1])
                for i in 1:det1.max
                    idx += 1
                    aconfig = [det1.config;det2.config.+det1.no;det3.config.+det1.no.+det2.no]
                    for (p_range, delta_e) in excitation_list
                        block2 = RasBlock(block1.focka.+delta_e, block1.fockb)
                        haskey(bra_rasvec.data, block2) || continue
                        for p in p_range
                            tmp = deepcopy(aconfig)
                            d1_c, d2_c, d3_c = breakup_config(tmp, ras1, ras2, ras3)
                            sgn_p, idx_new = apply_creation!(d1_c, d2_c, d3_c, ras1_bra, ras2_bra, ras3_bra, p)
                            sgn_p != 0 || continue

                            @views tdm_pqr = tdm[:,:,p]
                            @views v1_IJ = bra_rasvec.data[block2][idx_new, :, :]
                            @views v2_KL = ket_rasvec.data[block1][idx, :, :]
                            if sign_p == 1
                                @tensor begin
                                    tdm_pqr[s,t] += v1_IJ[K,s] * v2_KL[K,t]
                                end
                            else
                                @tensor begin 
                                    tdm_pqr[s,t] -= v1_IJ[K,s] * v2_KL[K,t]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    tdm = permutedims(tdm, [3,1,2])
    return tdm
end#=}}}=#

"""
    compute_operator_c_b(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a operator between states `bra_v` and `ket_v` for beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_c_b(bra::Solution{RASCIAnsatz_2,T}, 
                              ket::Solution{RASCIAnsatz_2,T}) where {T}

    bra.ansatz.na == ket.ansatz.na     || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb-1 == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)
    
    #   TDM[p,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no)
    
    excitation_list = make_excitation_classes_c(ket.ansatz.ras_spaces)
    bra_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(bra.vectors, bra.ansatz)
    ket_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(ket.vectors, ket.ansatz)
   
    ras1, ras2, ras3 = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(ket.ansatz.ras_spaces)
    ras1_bra, ras2_bra, ras3_bra = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(bra.ansatz.ras_spaces)
    
    sgnK = 1 
    if (ket.ansatz.na) % 2 != 0 
        sgnK = -sgnK
    end

    for (block1, vec) in ket_rasvec.data
        idx = 0
        det3 = SubspaceDeterminantString(prob.ras_spaces[3], block1.fockb[3])
        for n in 1:det3.max
            det2 = SubspaceDeterminantString(prob.ras_spaces[2], block1.fockb[2])
            for j in 1:det2.max
                det1 = SubspaceDeterminantString(prob.ras_spaces[1], block1.fockb[1])
                for i in 1:det1.max
                    idx += 1
                    aconfig = [det1.config;det2.config.+det1.no;det3.config.+det1.no.+det2.no]
                    for (p_range, delta_e) in excitation_list
                        block2 = RasBlock(block1.focka.+delta_e, block1.fockb)
                        haskey(bra_rasvec.data, block2) || continue
                        for p in p_range
                            tmp = deepcopy(aconfig)
                            d1_c, d2_c, d3_c = breakup_config(tmp, ras1, ras2, ras3)
                            sgn_p, idx_new = apply_creation!(d1_c, d2_c, d3_c, ras1_bra, ras2_bra, ras3_bra, p)
                            sgn_p != 0 || continue

                            @views tdm_pqr = tdm[:,:,p]
                            @views v1_IJ = bra_rasvec.data[block2][:,idx_new,:]
                            @views v2_KL = ket_rasvec.data[block1][:,idx,:]
                            if sign_p*sgnK == 1
                                @tensor begin
                                    tdm_pqr[s,t] += v1_IJ[K,s] * v2_KL[K,t]
                                end
                            else
                                @tensor begin 
                                    tdm_pqr[s,t] -= v1_IJ[K,s] * v2_KL[K,t]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    tdm = permutedims(tdm, [3,1,2])
    return tdm
end#=}}}=#



