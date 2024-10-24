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


### 
# Big question about apply_creation
# will there ever be a time where bra and ket have different ras spaces since apply_creation!(det1, det2, det3
# ras1, ras2, ras3, p) using the ras space from bra for the index but not sure if i need to also give ras 
# spaces from ket to find orbital p???
###


"""
    compute_operator_c_a(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}


Compute representation of a operator between states `bra_v` and `ket_v` for alpha
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_c_a(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}

    bra.ansatz.na-1 == ket.ansatz.na     || throw(DimzoensionMismatch)#={{{=#
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
        det3 = SubspaceDeterminantString(ket.ansatz.ras_spaces[3], block1.focka[3])
        for n in 1:det3.max
            det2 = SubspaceDeterminantString(ket.ansatz.ras_spaces[2], block1.focka[2])
            for j in 1:det2.max
                det1 = SubspaceDeterminantString(ket.ansatz.ras_spaces[1], block1.focka[1])
                for i in 1:det1.max
                    idx += 1
                    aconfig = [det1.config;det2.config.+det1.no;det3.config.+det1.no.+det2.no]
                    for (p_range, delta_e) in excitation_list
                        block2 = RasBlock(block1.focka.+delta_e, block1.fockb)
                        haskey(bra_rasvec.data, block2) || continue
                        for p in p_range
                            tmp = deepcopy(aconfig)
                            if p in tmp
                                continue
                            end

                            d1_c, d2_c, d3_c = breakup_config(tmp, ras1, ras2, ras3)
                            sgn_p, idx_new = apply_creation!(d1_c, d2_c, d3_c, ras1_bra, ras2_bra, ras3_bra, p)
                            sgn_p != 0 || continue

                            @views tdm_pqr = tdm[:,:,p]
                            @views v1_IJ = bra_rasvec.data[block2][idx_new, :, :]
                            @views v2_KL = ket_rasvec.data[block1][idx, :, :]
                            if sgn_p == 1
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
                    incr!(det1)
                end
                incr!(det2)
            end
            incr!(det3)
        end
    end
    tdm = permutedims(tdm, [3,1,2])
    return tdm
end#=}}}=#

"""
    compute_operator_c_b(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}

Compute representation of a operator between states `bra_v` and `ket_v` for beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_c_b(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}
#Sign is wrong for beta
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
        det3 = SubspaceDeterminantString(ket.ansatz.ras_spaces[3], block1.fockb[3])
        for n in 1:det3.max
            det2 = SubspaceDeterminantString(ket.ansatz.ras_spaces[2], block1.fockb[2])
            for j in 1:det2.max
                det1 = SubspaceDeterminantString(ket.ansatz.ras_spaces[1], block1.fockb[1])
                for i in 1:det1.max
                    idx += 1
                    bconfig = [det1.config;det2.config.+det1.no;det3.config.+det1.no.+det2.no]
                    for (p_range, delta_e) in excitation_list
                        block2 = RasBlock(block1.focka, block1.fockb.+delta_e)
                        haskey(bra_rasvec.data, block2) || continue
                        for p in p_range
                            tmp = deepcopy(bconfig)
                            if p in tmp
                                continue
                            end
                            
                            d1_c, d2_c, d3_c = breakup_config(tmp, ras1, ras2, ras3)
                            sgn_p, idx_new = apply_creation!(d1_c, d2_c, d3_c, ras1_bra, ras2_bra, ras3_bra, p)
                            sgn_p != 0 || continue

                            @views tdm_pqr = tdm[:,:,p]
                            @views v1_IJ = bra_rasvec.data[block2][:,idx_new,:]
                            @views v2_KL = ket_rasvec.data[block1][:,idx,:]
                            if sgn_p*sgnK == 1
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
                    incr!(det1)
                end
                incr!(det2)
            end
            incr!(det3)
        end
    end
    tdm = permutedims(tdm, [3,1,2])
    return tdm
end#=}}}=#

"""
    compute_operator_ca_aa(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}

Compute representation of a'a operators between states `bra_v` and `ket_v` for alpha-alpha
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_ca_aa(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}
    bra.ansatz.na == ket.ansatz.na     || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    # <s|p'q'|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|a'a|KL> c(KL,t) =
    # c(IJ,s) c(KL,t) <J|<I|a'a|K>|L>
    # c(IJ,s) c(KL,t) <J|L><I|a'a|K>     
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)
    
    #   TDM[pq,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no)
    
    excitation_list = make_excitation_classes_ca(ket.ansatz.ras_spaces)
    bra_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(bra.vectors, bra.ansatz)
    ket_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(ket.vectors, ket.ansatz)

    ras1, ras2, ras3 = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(ket.ansatz.ras_spaces)
    ras1_bra, ras2_bra, ras3_bra = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(bra.ansatz.ras_spaces)

    for (block1, vec) in ket_rasvec.data
        idx = 0
        det3 = SubspaceDeterminantString(ket.ansatz.ras_spaces[3], block1.focka[3])
        for n in 1:det3.max
            det2 = SubspaceDeterminantString(ket.ansatz.ras_spaces[2], block1.focka[2])
            for j in 1:det2.max
                det1 = SubspaceDeterminantString(ket.ansatz.ras_spaces[1], block1.focka[1])
                for i in 1:det1.max
                    idx += 1
                    aconfig = [det1.config;det2.config.+det1.no;det3.config.+det1.no.+det2.no]
                    for ((p_range, q_range), delta_e) in excitation_list
                        block2 = RasBlock(block1.focka.+delta_e, block1.fockb)
                        haskey(bra_rasvec.data, block2) || continue
                        for q in q_range
                            tmp = deepcopy(aconfig)
                            if q in tmp
                                sgn_q, det_a = apply_annihilation(tmp, q)

                                for p in p_range
                                    tmp2 = deepcopy(det_a)
                                    if p in tmp2
                                        continue
                                    end

                                    d1_c, d2_c, d3_c = breakup_config(tmp2, ras1, ras2, ras3)
                                    sgn_p, idx_new = apply_creation!(d1_c, d2_c, d3_c, ras1_bra, ras2_bra, ras3_bra, p)
                                    sgn_p != 0 || continue

                                    @views tdm_pqr = tdm[:,:,p,q]
                                    @views v1_IJ = bra_rasvec.data[block2][idx_new, :, :]
                                    @views v2_KL = ket_rasvec.data[block1][idx, :, :]
                                    if sgn_p*sgn_q == 1
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
                    incr!(det1)
                end
                incr!(det2)
            end
            incr!(det3)
        end
    end
    tdm = permutedims(tdm, [3,4,1,2])
    return tdm
end#=}}}=#
    

"""
    compute_operator_ca_bb(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}

Compute representation of a'a operators between states `bra_v` and `ket_v` for beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_ca_bb(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}
    bra.ansatz.na == ket.ansatz.na     || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    # <s|p'q'|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|b'b|KL> c(KL,t) =
    # c(IJ,s) c(KL,t) <J|<I|b'b|K>|L>
    # c(IJ,s) c(KL,t) <J|<I|b'|K>b|L> (-1)^ket.ansatz.na
    # c(IJ,s) c(KL,t) <I|K><J|b'b|L>     
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)
    
    #   TDM[pq,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no)

    excitation_list = make_excitation_classes_ca(ket.ansatz.ras_spaces)
    bra_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(bra.vectors, bra.ansatz)
    ket_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(ket.vectors, ket.ansatz)

    ras1, ras2, ras3 = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(ket.ansatz.ras_spaces)
    ras1_bra, ras2_bra, ras3_bra = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(bra.ansatz.ras_spaces)

    for (block1, vec) in ket_rasvec.data
        idx = 0
        det3 = SubspaceDeterminantString(ket.ansatz.ras_spaces[3], block1.fockb[3])
        for n in 1:det3.max
            det2 = SubspaceDeterminantString(ket.ansatz.ras_spaces[2], block1.fockb[2])
            for j in 1:det2.max
                det1 = SubspaceDeterminantString(ket.ansatz.ras_spaces[1], block1.fockb[1])
                for i in 1:det1.max
                    idx += 1
                    bconfig = [det1.config;det2.config.+det1.no;det3.config.+det1.no.+det2.no]
                    for ((p_range, q_range), delta_e) in excitation_list
                        block2 = RasBlock(block1.focka, block1.fockb.+delta_e)
                        haskey(bra_rasvec.data, block2) || continue
                        for q in q_range
                            tmp = deepcopy(bconfig)
                            if q in tmp
                                sgn_q, det_a = apply_annihilation(tmp, q)

                                for p in p_range
                                    tmp2 = deepcopy(det_a)
                                    if p in tmp2
                                        continue
                                    end

                                    d1_c, d2_c, d3_c = breakup_config(tmp2, ras1, ras2, ras3)
                                    sgn_p, idx_new = apply_creation!(d1_c, d2_c, d3_c, ras1_bra, ras2_bra, ras3_bra, p)
                                    sgn_p != 0 || continue

                                    @views tdm_pqr = tdm[:,:,p,q]
                                    @views v1_IJ = bra_rasvec.data[block2][:,idx_new,:]
                                    @views v2_KL = ket_rasvec.data[block1][:,idx, :]
                                    if sgn_p*sgn_q == 1
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
                    incr!(det1)
                end
                incr!(det2)
            end
            incr!(det3)
        end
    end
    tdm = permutedims(tdm, [3,4,1,2])
    return tdm
end#=}}}=#
    
"""
    compute_operator_ca_ab(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}

Compute representation of a'a operators between states `bra_v` and `ket_v` for alpha-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_ca_ab(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}
    bra.ansatz.na-1 == ket.ansatz.na     || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb+1 == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    # <s|p'q'|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|a'b|KL> c(KL,t) =
    # c(IJ,s) c(KL,t) <J|<I|a'b|K>|L>
    # c(IJ,s) c(KL,t) <J|<I|a'|K>b|L> (-1)^ket.ansatz.na
    # c(IJ,s) c(KL,t) <I|a'|K><J|b|L> (-1)^ket.ansatz.na
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)

    #   TDM[pq,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no)
    
    sgnK = 1 
    if (ket.ansatz.na) % 2 != 0 
        sgnK = -sgnK
    end
    
    create_list = make_excitation_classes_c(ket.ansatz.ras_spaces)
    ann_list = make_excitation_classes_a(ket.ansatz.ras_spaces)
    bra_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(bra.vectors, bra.ansatz)
    ket_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(ket.vectors, ket.ansatz)

    ras1, ras2, ras3 = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(ket.ansatz.ras_spaces)
    ras1_bra, ras2_bra, ras3_bra = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(bra.ansatz.ras_spaces)

    for (block1, vec) in ket_rasvec.data
        #loop over alpha strings
        idxa = 0
        det3a = SubspaceDeterminantString(ket.ansatz.ras_spaces[3], block1.focka[3])
        for na in 1:det3a.max
            det2a = SubspaceDeterminantString(ket.ansatz.ras_spaces[2], block1.focka[2])
            for ja in 1:det2a.max
                det1a = SubspaceDeterminantString(ket.ansatz.ras_spaces[1], block1.focka[1])
                for ia in 1:det1a.max
                    idxa += 1
                    aconfig = [det1a.config;det2a.config.+det1a.no;det3a.config.+det1a.no.+det2a.no]

                    #now beta strings
                    idxb=0
                    det3b = SubspaceDeterminantString(ket.ansatz.ras_spaces[3], block1.fockb[3])
                    for n in 1:det3b.max
                        det2b = SubspaceDeterminantString(ket.ansatz.ras_spaces[2], block1.fockb[2])
                        for j in 1:det2b.max
                            det1b = SubspaceDeterminantString(ket.ansatz.ras_spaces[1], block1.fockb[1])
                            for i in 1:det1b.max
                                idxb += 1
                                bconfig = [det1b.config;det2b.config.+det1b.no;det3b.config.+det1b.no.+det2b.no]
                                for (p_range, delta_c) in create_list
                                    for (q_range, delta_a) in ann_list
                                        block2 = RasBlock(block1.focka.+delta_c, block1.fockb.+delta_a)
                                        haskey(bra_rasvec.data, block2) || continue
                                        for q in q_range
                                            tmp = deepcopy(bconfig)
                                            if q in tmp
                                                sgn_q, det_a = apply_annihilation(tmp, q)
                                                d1_a, d2_a, d3_a = breakup_config(det_a, ras1_bra, ras2_bra, ras3_bra)
                                                det1_q = SubspaceDeterminantString(length(ras1_bra), length(d1_a), d1_a)
                                                det2_q = SubspaceDeterminantString(length(ras2_bra), length(d2_a), d2_a.-length(ras1_bra))
                                                det3_q = SubspaceDeterminantString(length(ras3_bra), length(d3_a), d3_a.-length(ras1_bra).-length(ras2_bra))

                                                idxb_new = calc_full_ras_index(det1_q, det2_q, det3_q)

                                                for p in p_range
                                                    tmp2 = deepcopy(aconfig)
                                                    if p in tmp2
                                                        continue
                                                    end

                                                    d1_c, d2_c, d3_c = breakup_config(tmp2, ras1, ras2, ras3)
                                                    sgn_p, idxa_new = apply_creation!(d1_c, d2_c, d3_c, ras1_bra, ras2_bra, ras3_bra, p)
                                                    sgn_p != 0 || continue

                                                    @views tdm_pqr = tdm[:,:,p,q]
                                                    @views v1_IJ = bra_rasvec.data[block2][idxa_new,idxb_new,:]
                                                    @views v2_KL = ket_rasvec.data[block1][idxa,idxb, :]
                                                    if sgn_p*sgn_q*sgnK == 1
                                                        @tensor begin
                                                            tdm_pqr[s,t] += v1_IJ[s] * v2_KL[t]
                                                        end
                                                    else
                                                        @tensor begin 
                                                            tdm_pqr[s,t] -= v1_IJ[s] * v2_KL[t]
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                                incr!(det1b)
                            end
                            incr!(det2b)
                        end
                        incr!(det3b)
                    end
                    incr!(det1a)
                end
                incr!(det2a)
            end
            incr!(det3a)
        end
    end
    tdm = permutedims(tdm, [3,4,1,2])
    return tdm
end#=}}}=#

"""
    compute_operator_cc_bb(bra::solution{A,t}, ket::solution{A,t}) where {t}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_cc_bb(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}
    bra.ansatz.na == ket.ansatz.na     || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb-2 == ket.ansatz.nb     || throw(DimensionMismatch) 

    # <s|p'q'|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|b'b'|KL> c(KL,t) =
    # c(IJ,s) c(KL,t) <J|<I|b'b'|K>|L>
    # c(IJ,s) c(KL,t) <I|K><J|b'b'|L>
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)
    
    #   TDM[pq,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no)
    
    excitation_list = make_excitation_classes_cc(ket.ansatz.ras_spaces)
    bra_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(bra.vectors, bra.ansatz)
    ket_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(ket.vectors, ket.ansatz)

    ras1, ras2, ras3 = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(ket.ansatz.ras_spaces)
    ras1_bra, ras2_bra, ras3_bra = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(bra.ansatz.ras_spaces)
    
    for (block1, vec) in ket_rasvec.data
        idx = 0
        det3 = SubspaceDeterminantString(ket.ansatz.ras_spaces[3], block1.fockb[3])
        for n in 1:det3.max
            det2 = SubspaceDeterminantString(ket.ansatz.ras_spaces[2], block1.fockb[2])
            for j in 1:det2.max
                det1 = SubspaceDeterminantString(ket.ansatz.ras_spaces[1], block1.fockb[1])
                for i in 1:det1.max
                    idx += 1
                    bconfig = [det1.config;det2.config.+det1.no;det3.config.+det1.no.+det2.no]
                    for ((p_range, q_range), delta_e) in excitation_list
                        block2 = RasBlock(block1.focka, block1.fockb.+delta_e)
                        haskey(bra_rasvec.data, block2) || continue
                        for q in q_range
                            tmp = deepcopy(bconfig)
                            if q in tmp
                                continue
                            else
                                sgn_q, det_a = apply_creation(tmp, q)

                                for p in p_range
                                    tmp2 = deepcopy(det_a)
                                    if p in tmp2
                                        continue
                                    end

                                    d1_c, d2_c, d3_c = breakup_config(tmp2, ras1, ras2, ras3)
                                    sgn_p, idx_new = apply_creation!(d1_c, d2_c, d3_c, ras1_bra, ras2_bra, ras3_bra, p)
                                    sgn_p != 0 || continue

                                    @views tdm_pqr = tdm[:,:,p,q]
                                    @views v1_IJ = bra_rasvec.data[block2][:,idx_new,:]
                                    @views v2_KL = ket_rasvec.data[block1][:,idx, :]
                                    if sgn_p*sgn_q == 1
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
                    incr!(det1)
                end
                incr!(det2)
            end
            incr!(det3)
        end
    end
    tdm = permutedims(tdm, [3,4,1,2])
    return tdm
end#=}}}=#

"""
    compute_operator_cc_aa(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for alpha-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_cc_aa(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}
    bra.ansatz.na-2 == ket.ansatz.na     || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    # <s|p'q'|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|a'a'|KL> c(KL,t) =
    # c(IJ,s) c(KL,t) <J|<I|a'a'|K>|L>
    # c(IJ,s) c(KL,t) <J|L><I|a'a'|K>
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)

    #   TDM[pq,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no)
    
    excitation_list = make_excitation_classes_cc(ket.ansatz.ras_spaces)
    bra_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(bra.vectors, bra.ansatz)
    ket_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(ket.vectors, ket.ansatz)

    ras1, ras2, ras3 = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(ket.ansatz.ras_spaces)
    ras1_bra, ras2_bra, ras3_bra = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(bra.ansatz.ras_spaces)

    for (block1, vec) in ket_rasvec.data
        idx = 0
        det3 = SubspaceDeterminantString(ket.ansatz.ras_spaces[3], block1.focka[3])
        for n in 1:det3.max
            det2 = SubspaceDeterminantString(ket.ansatz.ras_spaces[2], block1.focka[2])
            for j in 1:det2.max
                det1 = SubspaceDeterminantString(ket.ansatz.ras_spaces[1], block1.focka[1])
                for i in 1:det1.max
                    idx += 1
                    aconfig = [det1.config;det2.config.+det1.no;det3.config.+det1.no.+det2.no]
                    for ((p_range, q_range), delta_e) in excitation_list
                        block2 = RasBlock(block1.focka.+delta_e, block1.fockb)
                        haskey(bra_rasvec.data, block2) || continue
                        for q in q_range
                            tmp = deepcopy(aconfig)
                            if q in tmp
                                continue
                            else
                                sgn_q, det_a = apply_creation(tmp, q)
                                sgn_q != 0 || continue

                                for p in p_range
                                    tmp2 = deepcopy(det_a)
                                    if p in tmp2
                                        continue
                                    end

                                    d1_c, d2_c, d3_c = breakup_config(tmp2, ras1, ras2, ras3)
                                    sgn_p, idx_new = apply_creation!(d1_c, d2_c, d3_c, ras1_bra, ras2_bra, ras3_bra, p)
                                    sgn_p != 0 || continue

                                    @views tdm_pqr = tdm[:,:,p,q]
                                    @views v1_IJ = bra_rasvec.data[block2][idx_new, :, :]
                                    @views v2_KL = ket_rasvec.data[block1][idx, :, :]
                                    if sgn_p*sgn_q == 1
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
                    incr!(det1)
                end
                incr!(det2)
            end
            incr!(det3)
        end
    end
    tdm = permutedims(tdm, [3,4,1,2])
    return tdm
end#=}}}=#
    
"""
    compute_operator_cc_ab(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for alpha-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_cc_ab(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}
    bra.ansatz.na-1 == ket.ansatz.na     || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb-1 == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    # <s|p'q'|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|a'b'|KL> c(KL,t) = 
    # c(IJ,s) c(KL,t) <J|<I|a'b'|K>|L>
    # c(IJ,s) c(KL,t) <J|<I|a'|K>b'|L> (-1)^ket.ansatz.na
    # c(IJ,s) c(KL,t) <I|a'|K><J|b'|L> (-1)^ket.ansatz.na
    # 
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)

    #   TDM[pq,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no)
    
    sgnK = 1 
    if (ket.ansatz.na) % 2 != 0 
        sgnK = -sgnK
    end
    
    create_list = make_excitation_classes_c(ket.ansatz.ras_spaces)
    bra_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(bra.vectors, bra.ansatz)
    ket_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(ket.vectors, ket.ansatz)

    ras1, ras2, ras3 = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(ket.ansatz.ras_spaces)
    ras1_bra, ras2_bra, ras3_bra = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(bra.ansatz.ras_spaces)

    for (block1, vec) in ket_rasvec.data
        #loop over alpha strings
        idxa = 0
        det3a = SubspaceDeterminantString(ket.ansatz.ras_spaces[3], block1.focka[3])
        for na in 1:det3a.max
            det2a = SubspaceDeterminantString(ket.ansatz.ras_spaces[2], block1.focka[2])
            for ja in 1:det2a.max
                det1a = SubspaceDeterminantString(ket.ansatz.ras_spaces[1], block1.focka[1])
                for ia in 1:det1a.max
                    idxa += 1
                    aconfig = [det1a.config;det2a.config.+det1a.no;det3a.config.+det1a.no.+det2a.no]

                    #now beta strings
                    idxb=0
                    det3b = SubspaceDeterminantString(ket.ansatz.ras_spaces[3], block1.fockb[3])
                    for n in 1:det3b.max
                        det2b = SubspaceDeterminantString(ket.ansatz.ras_spaces[2], block1.fockb[2])
                        for j in 1:det2b.max
                            det1b = SubspaceDeterminantString(ket.ansatz.ras_spaces[1], block1.fockb[1])
                            for i in 1:det1b.max
                                idxb += 1
                                bconfig = [det1b.config;det2b.config.+det1b.no;det3b.config.+det1b.no.+det2b.no]
                                for (p_range, delta_ca) in create_list
                                    for (q_range, delta_cb) in create_list
                                        block2 = RasBlock(block1.focka.+delta_ca, block1.fockb.+delta_cb)
                                        haskey(bra_rasvec.data, block2) || continue
                                        for q in q_range
                                            tmp = deepcopy(bconfig)
                                            if q in tmp
                                                continue
                                            else
                                                d1_cb, d2_cb, d3_cb = breakup_config(tmp, ras1, ras2, ras3)
                                                sgn_q, idxb_new = apply_creation!(d1_cb, d2_cb, d3_cb, ras1_bra, ras2_bra, ras3_bra, q)
                                                sgn_q != 0 || continue

                                                for p in p_range
                                                    tmp2 = deepcopy(aconfig)
                                                    if p in tmp2
                                                        continue
                                                    end

                                                    d1_c, d2_c, d3_c = breakup_config(tmp2, ras1, ras2, ras3)
                                                    sgn_p, idxa_new = apply_creation!(d1_c, d2_c, d3_c, ras1_bra, ras2_bra, ras3_bra, p)
                                                    sgn_p != 0 || continue

                                                    @views tdm_pqr = tdm[:,:,p,q]
                                                    @views v1_IJ = bra_rasvec.data[block2][idxa_new,idxb_new,:]
                                                    @views v2_KL = ket_rasvec.data[block1][idxa,idxb, :]
                                                    if sgn_p*sgn_q*sgnK == 1
                                                        @tensor begin
                                                            tdm_pqr[s,t] += v1_IJ[s] * v2_KL[t]
                                                        end
                                                    else
                                                        @tensor begin 
                                                            tdm_pqr[s,t] -= v1_IJ[s] * v2_KL[t]
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                                incr!(det1b)
                            end
                            incr!(det2b)
                        end
                        incr!(det3b)
                    end
                    incr!(det1a)
                end
                incr!(det2a)
            end
            incr!(det3a)
        end
    end
    tdm = permutedims(tdm, [3,4,1,2])
    return tdm
end#=}}}=#
    
"""
    compute_operator_cca_aaa(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-alpha-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_cca_aaa(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}
    bra.ansatz.na - 1 == ket.ansatz.na || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    # <s|p'q'r|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|a'a'a|KL> c(KL,t) = 
    # c(IJ,s) c(KL,t) <J|<I|a'a'a|K>|L>
    # c(IJ,s) c(KL,t) <J|L><I|a'a'a|K>   
    bra_M = size(bra,2)
    ket_M = size(ket,2)
    
    #   TDM[pqr,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no, bra.ansatz.no)
    
    excitation_list = make_excitation_classes_cca(ket.ansatz.ras_spaces)
    bra_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(bra.vectors, bra.ansatz)
    ket_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(ket.vectors, ket.ansatz)

    ras1, ras2, ras3 = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(ket.ansatz.ras_spaces)
    ras1_bra, ras2_bra, ras3_bra = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(bra.ansatz.ras_spaces)

    for (block1, vec) in ket_rasvec.data
        idx = 0
        det3 = SubspaceDeterminantString(ket.ansatz.ras_spaces[3], block1.focka[3])
        for n in 1:det3.max
            det2 = SubspaceDeterminantString(ket.ansatz.ras_spaces[2], block1.focka[2])
            for j in 1:det2.max
                det1 = SubspaceDeterminantString(ket.ansatz.ras_spaces[1], block1.focka[1])
                for i in 1:det1.max
                    idx += 1
                    aconfig = [det1.config;det2.config.+det1.no;det3.config.+det1.no.+det2.no]
                    for ((p_range, q_range, r_range), delta_e) in excitation_list
                        block2 = RasBlock(block1.focka.+delta_e, block1.fockb)
                        haskey(bra_rasvec.data, block2) || continue
                        for r in r_range
                            tmp = deepcopy(aconfig)
                            if r in tmp
                                sgn_r, det_r = apply_annihilation(tmp, r)
                                sgn_r != 0 || continue
                                for q in q_range
                                    tmp2 = deepcopy(det_r)
                                    if q in tmp2
                                        continue
                                    else
                                        sgn_q, det_a = apply_creation(tmp2, q)
                                        sgn_q !=0 || continue

                                        for p in p_range
                                            tmp3 = deepcopy(det_a)
                                            if p in tmp3
                                                continue
                                            end

                                            d1_c, d2_c, d3_c = breakup_config(tmp3, ras1, ras2, ras3)
                                            sgn_p, idx_new = apply_creation!(d1_c, d2_c, d3_c, ras1_bra, ras2_bra, ras3_bra, p)
                                            sgn_p != 0 || continue

                                            @views tdm_pqr = tdm[:,:,p,q,r]
                                            @views v1_IJ = bra_rasvec.data[block2][idx_new, :, :]
                                            @views v2_KL = ket_rasvec.data[block1][idx, :, :]
                                            if sgn_p*sgn_q*sgn_r == 1
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
                    incr!(det1)
                end
                incr!(det2)
            end
            incr!(det3)
        end
    end
    tdm = permutedims(tdm, [3,4,5,1,2])
    return tdm
end#=}}}=#
    
"""
    compute_operator_cca_bbb(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for beta-beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_cca_bbb(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}
    bra.ansatz.na  == ket.ansatz.na || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb - 1 == ket.ansatz.nb  || throw(DimensionMismatch) 

    # <s|p'q'r|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|b'b'b|KL> c(KL,t) = 
    # c(IJ,s) c(KL,t) <J|<I|b'b'b|K>|L>
    # c(IJ,s) c(KL,t) <I|K><J|b'b'b|L> (-1)^ket.ansatz.na  
    bra_M = size(bra,2)
    ket_M = size(ket,2)
    
    sgnK = 1 
    if (ket.ansatz.na) % 2 != 0 
        sgnK = -sgnK
    end

    #   TDM[pqr,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no, bra.ansatz.no)
    
    excitation_list = make_excitation_classes_cca(ket.ansatz.ras_spaces)
    bra_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(bra.vectors, bra.ansatz)
    ket_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(ket.vectors, ket.ansatz)

    ras1, ras2, ras3 = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(ket.ansatz.ras_spaces)
    ras1_bra, ras2_bra, ras3_bra = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(bra.ansatz.ras_spaces)
    
    for (block1, vec) in ket_rasvec.data
        idx = 0
        det3 = SubspaceDeterminantString(ket.ansatz.ras_spaces[3], block1.fockb[3])
        for n in 1:det3.max
            det2 = SubspaceDeterminantString(ket.ansatz.ras_spaces[2], block1.fockb[2])
            for j in 1:det2.max
                det1 = SubspaceDeterminantString(ket.ansatz.ras_spaces[1], block1.fockb[1])
                for i in 1:det1.max
                    idx += 1
                    bconfig = [det1.config;det2.config.+det1.no;det3.config.+det1.no.+det2.no]
                    for ((p_range, q_range, r_range), delta_e) in excitation_list
                        block2 = RasBlock(block1.focka, block1.fockb.+delta_e)
                        haskey(bra_rasvec.data, block2) || continue
                        for r in r_range
                            tmp = deepcopy(bconfig)
                            if r in tmp
                                sgn_r, det_r = apply_annihilation(tmp, r)
                                for q in q_range
                                    tmp2 = deepcopy(det_r)
                                    if q in tmp2
                                        continue
                                    else
                                        sgn_q, det_a = apply_creation(tmp2, q)
                                        sgn_q != 0 || continue

                                        for p in p_range
                                            tmp3 = deepcopy(det_a)
                                            if p in tmp3
                                                continue
                                            end

                                            d1_c, d2_c, d3_c = breakup_config(tmp3, ras1, ras2, ras3)
                                            sgn_p, idx_new = apply_creation!(d1_c, d2_c, d3_c, ras1_bra, ras2_bra, ras3_bra, p)
                                            sgn_p != 0 || continue

                                            @views tdm_pqr = tdm[:,:,p,q,r]
                                            @views v1_IJ = bra_rasvec.data[block2][:,idx_new,:]
                                            @views v2_KL = ket_rasvec.data[block1][:,idx, :]
                                            if sgn_p*sgn_q*sgn_r*sgnK == 1
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
                    incr!(det1)
                end
                incr!(det2)
            end
            incr!(det3)
        end
    end
    tdm = permutedims(tdm, [3,4,5,1,2])
    return tdm
end#=}}}=#
    
"""
    compute_operator_cca_aba(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-beta-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_cca_aba(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}
    bra.ansatz.na  == ket.ansatz.na || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb - 1 == ket.ansatz.nb  || throw(DimensionMismatch) 
    
    # <s|p'q'r|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|a'b'a|KL> c(KL,t) = 
    # c(IJ,s) c(KL,t) <J|<I|a'ab'|K>|L> (-1) 
    # c(IJ,s) c(KL,t) <I|a'a|K><J|b'|L> (-1) (-1)^ket.ansatz.na  
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)
    
    alpha_ca_ex = make_excitation_classes_ca(ket.ansatz.ras_spaces)
    beta_c_ex = make_excitation_classes_c(ket.ansatz.ras_spaces)
    bra_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(bra.vectors, bra.ansatz)
    ket_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(ket.vectors, ket.ansatz)

    ras1, ras2, ras3 = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(ket.ansatz.ras_spaces)
    ras1_bra, ras2_bra, ras3_bra = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(bra.ansatz.ras_spaces)

    #   TDM[pq,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no, bra.ansatz.no)
    
    sgnK = -1 #start from -1 to account fro a'b'a -> a'ab' 
    if (ket.ansatz.na) % 2 != 0 
        sgnK = -sgnK
    end

    for (block1, vec) in ket_rasvec.data
        #loop over alpha strings
        idxa = 0
        det3a = SubspaceDeterminantString(ket.ansatz.ras_spaces[3], block1.focka[3])
        for na in 1:det3a.max
            det2a = SubspaceDeterminantString(ket.ansatz.ras_spaces[2], block1.focka[2])
            for ja in 1:det2a.max
                det1a = SubspaceDeterminantString(ket.ansatz.ras_spaces[1], block1.focka[1])
                for ia in 1:det1a.max
                    idxa += 1
                    aconfig = [det1a.config;det2a.config.+det1a.no;det3a.config.+det1a.no.+det2a.no]

                    #now beta strings
                    idxb=0
                    det3b = SubspaceDeterminantString(ket.ansatz.ras_spaces[3], block1.fockb[3])
                    for n in 1:det3b.max
                        det2b = SubspaceDeterminantString(ket.ansatz.ras_spaces[2], block1.fockb[2])
                        for j in 1:det2b.max
                            det1b = SubspaceDeterminantString(ket.ansatz.ras_spaces[1], block1.fockb[1])
                            for i in 1:det1b.max
                                idxb += 1
                                bconfig = [det1b.config;det2b.config.+det1b.no;det3b.config.+det1b.no.+det2b.no]
                                for ((p_range, r_range), delta_ca) in alpha_ca_ex
                                    for (q_range, delta_b) in beta_c_ex
                                        block2 = RasBlock(block1.focka.+delta_ca, block1.fockb.+delta_b)
                                        haskey(bra_rasvec.data, block2) || continue
                                        for q in q_range
                                            tmp = deepcopy(bconfig)
                                            if q in tmp
                                                continue
                                            else
                                                d1_b, d2_b, d3_b = breakup_config(tmp, ras1, ras2, ras3)
                                                sgn_q, idxb_new = apply_creation!(d1_b, d2_b, d3_b, ras1_bra, ras2_bra, ras3_bra, q)
                                                sgn_q != 0 || continue
                                                
                                                for r in r_range
                                                    tmp2 = deepcopy(aconfig)
                                                    if r in tmp2
                                                        sgn_r, det_a = apply_annihilation(tmp2, r)
                                                        sgn_r != 0 || continue

                                                        for p in p_range
                                                            tmp3 = deepcopy(det_a)
                                                            if p in tmp3
                                                                continue
                                                            end

                                                            d1_c, d2_c, d3_c = breakup_config(tmp3, ras1, ras2, ras3)
                                                            sgn_p, idxa_new = apply_creation!(d1_c, d2_c, d3_c, ras1_bra, ras2_bra, ras3_bra, p)
                                                            sgn_p != 0 || continue

                                                            @views tdm_pqr = tdm[:,:,p,q,r]
                                                            @views v1_IJ = bra_rasvec.data[block2][idxa_new,idxb_new,:]
                                                            @views v2_KL = ket_rasvec.data[block1][idxa,idxb, :]
                                                            if sgn_p*sgn_q*sgn_r*sgnK == 1
                                                                @tensor begin
                                                                    tdm_pqr[s,t] += v1_IJ[s] * v2_KL[t]
                                                                end
                                                            else
                                                                @tensor begin 
                                                                    tdm_pqr[s,t] -= v1_IJ[s] * v2_KL[t]
                                                                end
                                                            end
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                                incr!(det1b)
                            end
                            incr!(det2b)
                        end
                        incr!(det3b)
                    end
                    incr!(det1a)
                end
                incr!(det2a)
            end
            incr!(det3a)
        end
    end
    tdm = permutedims(tdm, [3,4,5,1,2])
    return tdm
end#=}}}=#

"""
    compute_operator_cca_abb(bra::Solution{A,T}, ket::Solution{A,T}) where {T, A<:Union{RASCIAnsatz_2, DDCIAnsatz}}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function compute_operator_cca_abb(bra::Solution{A}, ket::Solution{A}) where {T,A<:Union{RASCIAnsatz_2, DDCIAnsatz}}
    bra.ansatz.na - 1 == ket.ansatz.na || throw(DimensionMismatch) #={{{=#
    bra.ansatz.nb == ket.ansatz.nb     || throw(DimensionMismatch) 
    
    # <s|p'q'r|t>
    # I and K are α strings
    # J and L are β strings
    # c(IJ,s) <IJ|a'b'b|KL> c(KL,t) = 
    # c(IJ,s) c(KL,t) <J|<I|a'b'b|K>|L> 
    # c(IJ,s) c(KL,t) <I|a'|K><J|b'b|L> 
    
    bra_M = size(bra,2)
    ket_M = size(ket,2)
    
    #   TDM[pqr,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no, bra.ansatz.no)
    
    alpha_c_ex = make_excitation_classes_c(ket.ansatz.ras_spaces)
    beta_ca_ex = make_excitation_classes_ca(ket.ansatz.ras_spaces)
    bra_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(bra.vectors, bra.ansatz)
    ket_rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(ket.vectors, ket.ansatz)

    ras1, ras2, ras3 = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(ket.ansatz.ras_spaces)
    ras1_bra, ras2_bra, ras3_bra = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(bra.ansatz.ras_spaces)

    #   TDM[pq,s,t] = 
    tdm = zeros(Float64, bra_M, ket_M,  bra.ansatz.no, bra.ansatz.no, bra.ansatz.no)
    
    for (block1, vec) in ket_rasvec.data
        #loop over alpha strings
        idxa = 0
        det3a = SubspaceDeterminantString(ket.ansatz.ras_spaces[3], block1.focka[3])
        for na in 1:det3a.max
            det2a = SubspaceDeterminantString(ket.ansatz.ras_spaces[2], block1.focka[2])
            for ja in 1:det2a.max
                det1a = SubspaceDeterminantString(ket.ansatz.ras_spaces[1], block1.focka[1])
                for ia in 1:det1a.max
                    idxa += 1
                    aconfig = [det1a.config;det2a.config.+det1a.no;det3a.config.+det1a.no.+det2a.no]

                    #now beta strings
                    idxb=0
                    det3b = SubspaceDeterminantString(ket.ansatz.ras_spaces[3], block1.fockb[3])
                    for n in 1:det3b.max
                        det2b = SubspaceDeterminantString(ket.ansatz.ras_spaces[2], block1.fockb[2])
                        for j in 1:det2b.max
                            det1b = SubspaceDeterminantString(ket.ansatz.ras_spaces[1], block1.fockb[1])
                            for i in 1:det1b.max
                                idxb += 1
                                bconfig = [det1b.config;det2b.config.+det1b.no;det3b.config.+det1b.no.+det2b.no]
                                for (p_range, delta_c) in alpha_c_ex
                                    for ((q_range,r_range), delta_b) in beta_ca_ex
                                        block2 = RasBlock(block1.focka.+delta_c, block1.fockb.+delta_b)
                                        haskey(bra_rasvec.data, block2) || continue
                                        for p in p_range
                                            tmp = deepcopy(aconfig)
                                            if p in tmp
                                                continue
                                            else
                                                d1_b, d2_b, d3_b = breakup_config(tmp, ras1, ras2, ras3)
                                                sgn_p, idxa_new = apply_creation!(d1_b, d2_b, d3_b, ras1_bra, ras2_bra, ras3_bra, p)
                                                sgn_p != 0 || continue
                                                
                                                for r in r_range
                                                    tmp2 = deepcopy(bconfig)
                                                    if r in tmp2
                                                        sgn_r, det_a = apply_annihilation(tmp2, r)
                                                        sgn_r != 0 || continue

                                                        for q in q_range
                                                            tmp3 = deepcopy(det_a)
                                                            if q in tmp3
                                                                continue
                                                            end

                                                            d1_c, d2_c, d3_c = breakup_config(tmp3, ras1, ras2, ras3)
                                                            sgn_q, idxb_new = apply_creation!(d1_c, d2_c, d3_c, ras1_bra, ras2_bra, ras3_bra, q)
                                                            sgn_q != 0 || continue

                                                            @views tdm_pqr = tdm[:,:,p,q,r]
                                                            @views v1_IJ = bra_rasvec.data[block2][idxa_new,idxb_new,:]
                                                            @views v2_KL = ket_rasvec.data[block1][idxa,idxb, :]
                                                            if sgn_p*sgn_q*sgn_r == 1
                                                                @tensor begin
                                                                    tdm_pqr[s,t] += v1_IJ[s] * v2_KL[t]
                                                                end
                                                            else
                                                                @tensor begin 
                                                                    tdm_pqr[s,t] -= v1_IJ[s] * v2_KL[t]
                                                                end
                                                            end
                                                        end
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                                incr!(det1b)
                            end
                            incr!(det2b)
                        end
                        incr!(det3b)
                    end
                    incr!(det1a)
                end
                incr!(det2a)
            end
            incr!(det3a)
        end
    end
    tdm = permutedims(tdm, [3,4,5,1,2])
    return tdm
end#=}}}=#
    
