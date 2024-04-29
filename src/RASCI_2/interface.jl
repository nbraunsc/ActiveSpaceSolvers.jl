#using InCoreIntegrals
using ActiveSpaceSolvers
using QCBase
import LinearMaps
using OrderedCollections
using BlockDavidson
using StaticArrays
using LinearAlgebra
using Printf
using TimerOutputs

#Optional{T} = Union{T, Nothing}

struct RASCIAnsatz_2 <: Ansatz
    no::Int
    na::Int  # number of alpha
    nb::Int  # number of beta
    dim::Int
    ras_spaces::SVector{3, Int}   # Number of orbitals in each ras space (RAS1, RAS2, RAS3)
    max_h::Int8  #max number of holes in ras1 (GLOBAL, Slater Det)
    max_p::Int8 #max number of particles in ras3 (GLOBAL, Slater Det)
    max_h2::Int8 #use this for DDCI
    max_p2::Int8 #use this for DDCI
end

"""
    RASCIAnsatz_2(no, na, nb, ras_spaces::Any, max_h, max_p)
Constructor
# Arguments
- `no`: Number of spatial orbitals
- `na`: Number of α electrons
- `nb`: Number of β electrons
- `ras_spaces`: Number of orbitals in each (RAS1, RAS2, RAS3)
- `max_h`: Max number of holes in RAS1
- `max_p`: Max number of particles in RAS3
"""
function RASCIAnsatz_2(no::Int, na, nb, ras_spaces::Any; max_h=0, max_p=ras_spaces[3], max_h2=0, max_p2=0)
    na <= no || throw(DimensionMismatch)
    nb <= no || throw(DimensionMismatch)
    sum(ras_spaces) == no || throw(DimensionMismatch)
    ras_spaces = convert(SVector{3,Int},collect(ras_spaces))
    na = convert(Int, na)
    nb = convert(Int, nb)
    max_h = convert(Int8, max_h)
    max_p = convert(Int8, max_p)
    if max_h2 == 0 && max_p2 == 0
        max_h2=Int8(0)
        max_p2=Int8(0)
        rdim = calc_rdim(ras_spaces, na, nb, max_h, max_p)
    else
        max_h2 = convert(Int8, max_h2)
        max_p2 = convert(Int8, max_p2)
        rdim = calc_rdim_ddci(ras_spaces, na, nb, max_h, max_p, max_h2, max_p2)
    end

    return RASCIAnsatz_2(no, na, nb, rdim, ras_spaces, max_h, max_p, max_h2, max_p2);
end

function Base.display(p::RASCIAnsatz_2)
    @printf(" RASCI_2:: #Orbs = %-3i #α = %-2i #β = %-2i Fock Spaces: (%i, %i, %i) RASCI Dimension: %-3i MAX Holes: %i MAX Particles: %i MAX Holes(DDCI): %i MAX Particles(DDCI): %i \n",p.no,p.na,p.nb,p.ras_spaces[1], p.ras_spaces[2], p.ras_spaces[3], p.dim, p.max_h, p.max_p, p.max_h2, p.max_p2)
end

function Base.print(p::RASCIAnsatz_2)
    @printf(" RASCIAnsatz_2:: #Orbs = %-3i #α = %-2i #β = %-2i Fock Spaces: (%i, %i, %i) RASCI Dimension: %-3i MAX Holes: %i MAX Particles: %i\n",p.no,p.na,p.nb,p.ras_spaces[1], p.ras_spaces[2], p.ras_spaces[3], p.dim, p.max_h, p.max_p)
end

function DDCI(no::Int, na, nb, ras_spaces::Any; ddci="1x")
    na <= no || throw(DimensionMismatch)
    nb <= no || throw(DimensionMismatch)
    sum(ras_spaces) == no || throw(DimensionMismatch)
    ras_spaces = convert(SVector{3,Int},collect(ras_spaces))
    na = convert(Int, na)
    nb = convert(Int, nb)
    if ddci=="1x"
        h = 1
        p = 0
        h2 = 0
        p2 = 1
        max_h = convert(Int8, h)
        max_p = convert(Int8, p)
        max_h2 = convert(Int8, h2)
        max_p2 = convert(Int8, p2)
        rdim = calc_rdim_ddci(ras_spaces, na, nb, max_h, max_p, max_h2, max_p2)
        return RASCIAnsatz_2(no, na, nb, rdim, ras_spaces, max_h, max_p, max_h2, max_p2);

    elseif ddci=="2x"
        h = 2
        p = 1
        h2 = 1
        p2 = 2
        max_h = convert(Int8, h)
        max_p = convert(Int8, p)
        max_h2 = convert(Int8, h2)
        max_p2 = convert(Int8, p2)
        rdim = calc_rdim_ddci(ras_spaces, na, nb, max_h, max_p, max_h2, max_p2)
        return RASCIAnsatz_2(no, na, nb, rdim, ras_spaces, max_h, max_p, max_h2, max_p2);
    end
end



"""
    LinearMap(ints, prb::RASCIAnsatz_2)

Get LinearMap with takes a vector and returns action of H on that vector

# Arguments
- ints: `InCoreInts` object
- prb:  `RASCIAnsatz` object
"""
function LinearMaps.LinearMap(ints::InCoreInts, prob::RASCIAnsatz_2) where T
    iters = 0
    function mymatvec(v)
        rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(v, prob)
        lu = ActiveSpaceSolvers.RASCI_2.fill_lu(rasvec, prob.ras_spaces)

        iters += 1
        #@printf(" Iter: %4i", iters)
        print("Iter: ", iters, " ")
        #@printf(" %-50s", "Compute sigma 1: ")
        #flush(stdout)
        #display(size(v))
       
        nr = 0
        if length(size(v)) == 1
            nr = 1
        else 
            nr = size(v)[2]
        end
        
        sigma1 = ActiveSpaceSolvers.RASCI_2.sigma_one(rasvec, ints, prob.ras_spaces, lu)
        sigma2 = ActiveSpaceSolvers.RASCI_2.sigma_two(rasvec, ints, prob.ras_spaces, lu)
        sigma3 = ActiveSpaceSolvers.RASCI_2.sigma_three(rasvec, ints, prob.ras_spaces, lu)
        
        sig = sigma1 + sigma2 + sigma3
        sig .+= ints.h0*v
        return sig
    end
    return LinearMap(mymatvec, prob.dim, prob.dim, issymmetric=true, ismutating=false, ishermitian=true)
end

"""
    LinOpMat(ints, prb::RASCIAnsatz_2)

Get LinearMap with takes a vector and returns action of H on that vector

# Arguments
- ints: `InCoreInts` object
- prb:  `RASCIAnsatz_2` object
"""
function BlockDavidson.LinOpMat(ints::InCoreInts{T}, prob::RASCIAnsatz_2) where T

    iters = 0
    function mymatvec(v)
        rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(v, prob)
        lu = ActiveSpaceSolvers.RASCI_2.fill_lu(rasvec, prob.ras_spaces)

        iters += 1
        #@printf(" Iter: %4i", iters)
        #print("Iter: ", iters, " ")
        #flush(stdout)
        #display(size(v))
       
        nr = 0
        if length(size(v)) == 1
            nr = 1
        else 
            nr = size(v)[2]
        end
        
        sigma1 = ActiveSpaceSolvers.RASCI_2.sigma_one(rasvec, ints, prob.ras_spaces, lu)
        sigma2 = ActiveSpaceSolvers.RASCI_2.sigma_two(rasvec, ints, prob.ras_spaces, lu)
        sigma3 = ActiveSpaceSolvers.RASCI_2.sigma_three(rasvec, ints, prob.ras_spaces, lu)
        
        sig = sigma1 + sigma2 + sigma3
        
        sig .+= ints.h0*v
        return sig
    end
    return LinOpMat{T}(mymatvec, prob.dim, true)
end

function calc_rdim(ras_spaces::SVector{3, Int}, na::Int, nb::Int, max_h::Int8, max_p::Int8)
    a_blocks, fock_as = make_blocks(ras_spaces, na, max_h, max_p)#={{{=#
    b_blocks, fock_bs = make_blocks(ras_spaces, nb, max_h, max_p)
    
    start = 0
    for i in 1:length(a_blocks)
        dima = binomial(ras_spaces[1], fock_as[i][1])*binomial(ras_spaces[2], fock_as[i][2])*binomial(ras_spaces[3], fock_as[i][3])
        for j in 1:length(b_blocks)
            dimb = binomial(ras_spaces[1], fock_bs[j][1])*binomial(ras_spaces[2], fock_bs[j][2])*binomial(ras_spaces[3], fock_bs[j][3])
            if a_blocks[i][1]+b_blocks[j][1]<= max_h
                if a_blocks[i][2]+b_blocks[j][2] <= max_p
                    start += dima*dimb
                end
            end
        end
    end#=}}}=#
    return start
end

function calc_rdim_ddci(ras_spaces::SVector{3, Int}, na::Int, nb::Int, max_h::Int8, max_p::Int8, max_h2::Int8, max_p2::Int8)
    a_blocks, fock_as = make_blocks(ras_spaces, na, max_h, max_p)#={{{=#
    b_blocks, fock_bs = make_blocks(ras_spaces, nb, max_h, max_p)
    tmp = []
    
    start = 0
    for i in 1:length(a_blocks)
        dima = binomial(ras_spaces[1], fock_as[i][1])*binomial(ras_spaces[2], fock_as[i][2])*binomial(ras_spaces[3], fock_as[i][3])
        for j in 1:length(b_blocks)
            dimb = binomial(ras_spaces[1], fock_bs[j][1])*binomial(ras_spaces[2], fock_bs[j][2])*binomial(ras_spaces[3], fock_bs[j][3])
            if a_blocks[i][1]+b_blocks[j][1]<= max_h
                if a_blocks[i][2]+b_blocks[j][2] <= max_p
                    block1 = RasBlock(fock_as[i], fock_bs[j])
                    push!(tmp, block1)
                    start += dima*dimb
                end
            end
        end
    end
    
    a_blocks2, fock_as2 = make_blocks(ras_spaces, na, max_h2, max_p2)
    b_blocks2, fock_bs2 = make_blocks(ras_spaces, nb, max_h2, max_p2)
    for i in 1:length(a_blocks2)
        dima = binomial(ras_spaces[1], fock_as2[i][1])*binomial(ras_spaces[2], fock_as2[i][2])*binomial(ras_spaces[3], fock_as2[i][3])
        for j in 1:length(b_blocks2)
            dimb = binomial(ras_spaces[1], fock_bs2[j][1])*binomial(ras_spaces[2], fock_bs2[j][2])*binomial(ras_spaces[3], fock_bs2[j][3])
            if a_blocks2[i][1]+b_blocks2[j][1]<= max_h2
                if a_blocks2[i][2]+b_blocks2[j][2] <= max_p2
                    block1 = RasBlock(fock_as2[i], fock_bs2[j])
                    if block1 in tmp
                        continue
                    else
                        push!(tmp, block1)
                        start += dima*dimb
                    end
                end
            end
        end
    end#=}}}=#
    return start
end

"""
    ActiveSpaceSolvers.compute_s2(sol::Solution)

Compute the <S^2> expectation values for each state in `sol`
"""
function ActiveSpaceSolvers.compute_s2(sol::Solution{RASCIAnsatz_2,T}) where {T}
    return compute_S2_expval(sol.vectors, sol.ansatz)
end

"""
    build_S2_matrix(P::RASCIAnsatz_2)

Build the S2 matrix in the Slater Determinant Basis  specified by `P`
"""
function ActiveSpaceSolvers.apply_S2_matrix(P::RASCIAnsatz_2, v::AbstractArray{T}) where T
    return apply_S2_matrix(P,v)
end

"""
"""
function ActiveSpaceSolvers.apply_sminus(v::Matrix, ansatz::RASCIAnsatz_2)
    if ansatz.nb + 1 > ansatz.no#={{{=#
        error(" Can't decrease Ms further")
    end
    # Sm = b'a
    # = c(IJ,s) <IJ|b'a|KL> c(KL,t)
    # = c(IJ,s)c(KL,t) <J|<I|b'a|K>|L>
    # = c(IJ,s)c(KL,t) <J|<I|ab'|K>|L> (-1)
    # = c(IJ,s)c(KL,t) <J|<I|a|K>b'|L> (-1) (-1)^ket_a.ne
    # = c(IJ,s)c(KL,t) <I|a|K><J|b'|L> (-1) (-1)^ket_a.ne
    
    nroots = size(v,2)
    v2 = RASVector(v, ansatz)
    
    sgnK = -1
    if ansatz.na % 2 != 0 
        sgnK = -sgnK
    end

    bra_ansatz = RASCIAnsatz_2(ansatz.no, ansatz.na-1, ansatz.nb+1, ansatz.ras_spaces, max_h=ansatz.max_h, max_p=ansatz.max_p, max_h2=ansatz.max_h2, max_p2=ansatz.max_p2)
    wtmp = RASVector(zeros(bra_ansatz.dim, nroots), bra_ansatz)
    w = initalize_sig(wtmp)
    
    create_list = make_excitation_classes_c(ansatz.ras_spaces)
    ann_list = make_excitation_classes_a(ansatz.ras_spaces)
    ras1, ras2, ras3 = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(ansatz.ras_spaces)
    #note ras1 is same as ras1_bra so do not need to compute them
    
    for (block1, vec) in v2.data
        #loop over alpha strings
        idxa = 0
        det3a = SubspaceDeterminantString(ansatz.ras_spaces[3], block1.focka[3])
        for na in 1:det3a.max
            det2a = SubspaceDeterminantString(ansatz.ras_spaces[2], block1.focka[2])
            for ja in 1:det2a.max
                det1a = SubspaceDeterminantString(ansatz.ras_spaces[1], block1.focka[1])
                for ia in 1:det1a.max
                    idxa += 1
                    aconfig = [det1a.config;det2a.config.+det1a.no;det3a.config.+det1a.no.+det2a.no]

                    #now beta strings
                    idxb=0
                    det3b = SubspaceDeterminantString(ansatz.ras_spaces[3], block1.fockb[3])
                    for n in 1:det3b.max
                        det2b = SubspaceDeterminantString(ansatz.ras_spaces[2], block1.fockb[2])
                        for j in 1:det2b.max
                            det1b = SubspaceDeterminantString(ansatz.ras_spaces[1], block1.fockb[1])
                            for i in 1:det1b.max
                                idxb += 1
                                bconfig = [det1b.config;det2b.config.+det1b.no;det3b.config.+det1b.no.+det2b.no]
                                for (p_range, delta_c) in create_list
                                    for (q_range, delta_a) in ann_list
                                        block2 = RasBlock(block1.focka.+delta_a, block1.fockb.+delta_c)
                                        haskey(w, block2) || continue
                                        for q in q_range
                                            tmp = deepcopy(aconfig)
                                            if q in tmp
                                                sgn_q, det_a = apply_annihilation(tmp, q)
                                                sgn_q != 0 || continue
                                                d1_a, d2_a, d3_a = breakup_config(det_a, ras1, ras2, ras3)
                                                det1_q = SubspaceDeterminantString(length(ras1), length(d1_a), d1_a)
                                                det2_q = SubspaceDeterminantString(length(ras2), length(d2_a), d2_a.-length(ras1))
                                                det3_q = SubspaceDeterminantString(length(ras3), length(d3_a), d3_a.-length(ras1).-length(ras2))

                                                idxa_new = calc_full_ras_index(det1_q, det2_q, det3_q)
                                                for p in p_range
                                                    p == q || continue
                                                    tmp2 = deepcopy(bconfig)
                                                    if p in tmp2
                                                        continue
                                                    else
                                                        d1_c, d2_c, d3_c = breakup_config(tmp2, ras1, ras2, ras3)
                                                        sgn_p, idxb_new = apply_creation!(d1_c, d2_c, d3_c, ras1, ras2, ras3, p)
                                                        sgn_p != 0 || continue

                                                        w[block2][idxa_new, idxb_new,:] .+= sgnK*sgn_q*sgn_p*vec[idxa,idxb,:]
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

    starti = 1
    w2 = zeros(Float64, bra_ansatz.dim, nroots)
    for (block, vec) in w
        tmp = reshape(vec, (size(vec,1)*size(vec,2), nroots))
        w2[starti:starti+(size(vec,1)*size(vec,2))-1, :] .= tmp
        starti += (size(vec,1)*size(vec,2))
    end
    
    #only keep the states that aren't zero (that weren't killed by S-)
    wout = zeros(size(w2,1),0)
    for i in 1:nroots
        ni = norm(w2[:,i])
        if isapprox(ni, 0, atol=1e-4) == false
            wout = hcat(wout, w2[:,i]./ni)
        end
    end

    return wout, bra_ansatz#=}}}=#
end

"""
"""
function ActiveSpaceSolvers.apply_splus(v::Matrix, ansatz::RASCIAnsatz_2)

    # Sp = a'b{{{
    # = c(IJ,s) <IJ|a'b|KL> c(KL,t)
    # = c(IJ,s)c(KL,t) <J|<I|a'b|K>|L>
    # = c(IJ,s)c(KL,t) <J|<I|a'|K>b|L> (-1)^ket_a.ne
    # = c(IJ,s)c(KL,t) <I|a'|K><J|b|L> (-1)^ket_a.ne

    nroots = size(v,2)
    v2 = RASVector(v, ansatz)
    
    if ansatz.na + 1 > ansatz.no
        error(" Can't increase Ms further")
    end

    sgnK = 1
    if ansatz.na % 2 != 0 
        sgnK = -sgnK
    end
    
    bra_ansatz = RASCIAnsatz_2(ansatz.no, ansatz.na+1, ansatz.nb-1, ansatz.ras_spaces, max_h=ansatz.max_h, max_p=ansatz.max_p, max_h2=ansatz.max_h2, max_p2=ansatz.max_p2)
    wtmp = RASVector(zeros(bra_ansatz.dim, nroots), bra_ansatz)
    w = initalize_sig(wtmp)
    
    create_list = make_excitation_classes_c(ansatz.ras_spaces)
    ann_list = make_excitation_classes_a(ansatz.ras_spaces)
    ras1, ras2, ras3 = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(ansatz.ras_spaces)
    #note ras1 is same as ras1_bra so do not need to compute them
    
    for (block1, vec) in v2.data
        #loop over alpha strings
        idxa = 0
        det3a = SubspaceDeterminantString(ansatz.ras_spaces[3], block1.focka[3])
        for na in 1:det3a.max
            det2a = SubspaceDeterminantString(ansatz.ras_spaces[2], block1.focka[2])
            for ja in 1:det2a.max
                det1a = SubspaceDeterminantString(ansatz.ras_spaces[1], block1.focka[1])
                for ia in 1:det1a.max
                    idxa += 1
                    aconfig = [det1a.config;det2a.config.+det1a.no;det3a.config.+det1a.no.+det2a.no]

                    #now beta strings
                    idxb=0
                    det3b = SubspaceDeterminantString(ansatz.ras_spaces[3], block1.fockb[3])
                    for n in 1:det3b.max
                        det2b = SubspaceDeterminantString(ansatz.ras_spaces[2], block1.fockb[2])
                        for j in 1:det2b.max
                            det1b = SubspaceDeterminantString(ansatz.ras_spaces[1], block1.fockb[1])
                            for i in 1:det1b.max
                                idxb += 1
                                bconfig = [det1b.config;det2b.config.+det1b.no;det3b.config.+det1b.no.+det2b.no]
                                for (p_range, delta_c) in create_list
                                    for (q_range, delta_a) in ann_list
                                        block2 = RasBlock(block1.focka.+delta_c, block1.fockb.+delta_a)
                                        haskey(w, block2) || continue
                                        for q in q_range
                                            tmp = deepcopy(bconfig)
                                            if q in tmp
                                                sgn_q, det_a = apply_annihilation(tmp, q)
                                                sgn_q != 0 || continue
                                                d1_a, d2_a, d3_a = breakup_config(det_a, ras1, ras2, ras3)
                                                det1_q = SubspaceDeterminantString(length(ras1), length(d1_a), d1_a)
                                                det2_q = SubspaceDeterminantString(length(ras2), length(d2_a), d2_a.-length(ras1))
                                                det3_q = SubspaceDeterminantString(length(ras3), length(d3_a), d3_a.-length(ras1).-length(ras2))

                                                idxb_new = calc_full_ras_index(det1_q, det2_q, det3_q)
                                                for p in p_range
                                                    p == q || continue
                                                    tmp2 = deepcopy(aconfig)
                                                    if p in tmp2
                                                        continue
                                                    else
                                                        d1_c, d2_c, d3_c = breakup_config(tmp2, ras1, ras2, ras3)
                                                        sgn_p, idxa_new = apply_creation!(d1_c, d2_c, d3_c, ras1, ras2, ras3, p)
                                                        sgn_p != 0 || continue

                                                        w[block2][idxa_new, idxb_new,:] .+= sgnK*sgn_q*sgn_p*vec[idxa,idxb,:]
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

    starti = 1
    w2 = zeros(Float64, bra_ansatz.dim, nroots)
    for (block, vec) in w
        tmp = reshape(vec, (size(vec,1)*size(vec,2), nroots))
        w2[starti:starti+(size(vec,1)*size(vec,2))-1, :] .= tmp
        starti += (size(vec,1)*size(vec,2))
    end
    
    #only keep the states that aren't zero (that weren't killed by S-)
    wout = zeros(size(w2,1),0)
    for i in 1:nroots
        ni = norm(w2[:,i])
        display(ni)
        if isapprox(ni, 0, atol=1e-4) == false
            wout = hcat(wout, w2[:,i]./ni)
        end
    end

    return wout, bra_ansatz#=}}}=#
end

"""
    build_H_matrix(ints, P::RASCIAnsatz_2)

Build the Hamiltonian defined by `ints` in the Slater Determinant Basis  specified by `P`
"""
function ActiveSpaceSolvers.build_H_matrix(ints::InCoreInts, prob::RASCIAnsatz_2)
    nr = prob.dim
    v = Matrix(1.0I, nr, nr)
    rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(v, prob)
    lu = ActiveSpaceSolvers.RASCI_2.fill_lu(rasvec, prob.ras_spaces)
    sigma1 = ActiveSpaceSolvers.RASCI_2.sigma_one(rasvec, ints, prob.ras_spaces, lu)
    sigma2 = ActiveSpaceSolvers.RASCI_2.sigma_two(rasvec, ints, prob.ras_spaces, lu)
    sigma3 = ActiveSpaceSolvers.RASCI_2.sigma_three(rasvec, ints, prob.ras_spaces, lu)

    sig = sigma1 + sigma2 + sigma3

    Hmat = .5*(sig+sig')
    Hmat += 1.0I*ints.h0
    return Hmat
end

"""
    compute_1rdm(sol::Solution{RASCIAnsatz_2,T}; root=1) where {T}
"""
function ActiveSpaceSolvers.compute_1rdm(sol::Solution{RASCIAnsatz_2,T}; root=1) where {T}
    return ActiveSpaceSolvers.RASCI_2.compute_1rdm(sol.ansatz, sol.vectors[:,root])
end

"""
    compute_1rdm_2rdm(sol::Solution{A,T}; root=1) where {A,T}
"""
function ActiveSpaceSolvers.compute_1rdm_2rdm(sol::Solution{RASCIAnsatz_2,T}; root=1) where {T}
    return ActiveSpaceSolvers.RASCI_2.compute_1rdm_2rdm(sol.ansatz, sol.vectors[:,root])
end

"""
    compute_operator_c_a(bra::Solution{RASCIAnsatz_2,T}, ket::Solution{RASCIAnsatz_2,T}) where {T}

Compute representation of a operator between states `bra_v` and `ket_v` for alpha
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_c_a(bra::Solution{RASCIAnsatz_2,T}, 
                                                 ket::Solution{RASCIAnsatz_2,T}) where {T}
    return ActiveSpaceSolvers.RASCI_2.compute_operator_c_a(bra::Solution{RASCIAnsatz_2}, ket::Solution{RASCIAnsatz_2})
end

"""
    compute_operator_a_b(bra::Solution{RASCIAnsatz_2,T}, ket::Solution{RASCIAnsatz_2,T}) where {T}

Compute representation of a operator between states `bra_v` and `ket_v` for beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_c_b(bra::Solution{RASCIAnsatz_2,T}, 
                                                 ket::Solution{RASCIAnsatz_2,T}) where {T}
    return ActiveSpaceSolvers.RASCI_2.compute_operator_c_b(bra::Solution{RASCIAnsatz_2}, 
                                                 ket::Solution{RASCIAnsatz_2})
end

"""
    compute_operator_ca_aa(bra::Solution{RASCIAnsatz_2,T}, ket::Solution{RASCIAnsatz_2,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for alpha-alpha
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_ca_aa(bra::Solution{RASCIAnsatz_2,T}, 
                                                   ket::Solution{RASCIAnsatz_2,T}) where {T}
    return ActiveSpaceSolvers.RASCI_2.compute_operator_ca_aa(bra::Solution{RASCIAnsatz_2}, 
                                                 ket::Solution{RASCIAnsatz_2})
end

"""
    compute_operator_ca_bb(bra::Solution{RASCIAnsatz_2,T}, ket::Solution{RASCIAnsatz_2,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_ca_bb(bra::Solution{RASCIAnsatz_2,T}, 
                                                   ket::Solution{RASCIAnsatz_2,T}) where {T}
    return ActiveSpaceSolvers.RASCI_2.compute_operator_ca_bb(bra::Solution{RASCIAnsatz_2}, 
                                                 ket::Solution{RASCIAnsatz_2})
end


"""
    compute_operator_ca_ab(bra::Solution{RASCIAnsatz_2,T}, ket::Solution{RASCIAnsatz_2,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for alpha-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_ca_ab(bra::Solution{RASCIAnsatz_2,T}, 
                                                   ket::Solution{RASCIAnsatz_2,T}) where {T}
    return ActiveSpaceSolvers.RASCI_2.compute_operator_ca_ab(bra::Solution{RASCIAnsatz_2}, 
                                                           ket::Solution{RASCIAnsatz_2})
end


"""
    compute_operator_cc_aa(bra::Solution{RASCIAnsatz_2,T}, ket::Solution{RASCIAnsatz_2,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cc_bb(bra::Solution{RASCIAnsatz_2,T}, 
                                                   ket::Solution{RASCIAnsatz_2,T}) where {T}
    return ActiveSpaceSolvers.RASCI_2.compute_operator_cc_bb(bra::Solution{RASCIAnsatz_2}, 
                                                 ket::Solution{RASCIAnsatz_2})
end


"""
    compute_operator_cc_aa(bra::Solution{RASCIAnsatz_2,T}, ket::Solution{RASCIAnsatz_2,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for alpha-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cc_aa(bra::Solution{RASCIAnsatz_2,T}, 
                                                   ket::Solution{RASCIAnsatz_2,T}) where {T}
    return ActiveSpaceSolvers.RASCI_2.compute_operator_cc_aa(bra::Solution{RASCIAnsatz_2}, 
                                                 ket::Solution{RASCIAnsatz_2})
end


"""
    compute_operator_cc_ab(bra::Solution{RASCIAnsatz_2,T}, ket::Solution{RASCIAnsatz_2,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for alpha-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cc_ab(bra::Solution{RASCIAnsatz_2,T}, 
                                                   ket::Solution{RASCIAnsatz_2,T}) where {T}
    return ActiveSpaceSolvers.RASCI_2.compute_operator_cc_ab(bra::Solution{RASCIAnsatz_2}, 
                                                 ket::Solution{RASCIAnsatz_2})
end


"""
    compute_operator_cca_aaa(bra::Solution{RASCIAnsatz_2,T}, ket::Solution{RASCIAnsatz_2,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-alpha-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_aaa(bra::Solution{RASCIAnsatz_2,T}, 
                                                   ket::Solution{RASCIAnsatz_2,T}) where {T}
    return ActiveSpaceSolvers.RASCI_2.compute_operator_cca_aaa(bra::Solution{RASCIAnsatz_2}, 
                                                 ket::Solution{RASCIAnsatz_2})
end


"""
    compute_operator_cca_bbb(bra::Solution{RASCIAnsatz_2,T}, ket::Solution{RASCIAnsatz_2,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for beta-beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_bbb(bra::Solution{RASCIAnsatz_2,T}, 
                                                   ket::Solution{RASCIAnsatz_2,T}) where {T}
    return ActiveSpaceSolvers.RASCI_2.compute_operator_cca_bbb(bra::Solution{RASCIAnsatz_2}, 
                                                 ket::Solution{RASCIAnsatz_2})
end


"""
    compute_operator_cca_aba(bra::Solution{RASCIAnsatz_2,T}, ket::Solution{RASCIAnsatz_2,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-beta-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_aba(bra::Solution{RASCIAnsatz_2,T}, 
                                                     ket::Solution{RASCIAnsatz_2,T}) where {T}
    return ActiveSpaceSolvers.RASCI_2.compute_operator_cca_aba(bra::Solution{RASCIAnsatz_2}, 
                                                             ket::Solution{RASCIAnsatz_2})
end


"""
    compute_operator_cca_abb(bra::Solution{RASCIAnsatz_2,T}, ket::Solution{RASCIAnsatz_2,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_abb(bra::Solution{RASCIAnsatz_2,T}, 
                                                     ket::Solution{RASCIAnsatz_2,T}) where {T}
    return ActiveSpaceSolvers.RASCI_2.compute_operator_cca_abb(bra::Solution{RASCIAnsatz_2}, 
                                                 ket::Solution{RASCIAnsatz_2}) 
end


    

