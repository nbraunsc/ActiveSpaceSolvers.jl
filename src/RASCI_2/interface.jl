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
function RASCIAnsatz_2(no::Int, na, nb, ras_spaces::Any; h=0, p=ras_spaces[3], h2=0, p2=0)
    na <= no || throw(DimensionMismatch)
    nb <= no || throw(DimensionMismatch)
    sum(ras_spaces) == no || throw(DimensionMismatch)
    ras_spaces = convert(SVector{3,Int},collect(ras_spaces))
    na = convert(Int, na)
    nb = convert(Int, nb)
    max_h = convert(Int8, h)
    max_p = convert(Int8, p)
    max_h2 = convert(Int8, h2)
    max_p2 = convert(Int8, p2)
    rdim = calc_rdim(ras_spaces, na, nb, max_h, max_p, max_h2, max_p2)
    return RASCIAnsatz_2(no, na, nb, rdim, ras_spaces, max_h, max_p, max_h2, max_p2);
end



function Base.display(p::RASCIAnsatz_2)
    @printf(" RASCIAnsatz:: #Orbs = %-3i #α = %-2i #β = %-2i Fock Spaces: (%i, %i, %i) RASCI Dimension: %-3i MAX Holes: %i MAX Particles: %i\n",p.no,p.na,p.nb,p.ras_spaces[1], p.ras_spaces[2], p.ras_spaces[3], p.dim, p.max_h, p.max_p)
end

function Base.print(p::RASCIAnsatz_2)
    @printf(" RASCIAnsatz:: #Orbs = %-3i #α = %-2i #β = %-2i Fock Spaces: (%i, %i, %i) RASCI Dimension: %-3i MAX Holes: %i MAX Particles: %i\n",p.no,p.na,p.nb,p.ras_spaces[1], p.ras_spaces[2], p.ras_spaces[3], p.dim, p.max_h, p.max_p)
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
        @printf(" Iter: %4i", iters)
        #print("Iter: ", iters, " ")
        #@printf(" %-50s", "Compute sigma 1: ")
        flush(stdout)
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
    LinOpMat(ints, prb::FCIAnsatz)

Get LinearMap with takes a vector and returns action of H on that vector

# Arguments
- ints: `InCoreInts` object
- prb:  `FCIAnsatz` object
"""
function BlockDavidson.LinOpMat(ints::InCoreInts{T}, prb::RASCIAnsatz) where T

    iters = 0
    function mymatvec(v)
        rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(v, prob)
        lu = ActiveSpaceSolvers.RASCI_2.fill_lu(rasvec, prob.ras_spaces)

        iters += 1
        @printf(" Iter: %4i", iters)
        #print("Iter: ", iters, " ")
        #@printf(" %-50s", "Compute sigma 1: ")
        flush(stdout)
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

function calc_rdim(ras_spaces::SVector{3, Int}, na::Int, nb::Int, max_h::Int8, max_p::Int8, max_h2::Int8, max_p2::Int8)
    a_blocks, fock_as = make_blocks(ras_spaces, na, max_h, max_p)
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
    end
    
    if max_h2 != 0 && max_p2 != 0
        a_blocks2, fock_as2 = make_blocks(ras_spaces, na, max_h2, max_p2)
        b_blocks2, fock_bs2 = make_blocks(ras_spaces, nb, max_h2, max_p2)
        for i in 1:length(a_blocks2)
            dima = binomial(ras_spaces[1], fock_as2[i][1])*binomial(ras_spaces[2], fock_as2[i][2])*binomial(ras_spaces[3], fock_as2[i][3])
            for j in 1:length(b_blocks2)
                dimb = binomial(ras_spaces[1], fock_bs2[j][1])*binomial(ras_spaces[2], fock_bs2[j][2])*binomial(ras_spaces[3], fock_bs2[j][3])
                if a_blocks2[i][1]+b_blocks2[j][1]<= max_h2
                    if a_blocks2[i][2]+b_blocks2[j][2] <= max_p2
                        start += dima*dimb
                    end
                end
            end
        end
    end
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
    build_S2_matrix(P::RASCIAnsatz)

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
    
    bra_ansatz = RASCIAnsatz(ansatz.no, ansatz.na-1, ansatz.nb+1, ansatz.ras_spaces,  max_h=ansatz.max_h, max_p=ansatz.max_p)
    
    cats_a_bra, cats_a = ActiveSpaceSolvers.RASCI.fill_lu_HP(bra_ansatz, ansatz, spin="alpha", type="a")
    cats_b_bra, cats_b = ActiveSpaceSolvers.RASCI.fill_lu_HP(bra_ansatz, ansatz, spin="beta", type="c")
    spin_pairs = ActiveSpaceSolvers.RASCI.make_spin_pairs(ansatz, cats_a, cats_b)
    spin_pairs_bra = ActiveSpaceSolvers.RASCI.make_spin_pairs(bra_ansatz, cats_a_bra, cats_b_bra)
    
    v2 = Dict{Int, Array{Float64, 3}}()
    start = 1
    for m in 1:length(spin_pairs)
        tmp = v[start:start+spin_pairs[m].dim-1, :]
        v2[m] = reshape(tmp, (length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs), nroots))
        start += spin_pairs[m].dim
    end
    
    w = Dict{Int, Array{Float64, 3}}()
    for m in 1:length(spin_pairs_bra)
        w[m] = zeros(length(cats_a_bra[spin_pairs_bra[m].pair[1]].idxs), length(cats_b_bra[spin_pairs_bra[m].pair[2]].idxs), nroots)
    end

    sgnK = -1
    if ansatz.na % 2 != 0 
        sgnK = -sgnK
    end
    
    for m in 1:length(spin_pairs)
        cat_Ia = cats_a[spin_pairs[m].pair[1]]
        cat_Ib = cats_b[spin_pairs[m].pair[2]]
        for Ib in cats_b[spin_pairs[m].pair[2]].idxs
            Ib_local = Ib-cat_Ib.shift
            for Ia in cats_a[spin_pairs[m].pair[1]].idxs
                Ia_local = Ia-cat_Ia.shift
                for p in 1:ansatz.no
                    Ja = cat_Ia.lookup[p,Ia_local]
                    Ja != 0 || continue
                    Ja_sign = sign(Ja)
                    Ja = abs(Ja)
                    cata_Ja = find_cat(Ja, cats_a_bra)
                    Jb = cat_Ib.lookup[p,Ib_local]
                    Jb != 0 || continue
                    Jb_sign = sign(Jb)
                    Jb = abs(Jb)
                    catb_Jb = find_cat(Jb, cats_b_bra)
                    n = find_spin_pair(spin_pairs_bra, (cata_Ja.idx, catb_Jb.idx))
                    n != 0 || continue
                    Ja_local = Ja-cata_Ja.shift
                    Jb_local = Jb-catb_Jb.shift
                    w[n][Ja_local, Jb_local, :] .+= sgnK*Ja_sign*Jb_sign*v2[m][Ia_local, Ib_local, :]
                end
            end
        end
    end
    
    starti = 1
    w2 = zeros(Float64, bra_ansatz.dim, nroots)
    for m in 1:length(spin_pairs_bra)
        tmp = reshape(w[m], (size(w[m],1)*size(w[m],2), nroots))
        w2[starti:starti+spin_pairs_bra[m].dim-1, :] .= tmp
        starti += spin_pairs_bra[m].dim
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
    
    if ansatz.na + 1 > ansatz.no
        error(" Can't increase Ms further")
    end

    sgnK = 1
    if ansatz.na % 2 != 0 
        sgnK = -sgnK
    end

    bra_ansatz = RASCIAnsatz(ansatz.no, ansatz.na+1, ansatz.nb-1, ansatz.ras_spaces,  max_h=ansatz.max_h, max_p=ansatz.max_p)
    
    cats_a_bra, cats_a = ActiveSpaceSolvers.RASCI.fill_lu_HP(bra_ansatz, ansatz, spin="alpha", type="c")
    cats_b_bra, cats_b = ActiveSpaceSolvers.RASCI.fill_lu_HP(bra_ansatz, ansatz, spin="beta", type="a")
    spin_pairs = ActiveSpaceSolvers.RASCI.make_spin_pairs(ansatz, cats_a, cats_b)
    spin_pairs_bra = ActiveSpaceSolvers.RASCI.make_spin_pairs(bra_ansatz, cats_a_bra, cats_b_bra)
    
    v2 = Dict{Int, Array{Float64, 3}}()
    start = 1
    for m in 1:length(spin_pairs)
        tmp = v[start:start+spin_pairs[m].dim-1, :]
        v2[m] = reshape(tmp, (length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs), nroots))
        start += spin_pairs[m].dim
    end
    
    w = Dict{Int, Array{Float64, 3}}()
    for m in 1:length(spin_pairs_bra)
        w[m] = zeros(length(cats_a_bra[spin_pairs_bra[m].pair[1]].idxs), length(cats_b_bra[spin_pairs_bra[m].pair[2]].idxs), nroots)
    end
    
    for m in 1:length(spin_pairs)
        cat_Ia = cats_a[spin_pairs[m].pair[1]]
        cat_Ib = cats_b[spin_pairs[m].pair[2]]
        for Ib in cats_b[spin_pairs[m].pair[2]].idxs
            Ib_local = Ib-cat_Ib.shift
            for Ia in cats_a[spin_pairs[m].pair[1]].idxs
                Ia_local = Ia-cat_Ia.shift
                for p in 1:ansatz.no
                    Ja = cat_Ia.lookup[p,Ia_local]
                    Ja != 0 || continue
                    Ja_sign = sign(Ja)
                    Ja = abs(Ja)
                    cata_Ja = find_cat(Ja, cats_a_bra)
                    Jb = cat_Ib.lookup[p,Ib_local]
                    Jb != 0 || continue
                    Jb_sign = sign(Jb)
                    Jb = abs(Jb)
                    catb_Jb = find_cat(Jb, cats_b_bra)
                    n = find_spin_pair(spin_pairs_bra, (cata_Ja.idx, catb_Jb.idx))
                    n != 0 || continue
                    Ja_local = Ja-cata_Ja.shift
                    Jb_local = Jb-catb_Jb.shift
                    w[n][Ja_local, Jb_local, :] .+= sgnK*Ja_sign*Jb_sign*v2[m][Ia_local, Ib_local, :]
                end
            end
        end
    end
    
    starti = 1
    w2 = zeros(Float64, bra_ansatz.dim, nroots)
    for m in 1:length(spin_pairs_bra)
        tmp = reshape(w[m], (size(w[m],1)*size(w[m],2), nroots))
        w2[starti:starti+spin_pairs_bra[m].dim-1, :] .= tmp
        starti += spin_pairs_bra[m].dim
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
    build_H_matrix(ints, P::RASCIAnsatz)

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

    

