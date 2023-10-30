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

"""
Type containing all the metadata needed to define a RASCI problem 

    no::Int  # number of orbitals
    na::Int  # number of alpha
    nb::Int  # number of beta
    dima::Int 
    dimb::Int 
    dim::Int
    ras_spaces::SVector{3, Int}   # Number of orbitals in each ras space (RAS1, RAS2, RAS3)
    max_h::Int  #max number of holes in ras1 (GLOBAL, Slater Det)
    max_p::Int #max number of particles in ras3 (GLOBAL, Slater Det)
"""
struct RASCIAnsatz <: Ansatz
    no::Int  # number of orbitals
    na::Int  # number of alpha
    nb::Int  # number of beta
    dima::Int 
    dimb::Int 
    dim::Int
    ras_spaces::SVector{3, Int}   # Number of orbitals in each ras space (RAS1, RAS2, RAS3)
    max_h::Int  #max number of holes in ras1 (GLOBAL, Slater Det)
    max_p::Int #max number of particles in ras3 (GLOBAL, Slater Det)
end

"""
    RASCIAnsatz(no, na, nb, ras_spaces::Any, ras1_min=1, ras3_max=2)
Constructor
# Arguments
- `no`: Number of spatial orbitals
- `na`: Number of α electrons
- `nb`: Number of β electrons
- `ras_spaces`: Number of orbitals in each (RAS1, RAS2, RAS3)
- `max_h`: Max number of holes in RAS1
- `max_p`: Max number of particles in RAS3
"""
function RASCIAnsatz(no::Int, na, nb, ras_spaces::Any; max_h=0, max_p=ras_spaces[3])
    na <= no || throw(DimensionMismatch)
    nb <= no || throw(DimensionMismatch)
    sum(ras_spaces) == no || throw(DimensionMismatch)
    ras_spaces = convert(SVector{3,Int},collect(ras_spaces))
    na = convert(Int, na)
    nb = convert(Int, nb)
    tmp = RASCIAnsatz(no, na, nb, ras_spaces, max_h, max_p)
    dima, dimb, ras_dim = calc_ras_dim(tmp)
    return RASCIAnsatz(no, na, nb, dima, dimb, ras_dim, ras_spaces, max_h, max_p);
end

function RASCIAnsatz(no::Int, na::Int, nb::Int, ras_spaces::SVector{3,Int}, max_h, max_p)
    return RASCIAnsatz(no, na, nb, 0, 0, 0, ras_spaces, max_h, max_p);
end

function Base.display(p::RASCIAnsatz)
    @printf(" RASCIAnsatz:: #Orbs = %-3i #α = %-2i #β = %-2i Fock Spaces: (%i, %i, %i) RASCI Dimension: %-3i MAX Holes: %i MAX Particles: %i\n",p.no,p.na,p.nb,p.ras_spaces[1], p.ras_spaces[2], p.ras_spaces[3], p.dim, p.max_h, p.max_p)
end

function Base.print(p::RASCIAnsatz)
    @printf(" RASCIAnsatz:: #Orbs = %-3i #α = %-2i #β = %-2i Fock Spaces: (%i, %i, %i) RASCI Dimension: %-3i MAX Holes: %i MAX Particles: %i\n",p.no,p.na,p.nb,p.ras_spaces[1], p.ras_spaces[2], p.ras_spaces[3], p.dim, p.max_h, p.max_p)
end

"""
    LinearMap(ints, prb::RASCIAnsatz)

Get LinearMap with takes a vector and returns action of H on that vector

# Arguments
- ints: `InCoreInts` object
- prb:  `RASCIAnsatz` object
"""
function LinearMaps.LinearMap(ints::InCoreInts, prb::RASCIAnsatz)
    spin_pairs, a_categories, b_categories, = ActiveSpaceSolvers.RASCI.make_spin_pairs(prb)

    iters = 0
    function mymatvec(v)
        iters += 1
        #@printf(" Iter: %4i", iters)
        #print("Iter: ", iters, " ")
        #@printf(" %-50s", "Compute sigma 1: ")
        #flush(stdout)
        #display(size(v))
       
        nr = 0
        if length(size(v)) == 1
            nr = 1
            v = reshape(v,prb.dim, nr)
        else 
            nr = size(v)[2]
        end
        #v = reshape(v, prb.dima, prb.dimb, nr)
        
        sigma1 = ActiveSpaceSolvers.RASCI.sigma_one(prb, spin_pairs, a_categories, b_categories, ints, v)
        sigma2 = ActiveSpaceSolvers.RASCI.sigma_two(prb, spin_pairs, a_categories, b_categories, ints, v)
        sigma3 = ActiveSpaceSolvers.RASCI.sigma_three(prb, spin_pairs, a_categories, b_categories, ints, v)
        
        sig = sigma1 + sigma2 + sigma3
        
        #v = reshape(v,prb.dim, nr)
        #sig = reshape(sig, prb.dim, nr)
        sig .+= ints.h0*v
        return sig
    end
    return LinearMap(mymatvec, prb.dim, prb.dim, issymmetric=true, ismutating=false, ishermitian=true)
end

"""
    LinOpMat(ints, prb::FCIAnsatz)

Get LinearMap with takes a vector and returns action of H on that vector

# Arguments
- ints: `InCoreInts` object
- prb:  `FCIAnsatz` object
"""
function BlockDavidson.LinOpMat(ints::InCoreInts{T}, prb::RASCIAnsatz) where T
    spin_pairs, a_categories, b_categories, = ActiveSpaceSolvers.RASCI.make_spin_pairs(prb)

    iters = 0
    function mymatvec(v)
        iters += 1
        #@printf(" Iter: %4i", iters)
        #print("Iter: ", iters, " ")
        #flush(stdout)
        #display(size(v))
       
        nr = 0
        if length(size(v)) == 1
            nr = 1
            v = reshape(v,prb.dim, nr)
        else 
            nr = size(v)[2]
        end
        sigma1 = ActiveSpaceSolvers.RASCI.sigma_one(prb, spin_pairs, a_categories, b_categories, ints, v)
        sigma2 = ActiveSpaceSolvers.RASCI.sigma_two(prb, spin_pairs, a_categories, b_categories, ints, v)
        sigma3 = ActiveSpaceSolvers.RASCI.sigma_three(prb, spin_pairs, a_categories, b_categories, ints, v)
        
        sig = sigma1 + sigma2 + sigma3
        
        sig .+= ints.h0*v
        return sig
    end
    return LinOpMat{T}(mymatvec, prb.dim, true)
end

function ras_calc_ndets(no, nelec, ras_spaces, ras1_min, ras3_max)
    x = ActiveSpaceSolvers.RASCI.make_ras_x(no, nelec, ras_spaces, ras1_min, ras3_max)
    dim_x = findmax(x)[1]
    #dim_x = no 
    return dim_x, x
end

function calc_ndets(no,nelec)
    if no > 20
        x = factorial(big(no))
        y = factorial(nelec)
        z = factorial(big(no-nelec))
        return Int64(x÷(y*z))
    end

    return factorial(no)÷(factorial(nelec)*factorial(no-nelec))
end

function calc_ras_dim_old(prob::RASCIAnsatz)
    all_cats_a = Vector{HP_Category_CA}()#={{{=#
    all_cats_b = Vector{HP_Category_CA}()
    categories = ActiveSpaceSolvers.RASCI.generate_spin_categories(prob)
    cats_a = deepcopy(categories)
    cats_b = deepcopy(categories)
    fock_list_a, del_at_a = make_fock_from_categories(categories, prob, "alpha")
    deleteat!(cats_a, del_at_a)
    len_cat_a = length(cats_a)
        
    fock_list_b, del_at_b = make_fock_from_categories(categories, prob, "beta")
    deleteat!(cats_b, del_at_b)
    len_cat_b = length(cats_b)
    
    #compute alpha configs
    connected = make_spincategory_connections(cats_a, cats_b, prob)
    as = compute_config_dict(fock_list_a, prob, "alpha")
    rev_as = Dict(value => key for (key, value) in as)
    max_a = length(as)

    for j in 1:len_cat_a
        idxas = Vector{Int}()
        graph_a = make_cat_graphs(fock_list_a[j], prob)
        idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
        lu = zeros(Int, graph_a.no, graph_a.no, length(idxas))
        cat_lu = zeros(Int, graph_a.no, graph_a.no, length(idxas))
        push!(all_cats_a, HP_Category_CA(j, cats_a[j], connected[j], idxas, 0, lu, cat_lu))

    end
        
    
    #compute beta configs
    connected_b = make_spincategory_connections(cats_b, cats_a, prob)
    bs = compute_config_dict(fock_list_b, prob, "beta")
    rev_bs = Dict(value => key for (key, value) in bs)
    max_b = length(bs)

    for j in 1:len_cat_b
        idxbs = Vector{Int}()
        graph_b = make_cat_graphs(fock_list_b[j], prob)
        idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
        lu = zeros(Int, graph_b.no, graph_b.no, length(idxbs))
        cat_lu = zeros(Int, graph_b.no, graph_b.no, length(idxbs))
        push!(all_cats_b, HP_Category_CA(j, cats_b[j], connected_b[j], idxbs,0, lu, cat_lu))
    end

    count = 0
    for Ia in 1:max_a
        cat_Ia = ActiveSpaceSolvers.RASCI.find_cat(Ia, all_cats_a)
        for m in cat_Ia.connected
            catb_Ib = all_cats_b[m]
            for Ib in catb_Ib.idxs
                count += 1
            end
        end
    end
    return max_a, max_b,count#=}}}=#
end

function calc_ras_dim(prob::RASCIAnsatz)
    all_cats_a = Vector{Tuple{Vector{Int}, Vector{Int}}}()
    all_cats_b = Vector{Tuple{Vector{Int}, Vector{Int}}}()
    categories = ActiveSpaceSolvers.RASCI.generate_spin_categories(prob)
    cats_a = deepcopy(categories)
    cats_b = deepcopy(categories)
    fock_list_a, del_at_a = make_fock_from_categories(categories, prob, "alpha")
    deleteat!(cats_a, del_at_a)
    len_cat_a = length(cats_a)
        
    fock_list_b, del_at_b = make_fock_from_categories(categories, prob, "beta")
    deleteat!(cats_b, del_at_b)
    len_cat_b = length(cats_b)
    
    #compute alpha configs
    connected = make_spincategory_connections(cats_a, cats_b, prob)
    as = compute_config_dict(fock_list_a, prob, "alpha")
    rev_as = Dict(value => key for (key, value) in as)
    max_a = length(as)

    for j in 1:len_cat_a
        idxas = Vector{Int}()
        graph_a = make_cat_graphs(fock_list_a[j], prob)
        idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
        push!(all_cats_a, (connected[j], idxas))
    end
        
    #compute beta configs
    connected_b = make_spincategory_connections(cats_b, cats_a, prob)
    bs = compute_config_dict(fock_list_b, prob, "beta")
    rev_bs = Dict(value => key for (key, value) in bs)
    max_b = length(bs)

    for j in 1:len_cat_b
        idxbs = Vector{Int}()
        graph_b = make_cat_graphs(fock_list_b[j], prob)
        idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
        push!(all_cats_b, (connected_b[j], idxbs))
    end
    
    count = 0
    for i in 1:length(all_cats_a)
        dima = length(all_cats_a[i][2])
        for j in all_cats_a[i][1]
            dimb = length(all_cats_b[j][2])
            count += dima*dimb
        end
    end

    return max_a, max_b,count
end

"""
    ActiveSpaceSolvers.compute_s2(sol::Solution)

Compute the <S^2> expectation values for each state in `sol`
"""
function ActiveSpaceSolvers.compute_s2(sol::Solution{RASCIAnsatz,T}) where {T}
    return compute_S2_expval(sol.vectors, sol.ansatz)
end

"""
    build_S2_matrix(P::RASCIAnsatz)

Build the S2 matrix in the Slater Determinant Basis  specified by `P`
"""
function ActiveSpaceSolvers.apply_S2_matrix(P::RASCIAnsatz, v::AbstractArray{T}) where T
    return apply_S2_matrix(P,v)
end

"""
"""
function ActiveSpaceSolvers.apply_sminus(v::Matrix, ansatz::RASCIAnsatz)
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
function ActiveSpaceSolvers.apply_splus(v::Matrix, ansatz::RASCIAnsatz)

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
function ActiveSpaceSolvers.build_H_matrix(ints::InCoreInts, p::RASCIAnsatz)
    spin_pairs, a_categories, b_categories, = ActiveSpaceSolvers.RASCI.make_spin_pairs(p)
    nr = p.dim
    v = Matrix(1.0I, nr, nr)
    sigma1 = ActiveSpaceSolvers.RASCI.sigma_one(p, spin_pairs, a_categories, b_categories, ints, v)
    sigma2 = ActiveSpaceSolvers.RASCI.sigma_two(p, spin_pairs, a_categories, b_categories, ints, v)
    sigma3 = ActiveSpaceSolvers.RASCI.sigma_three(p, spin_pairs, a_categories, b_categories, ints, v)

    sig = sigma1 + sigma2 + sigma3

    Hmat = .5*(sig+sig')
    Hmat += 1.0I*ints.h0
    return Hmat
end

"""
    compute_operator_c_a(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a operator between states `bra_v` and `ket_v` for alpha
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_c_a(bra::Solution{RASCIAnsatz,T}, 
                                                 ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_c_a(bra::Solution{RASCIAnsatz}, ket::Solution{RASCIAnsatz})
end

"""
    compute_operator_a_b(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a operator between states `bra_v` and `ket_v` for beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_c_b(bra::Solution{RASCIAnsatz,T}, 
                                                 ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_c_b(bra::Solution{RASCIAnsatz}, 
                                                 ket::Solution{RASCIAnsatz})
end

"""
    compute_operator_ca_aa(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for alpha-alpha
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_ca_aa(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_ca_aa(bra::Solution{RASCIAnsatz}, 
                                                 ket::Solution{RASCIAnsatz})
end

"""
    compute_operator_ca_bb(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_ca_bb(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_ca_bb(bra::Solution{RASCIAnsatz}, 
                                                 ket::Solution{RASCIAnsatz})
end


"""
    compute_operator_ca_ab(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for alpha-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_ca_ab(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_ca_ab(bra::Solution{RASCIAnsatz}, 
                                                           ket::Solution{RASCIAnsatz})
end


"""
    compute_operator_cc_aa(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cc_bb(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_cc_bb(bra::Solution{RASCIAnsatz}, 
                                                 ket::Solution{RASCIAnsatz})
end


"""
    compute_operator_cc_aa(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for alpha-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cc_aa(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_cc_aa(bra::Solution{RASCIAnsatz}, 
                                                 ket::Solution{RASCIAnsatz})
end


"""
    compute_operator_cc_ab(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for alpha-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cc_ab(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_cc_ab(bra::Solution{RASCIAnsatz}, 
                                                 ket::Solution{RASCIAnsatz})
end


"""
    compute_operator_cca_aaa(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-alpha-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_aaa(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_cca_aaa(bra::Solution{RASCIAnsatz}, 
                                                 ket::Solution{RASCIAnsatz})
end


"""
    compute_operator_cca_bbb(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for beta-beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_bbb(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_cca_bbb(bra::Solution{RASCIAnsatz}, 
                                                 ket::Solution{RASCIAnsatz})
end


"""
    compute_operator_cca_aba(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-beta-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_aba(bra::Solution{RASCIAnsatz,T}, 
                                                     ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_cca_aba(bra::Solution{RASCIAnsatz}, 
                                                             ket::Solution{RASCIAnsatz})
end


"""
    compute_operator_cca_abb(bra::Solution{RASCIAnsatz,T}, ket::Solution{FCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_abb(bra::Solution{RASCIAnsatz,T}, 
                                                     ket::Solution{RASCIAnsatz,T}) where {T}
    return ActiveSpaceSolvers.RASCI.compute_operator_cca_abb(bra::Solution{RASCIAnsatz}, 
                                                 ket::Solution{RASCIAnsatz}) 
end


"""
    compute_1rdm(sol::Solution{RASCIAnsatz,T}; root=1) where {T}
"""
function ActiveSpaceSolvers.compute_1rdm(sol::Solution{RASCIAnsatz,T}; root=1) where {T}
    return ActiveSpaceSolvers.RASCI.compute_1rdm(sol.ansatz, sol.vectors[:,root])
end

"""
    compute_1rdm_2rdm(sol::Solution{A,T}; root=1) where {A,T}
"""
function ActiveSpaceSolvers.compute_1rdm_2rdm(sol::Solution{RASCIAnsatz,T}; root=1) where {T}
    return ActiveSpaceSolvers.RASCI.compute_1rdm_2rdm(sol.ansatz, sol.vectors[:,root])
end
