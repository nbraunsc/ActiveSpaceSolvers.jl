using ActiveSpaceSolvers
import LinearMaps
using OrderedCollections
using StaticArrays

"""
Type containing all the metadata needed to define a RASCI problem 

    no::Int  # number of orbitals
    na::Int  # number of alpha
    nb::Int  # number of beta
    fock::SVector{3, Int}   #fock section working in (ras1, ras2, ras3)
    ras1_min::Int       #min electrons in ras1
    ras3_max::Int       #max electrons in ras3
    dima::Int 
    dimb::Int 
    dim::Int
    converged::Bool
    restarted::Bool
    iteration::Int
    algorithm::String   #  options: direct/davidson
    n_roots::Int
"""
struct RASCIAnsatz <: Ansatz 
    no::Int  # number of orbitals
    na::Int  # number of alpha
    nb::Int  # number of beta
    fock::SVector{3, Int}   #fock section working in (ras1, ras2, ras3)
    ras1_min::Int       #min electrons in ras1
    ras3_max::Int       #max electrons in ras3
    dima::Int 
    dimb::Int 
    dim::Int
    converged::Bool
    restarted::Bool
    iteration::Int
    algorithm::String   #  options: direct/davidson
    n_roots::Int
    xalpha::Array{Int}
    xbeta::Array{Int}
end

"""
    RASCIAnsatz(no, na, nb, fock::Any, ras1_min=1, ras3_max=2)

Constructor

# Arguments
- `no`: Number of spatial orbitals
- `na`: Number of α electrons
- `nb`: Number of β electrons
- `fock`: Number of orbitals in each ras space, defining the Fock sector
"""
function RASCIAnsatz(no, na, nb, fock::Any, ras1_min=1, ras3_max=2)
    na <= no || throw(DimensionMismatch)
    nb <= no || throw(DimensionMismatch)
    fock = convert(SVector{3,Int},collect(fock))
    dima, xalpha = ras_calc_ndets(no, na, fock, ras1_min, ras3_max)
    dimb, xbeta = ras_calc_ndets(no, nb, fock, ras1_min, ras3_max)
    return RASCIAnsatz(no, na, nb, fock, ras1_min, ras3_max, dima, dimb, dima*dimb, false, false, 1, "direct", 1,xalpha, xbeta)
end

function ras_calc_ndets(no, nelec, fock, ras1_min, ras3_max)
    x = RASCI.make_ras_x(no, nelec, fock, ras1_min, ras3_max)
    dim_x = findmax(x)[1]
    return dim_x, x
end

function display(p::RASCIAnsatz)
    @printf(" RASCIAnsatz:: #Orbs = %-3i #α = %-2i #β = %-2i Dimension: %-9i\n",p.no,p.na,p.nb,p.dim)
end


"""
    LinearMap(ints, prb::RASCIAnsatz)

Get LinearMap with takes a vector and returns action of H on that vector

# Arguments
- ints: `InCoreInts` object
- prb:  `RASCIAnsatz` object
"""
function LinearMaps.LinearMap(ints::InCoreInts, prb::RASCIAnsatz)
    #={{{=#
    ket_a = DeterminantString(prb.no, prb.na)
    ket_b = DeterminantString(prb.no, prb.nb)

    #@btime lookup_a = $fill_ca_lookup2($ket_a)
    lookup_a = fill_ca_lookup2(ket_a)
    lookup_b = fill_ca_lookup2(ket_b)
    iters = 0
    function mymatvec(v)
        iters += 1
        #@printf(" Iter: %4i\n", iters)
        nr = 0
        if length(size(v)) == 1
            nr = 1
            v = reshape(v,ket_a.max*ket_b.max, nr)
        else 
            nr = size(v)[2]
        end
        v = reshape(v, ket_a.max, ket_b.max, nr)
        sig = compute_ab_terms2(v, ints, prb, lookup_a, lookup_b)
        sig += compute_ss_terms2(v, ints, prb, lookup_a, lookup_b)

        v = reshape(v, ket_a.max*ket_b.max, nr)
        sig = reshape(sig, ket_a.max*ket_b.max, nr)
        return sig 
    end
    return LinearMap(mymatvec, prb.dim, prb.dim; issymmetric=true, ismutating=false, ishermitian=true)
end
#=}}}=#


"""
    build_H_matrix(ints, P::RASCIAnsatz)

Build the Hamiltonian defined by `ints` in the Slater Determinant Basis  specified by `P`
"""
function ActiveSpaceSolvers.build_H_matrix(ints::InCoreInts{T}, P::RASCIAnsatz) where T
#={{{=#
    Hmat = zeros(T, P.dim, P.dim)

    Hdiag_a = precompute_spin_diag_terms(ints,P,P.na)
    Hdiag_b = precompute_spin_diag_terms(ints,P,P.nb)
    # 
    #   Create ci_strings
    ket_a = DeterminantString(P.no, P.na)
    ket_b = DeterminantString(P.no, P.nb)
    bra_a = DeterminantString(P.no, P.na)
    bra_b = DeterminantString(P.no, P.nb)
    #   
    #   Add spin diagonal components
    Hmat += kron(Matrix(1.0I, P.dimb, P.dimb), Hdiag_a)
    Hmat += kron(Hdiag_b, Matrix(1.0I, P.dima, P.dima))
    #
    #   Add opposite spin term (todo: make this reasonably efficient)
    Hmat += compute_ab_terms_full(ints, P, T=T)
    
    Hmat = .5*(Hmat+Hmat')

    return Hmat
end
#=}}}=#


"""
    build_S2_matrix(P::RASCIAnsatz)

Build the S2 matrix in the Slater Determinant Basis  specified by `P`
"""
function ActiveSpaceSolvers.build_S2_matrix(P::RASCIAnsatz) where T
#={{{=#
    return build_S2_matrix(P)
end
#=}}}=#


"""
    compute_operator_a_a(bra::Solution{RASCIAnsatz,T}, ket::Solution{RASCIAnsatz,T}) where {T}

Compute representation of a operator between states `bra_v` and `ket_v` for alpha
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_a_a(bra::Solution{RASCIAnsatz,T}, 
                                                 ket::Solution{RASCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_annihilation(n_orbs(bra), 
                                n_elec_a(bra), n_elec_b(bra),
                                n_elec_a(ket), n_elec_b(ket),
                                bra.vectors, ket.vectors,
                                "alpha")

    
#=}}}=#
end



"""
    compute_operator_a_b(bra::Solution{RASCIAnsatz,T}, ket::Solution{RASCIAnsatz,T}) where {T}

Compute representation of a operator between states `bra_v` and `ket_v` for beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_a_b(bra::Solution{RASCIAnsatz,T}, 
                                                 ket::Solution{RASCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_annihilation(n_orbs(bra), 
                                n_elec_a(bra), n_elec_b(bra),
                                n_elec_a(ket), n_elec_b(ket),
                                bra.vectors, ket.vectors,
                                "beta")

    
#=}}}=#
end



"""
    compute_operator_ca_aa(bra::Solution{RASCIAnsatz,T}, ket::Solution{RASCIAnsatz,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for alpha-alpha
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_ca_aa(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_Aa(n_orbs(bra), 
                      n_elec_a(bra), n_elec_b(bra),
                      n_elec_a(ket), n_elec_b(ket),
                      bra.vectors, ket.vectors,
                      "alpha")

    
#=}}}=#
end

"""
    compute_operator_ca_bb(bra::Solution{RASCIAnsatz,T}, ket::Solution{RASCIAnsatz,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_ca_bb(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_Aa(n_orbs(bra), 
                      n_elec_a(bra), n_elec_b(bra),
                      n_elec_a(ket), n_elec_b(ket),
                      bra.vectors, ket.vectors,
                      "beta")

    
#=}}}=#
end


"""
    compute_operator_ca_ab(bra::Solution{RASCIAnsatz,T}, ket::Solution{RASCIAnsatz,T}) where {T}

Compute representation of a'a operators between states `bra_v` and `ket_v` for alpha-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_ca_ab(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_Ab(n_orbs(bra), 
                      n_elec_a(bra), n_elec_b(bra),
                      n_elec_a(ket), n_elec_b(ket),
                      bra.vectors, ket.vectors)

    
#=}}}=#
end


"""
    compute_operator_cc_aa(bra::Solution{RASCIAnsatz,T}, ket::Solution{RASCIAnsatz,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cc_bb(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_AA(n_orbs(bra), 
                      n_elec_a(bra), n_elec_b(bra),
                      n_elec_a(ket), n_elec_b(ket),
                      bra.vectors, ket.vectors,
                      "beta")

    
#=}}}=#
end


"""
    compute_operator_cc_aa(bra::Solution{RASCIAnsatz,T}, ket::Solution{RASCIAnsatz,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for alpha-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cc_aa(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_AA(n_orbs(bra), 
                      n_elec_a(bra), n_elec_b(bra),
                      n_elec_a(ket), n_elec_b(ket),
                      bra.vectors, ket.vectors,
                      "alpha")

    
#=}}}=#
end


"""
    compute_operator_cc_ab(bra::Solution{RASCIAnsatz,T}, ket::Solution{RASCIAnsatz,T}) where {T}

Compute representation of a'a' operators between states `bra_v` and `ket_v` for alpha-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cc_ab(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_AB(n_orbs(bra), 
                      n_elec_a(bra), n_elec_b(bra),
                      n_elec_a(ket), n_elec_b(ket),
                      bra.vectors, ket.vectors)

    
#=}}}=#
end


"""
    compute_operator_cca_aaa(bra::Solution{RASCIAnsatz,T}, ket::Solution{RASCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-alpha-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_aaa(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_AAa(n_orbs(bra), 
                      n_elec_a(bra), n_elec_b(bra),
                      n_elec_a(ket), n_elec_b(ket),
                      bra.vectors, ket.vectors, 
                      "alpha")

    
#=}}}=#
end


"""
    compute_operator_cca_bbb(bra::Solution{RASCIAnsatz,T}, ket::Solution{RASCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for beta-beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_bbb(bra::Solution{RASCIAnsatz,T}, 
                                                   ket::Solution{RASCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_AAa(n_orbs(bra), 
                      n_elec_a(bra), n_elec_b(bra),
                      n_elec_a(ket), n_elec_b(ket),
                      bra.vectors, ket.vectors, 
                      "beta")

    
#=}}}=#
end


"""
    compute_operator_cca_aba(bra::Solution{RASCIAnsatz,T}, ket::Solution{RASCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-beta-alpha 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_aba(bra::Solution{RASCIAnsatz,T}, 
                                                     ket::Solution{RASCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_ABa(n_orbs(bra), 
                      n_elec_a(bra), n_elec_b(bra),
                      n_elec_a(ket), n_elec_b(ket),
                      bra.vectors, ket.vectors)

    
#=}}}=#
end


"""
    compute_operator_cca_abb(bra::Solution{RASCIAnsatz,T}, ket::Solution{RASCIAnsatz,T}) where {T}

Compute representation of a'a'a operators between states `bra_v` and `ket_v` for alpha-beta-beta 
# Arguments
- `bra`: solutions for the left hand side
- `ket`: solutions for the right hand side

"""
function ActiveSpaceSolvers.compute_operator_cca_abb(bra::Solution{RASCIAnsatz,T}, 
                                                     ket::Solution{RASCIAnsatz,T}) where {T}
    #={{{=#
    n_orbs(bra) == n_orbs(ket) || throw(DimensionMismatch) 
    return compute_ABb(n_orbs(bra), 
                      n_elec_a(bra), n_elec_b(bra),
                      n_elec_a(ket), n_elec_b(ket),
                      bra.vectors, ket.vectors)

    
#=}}}=#
end


"""
    compute_1rdm(sol::Solution{RASCIAnsatz,T}; root=1) where {T}

"""
function ActiveSpaceSolvers.compute_1rdm(sol::Solution{RASCIAnsatz,T}; root=1) where {T}
    #={{{=#

    rdma = compute_Aa(n_orbs(sol),
                      n_elec_a(sol), n_elec_b(sol),                     
                      n_elec_a(sol), n_elec_b(sol),                     
                      reshape(sol.vectors[:,root], dim(sol), 1), 
                      reshape(sol.vectors[:,root], dim(sol), 1), 
                      "alpha") 

    rdmb = compute_Aa(n_orbs(sol),
                      n_elec_a(sol), n_elec_b(sol),                     
                      n_elec_a(sol), n_elec_b(sol),                     
                      reshape(sol.vectors[:,root], dim(sol), 1), 
                      reshape(sol.vectors[:,root], dim(sol), 1), 
                      "beta") 


    rdma = reshape(rdma, n_orbs(sol), n_orbs(sol))
    rdmb = reshape(rdmb, n_orbs(sol), n_orbs(sol))
    return rdma, rdmb
end
#=}}}=#


"""
    compute_2rdm(sol::Solution{A,T}; root=1) where {A,T}

"""
function ActiveSpaceSolvers.compute_1rdm_2rdm(sol::Solution{RASCIAnsatz,T}; root=1) where {T}
    #={{{=#

    return compute_rdm1_rdm2(sol.ansatz, sol.vectors[:,root], sol.vectors[:,root])
end
#=}}}=#




"""
    svd_state(sol::Solution{RASCIAnsatz,T},norbs1,norbs2,svd_thresh; root=1) where T
Do an SVD of the RASCI vector partitioned into clusters with (norbs1 | norbs2)
where the orbitals are assumed to be ordered for cluster 1| cluster 2 haveing norbs1 and 
norbs2, respectively.

- `sol`: Solution just defines the current CI states 
- `norbs1`:number of orbitals in left cluster
- `norbs2`:number of orbitals in right cluster
- `svd_thresh`: the threshold below which the states will be discarded
- `root`: which root to SVD
"""
function ActiveSpaceSolvers.svd_state(sol::Solution{RASCIAnsatz,T},norbs1,norbs2,svd_thresh; root=1) where T
    #={{{=#

    @assert(norbs1+norbs2 == n_orbs(sol))

    schmidt_basis = OrderedDict()
    #vector = OrderedDict{Tuple{UInt8,UInt8},Float64}()
    vector = OrderedDict{Tuple{Int,Int},Any}()

    #schmidt_basis = Dict{Tuple,Matrix{Float64}}

    println("----------------------------------------")
    println("          SVD of state")
    println("----------------------------------------")

    # Create ci_strings
    ket_a = DeterminantString(n_orbs(sol), n_elec_a(sol))
    ket_b = DeterminantString(n_orbs(sol), n_elec_b(sol))
    
    v = sol.vectors[:,root]
    v = reshape(v,(ket_a.max, ket_b.max))
    @assert(size(v,1) == ket_a.max)
    @assert(size(v,2) == ket_b.max)

    fock_labels_a = Array{Int,1}(undef,ket_a.max)
    fock_labels_b = Array{Int,1}(undef,ket_b.max)


    # Get the fock space using the bisect method in python
    #bisect = pyimport("bisect")
    for I in 1:ket_a.max
        label = 0
        for i in 1:length(ket_a.config)
            if ket_a.config[i] <= norbs1
                label += 1
            end
        end
        fock_labels_a[I] = label
        #println(ket_a.config, " ", norbs1, " ", label)
        incr!(ket_a)
    end
    for I in 1:ket_b.max
        label = 0
        for i in 1:length(ket_b.config)
            if ket_b.config[i] <= norbs1
                label += 1
            end
        end
        fock_labels_b[I] = label
        #println(ket_b.config, " ", norbs1, " ", label)
        incr!(ket_b)
    end
    for J in 1:ket_b.max
        for I in 1:ket_a.max
            fock = (fock_labels_a[I], fock_labels_b[J])

            #if fock in vector
            #    append!(vector[fock],v[I,J])
            #else
            #    vector[fock] = [v[I,J]]
            #end
            try
                append!(vector[tuple(fock_labels_a[I],fock_labels_b[J])],v[I,J])
            catch
                vector[tuple(fock_labels_a[I],fock_labels_b[J])] = [v[I,J]]
            end
        end
    end

    for (fock,fvec)  in vector

        println()
        @printf("Prepare Fock Space:  %iα, %iβ\n",fock[1] ,fock[2] )

        ket_a1 = DeterminantString(norbs1, fock[1])
        ket_b1 = DeterminantString(norbs1, fock[2])

        ket_a2 = DeterminantString(norbs2, n_elec_a(sol) - fock[1])
        ket_b2 = DeterminantString(norbs2, n_elec_b(sol) - fock[2])


        temp_fvec = reshape(fvec,ket_b1.max*ket_b2.max,ket_a1.max*ket_a2.max)'
        #temp_fvec = reshape(fvec,ket_b1.max*ket_b2.max,ket_a1.max*ket_a2.max)'
        #st = "temp_fvec"*string(fock)
        #npzwrite(st, temp_fvec)


        #when swapping alpha2 and beta1 do we flip sign?
        sign = 1
        if (n_elec_a(sol)-fock[1])%2==1 && fock[2]%2==1
            sign = -1
        end
        #println("sign",sign)
        @printf("   Dimensions: %5i x %-5i \n",ket_a1.max*ket_b1.max, ket_a2.max*ket_b2.max)

        norm_curr = fvec' * fvec
        @printf("   Norm: %12.8f\n",sqrt(norm_curr))
        #println(size(fvec))
        #display(fvec)

        fvec = sign *fvec

        #opposite to python with transpose on fvec
        #fvec2 = reshape(fvec',ket_b2.max,ket_b1.max,ket_a2.max,ket_a1.max)
        fvec2 = reshape(fvec,ket_a1.max,ket_a2.max,ket_b1.max,ket_b2.max)
        fvec3 = permutedims(fvec2, [ 1, 3, 2, 4])
        fvec4 = reshape(fvec3,ket_a1.max*ket_b1.max,ket_a2.max*ket_b2.max)

        # fvec4 is transpose of what we have in python code
        fvec5 = fvec4'

        F = svd(fvec5,full=true)


        nkeep = 0
        @printf("   %5s %12s\n","State","Weight")
        for (ni_idx,ni) in enumerate(F.S)
            if ni > svd_thresh
                nkeep += 1
                @printf("   %5i %12.8f\n",ni_idx,ni)
            else
                @printf("   %5i %12.8f (discarded)\n",ni_idx,ni)
            end
        end
        

        if nkeep > 0
            schmidt_basis[fock] = Matrix(F.U[:,1:nkeep])
            #st = "fin_vec"*string(fock)
            #npzwrite(st, F.U[:,1:nkeep])
        end

        #norm += norm_curr
    end

    return schmidt_basis
end
#=}}}=#


