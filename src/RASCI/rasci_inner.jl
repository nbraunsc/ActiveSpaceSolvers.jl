using JLD2
using InCoreIntegrals

"""
    compute_S2_expval(prb::RASCIAnsatz)
- `prb`: RASCIAnsatz just defines the current CI ansatz (i.e., ras_spaces sector)
"""
function compute_S2_expval(C::Matrix, P::RASCIAnsatz)
    ###{{{
    #S2 = (S+S- + S-S+)1/2 + Sz.Sz
    #   = 1/2 sum_ij(ai'bi bj'ai + bj'aj ai'bi) + Sz.Sz
    #   do swaps and you can end up adding the two together to get rid
    #   of the 1/2 factor so 
    #   = (-1) sum_ij(ai'aj|alpha>bj'bi|beta> + Sz.Sz
    ###
    
    as, bs, rev_as, rev_bs, all_cats_a, all_cats_b = S2_helper(P)
    spin_pairs = ActiveSpaceSolvers.RASCI.make_spin_pairs(P, all_cats_a, all_cats_b)
    
    nr = size(C,2)
    s2 = zeros(nr)
    
    v = Dict{Int, Array{Float64, 3}}()
    
    start = 1
    for m in 1:length(spin_pairs)
        tmp = C[start:start+spin_pairs[m].dim-1, :]
        v[m] = reshape(tmp, (length(all_cats_a[spin_pairs[m].pair[1]].idxs), length(all_cats_b[spin_pairs[m].pair[2]].idxs), nr))
        start += spin_pairs[m].dim
    end
    
    for m in 1:length(spin_pairs)
        cat_Ia = all_cats_a[spin_pairs[m].pair[1]]
        cat_Ib = all_cats_b[spin_pairs[m].pair[2]]
        for Ia in all_cats_a[spin_pairs[m].pair[1]].idxs
            config_a = as[Ia]
            Ia_local = Ia-cat_Ia.shift
            for Ib in all_cats_b[spin_pairs[m].pair[2]].idxs
                config_b = bs[Ib]
                Ib_local = Ib-cat_Ib.shift

                #Sz.Sz (α) 
                count_a = (P.na-1)*P.na
                for i in 1:count_a
                    for r in 1:nr
                        s2[r] += 0.25*v[m][Ia_local, Ib_local, r]*v[m][Ia_local, Ib_local, r]
                    end
                end
                
                #Sz.Sz (β)
                count_b = (P.nb-1)*P.nb
                for i in 1:count_b
                    for r in 1:nr
                        s2[r] += 0.25*v[m][Ia_local, Ib_local, r]*v[m][Ia_local, Ib_local, r]
                    end
                end

                #Sz.Sz (α,β)
                for ai in config_a
                    for bj in config_b
                        if ai != bj
                            for r in 1:nr
                                s2[r] -= .5 * v[m][Ia_local, Ib_local, r]*v[m][Ia_local, Ib_local, r] 
                            end
                        end
                    end
                end

                ##Sp.Sm + Sm.Sp Diagonal Part
                for ai in config_a
                    if ai in config_b
                    else
                        for r in 1:nr
                            s2[r] += .75 * v[m][Ia_local, Ib_local, r]*v[m][Ia_local, Ib_local, r] 
                        end
                    end
                end

                for bi in config_b
                    if bi in config_a
                    else
                        for r in 1:nr
                            s2[r] += .75 * v[m][Ia_local, Ib_local, r]*v[m][Ia_local, Ib_local, r] 
                        end
                    end
                end
                
                #(Sp.Sm + Sm.Sp)1/2 Off Diagonal Part
                for ai in config_a
                    for bj in config_b
                        if ai ∉ config_b
                            if bj ∉ config_a
                                #Sp.Sm + Sm.Sp
                                La = cat_Ia.lookup[ai,bj,Ia_local]
                                La != 0 || continue
                                sign_a = sign(La)
                                La = abs(La)
                                cat_La = find_cat(La, all_cats_a)
                                Lb = cat_Ib.lookup[bj,ai,Ib_local]
                                Lb != 0 || continue
                                sign_b = sign(Lb)
                                Lb = abs(Lb)
                                cat_Lb = find_cat(Lb, all_cats_b)
                                n = find_spin_pair(spin_pairs, (cat_La.idx, cat_Lb.idx))
                                n != 0 || continue
                                La_local = La-cat_La.shift
                                Lb_local = Lb-cat_Lb.shift
                                for r in 1:nr
                                    s2[r] -= sign_a*sign_b*v[m][Ia_local, Ib_local,r]*v[n][La_local, Lb_local, r]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return s2#=}}}=#
end

"""
    apply_S2_matrix(P::RASCIAnsatz, C::AbstractArray{T}) where {T}
- `P`: RASCIAnsatz just defines the current CI ansatz (i.e., ras_spaces sector)
"""
function apply_S2_matrix(P::RASCIAnsatz, C::AbstractArray{T}) where T
    #S2 = (S+S- + S-S+)1/2 + Sz.Sz{{{
    #   = 1/2 sum_ij(ai'bi bj'ai + bj'aj ai'bi) + Sz.Sz
    #   do swaps and you can end up adding the two together to get rid
    #   of the 1/2 factor so 
    #   = (-1) sum_ij(ai'aj|alpha>bj'bi|beta> + Sz.Sz
    ###
    
    as, bs, rev_as, rev_bs, all_cats_a, all_cats_b = S2_helper(P)
    spin_pairs = ActiveSpaceSolvers.RASCI.make_spin_pairs(P, all_cats_a, all_cats_b)
    
    P.dim == size(C,1) || throw(DimensionMismatch)
    
    v = Dict{Int, Array{Float64, 3}}()
    S2v = Dict{Int, Array{Float64, 3}}()
    
    nr = size(C, 2)
    start = 1
    for m in 1:length(spin_pairs)
        S2v[m] = zeros(length(all_cats_a[spin_pairs[m].pair[1]].idxs), length(all_cats_b[spin_pairs[m].pair[2]].idxs), nr)
        tmp = C[start:start+spin_pairs[m].dim-1, :]
        v[m] = reshape(tmp, (length(all_cats_a[spin_pairs[m].pair[1]].idxs), length(all_cats_b[spin_pairs[m].pair[2]].idxs), nr))
        start += spin_pairs[m].dim
    end
    
    for m in 1:length(spin_pairs)
        cat_Ia = all_cats_a[spin_pairs[m].pair[1]]
        cat_Ib = all_cats_b[spin_pairs[m].pair[2]]
        for Ia in all_cats_a[spin_pairs[m].pair[1]].idxs
            config_a = as[Ia]
            Ia_local = Ia-cat_Ia.shift
            for Ib in all_cats_b[spin_pairs[m].pair[2]].idxs
                config_b = bs[Ib]
                Ib_local = Ib-cat_Ib.shift

                #Sz.Sz (α) 
                count_a = (P.na-1)*P.na
                for i in 1:count_a
                    S2v[m][Ia_local, Ib_local, :] .+= 0.25.*v[m][Ia_local, Ib_local, :]
                end
                
                #Sz.Sz (β)
                count_b = (P.nb-1)*P.nb
                for i in 1:count_b
                    S2v[m][Ia_local, Ib_local, :] .+= 0.25.*v[m][Ia_local, Ib_local, :]
                end

                #Sz.Sz (α,β)
                for ai in config_a
                    for bj in config_b
                        if ai != bj
                            S2v[m][Ia_local, Ib_local, :] .-= 0.5.*v[m][Ia_local, Ib_local, :]
                        end
                    end
                end

                ##Sp.Sm + Sm.Sp Diagonal Part
                for ai in config_a
                    if ai in config_b
                    else
                        S2v[m][Ia_local, Ib_local, :] .+= 0.75.*v[m][Ia_local, Ib_local, :]
                    end
                end

                for bi in config_b
                    if bi in config_a
                    else
                        S2v[m][Ia_local, Ib_local, :] .+= 0.75.*v[m][Ia_local, Ib_local, :]
                    end
                end
                
                #(Sp.Sm + Sm.Sp)1/2 Off Diagonal Part
                for ai in config_a
                    for bj in config_b
                        if ai ∉ config_b
                            if bj ∉ config_a
                                #Sp.Sm + Sm.Sp
                                La = cat_Ia.lookup[ai,bj,Ia_local]
                                La != 0 || continue
                                sign_a = sign(La)
                                La = abs(La)
                                cat_La = find_cat(La, all_cats_a)
                                Lb = cat_Ib.lookup[bj,ai,Ib_local]
                                Lb != 0 || continue
                                sign_b = sign(Lb)
                                Lb = abs(Lb)
                                cat_Lb = find_cat(Lb, all_cats_b)
                                n = find_spin_pair(spin_pairs, (cat_La.idx, cat_Lb.idx))
                                n != 0 || continue
                                La_local = La-cat_La.shift
                                Lb_local = Lb-cat_Lb.shift
                                S2v[m][Ia_local, Ib_local, :] .-= sign_a*sign_b*v[n][La_local, Lb_local,:]
                            end
                        end
                    end
                end
            end
        end
    end
    
    starti = 1
    S2 = zeros(Float64, P.dim, nr)
    for m in 1:length(spin_pairs)
        tmp = reshape(S2v[m], (size(S2v[m],1)*size(S2v[m],2), nr))
        S2[starti:starti+spin_pairs[m].dim-1, :] .= tmp
        starti += spin_pairs[m].dim
    end
    
    return S2#=}}}=#
end

"""
    S2_helper(P::RASCIAnsatz)

A helper function for apply_S2_matrix
"""
function S2_helper(P::RASCIAnsatz)
    categories = ActiveSpaceSolvers.RASCI.generate_spin_categories(P)#={{{=#
    all_cats_a = Vector{HP_Category_CA}()
    all_cats_b = Vector{HP_Category_CA}()
    all_cats_bra_a = Vector{ActiveSpaceSolvers.RASCI.HP_Category_Bra}()
    all_cats_bra_b = Vector{ActiveSpaceSolvers.RASCI.HP_Category_Bra}()
    
    cats_a = deepcopy(categories)
    cats_b = deepcopy(categories)
    fock_list_a, del_at_a = make_fock_from_categories(categories, P, "alpha")
    deleteat!(cats_a, del_at_a)
    len_cat_a = length(cats_a)
        
    fock_list_b, del_at_b = make_fock_from_categories(categories, P, "beta")
    deleteat!(cats_b, del_at_b)
    len_cat_b = length(cats_b)

    #alpha
    connected_a = make_spincategory_connections(cats_a, cats_b, P)

    #compute configs
    as = compute_config_dict(fock_list_a, P, "alpha")
    rev_as = Dict(value => key for (key, value) in as)
    #this reverses the config dictionary to get the index as the key 

    shift = 0
    for j in 1:len_cat_a
        idxas = Vector{Int}()
        graph_a = make_cat_graphs(fock_list_a[j], P)
        idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
        sort!(idxas)
        lu = zeros(Int, graph_a.no, graph_a.no, length(idxas))
        cat_lu = zeros(Int, graph_a.no, graph_a.no, length(idxas))
        push!(all_cats_a, HP_Category_CA(j, cats_a[j], connected_a[j], idxas, shift, lu, cat_lu))
        push!(all_cats_bra_a, HP_Category_Bra(j, connected_a[j], idxas, shift))
        shift += length(idxas)
    end

    for k in 1:len_cat_a
        graph_a = make_cat_graphs(fock_list_a[k], P)
        lu = ActiveSpaceSolvers.RASCI.dfs_ca(graph_a, 1, graph_a.max, all_cats_a[k].lookup, all_cats_a, all_cats_bra_a, rev_as, rev_as)
        all_cats_a[k].lookup .= lu
    end

    #beta
    connected_b = make_spincategory_connections(cats_b, cats_a, P)
    #compute configs
    bs = compute_config_dict(fock_list_b, P, "beta")
    rev_bs = Dict(value => key for (key, value) in bs)
    
    shiftb = 0
    for j in 1:len_cat_b
        idxbs = Vector{Int}()
        graph_b = make_cat_graphs(fock_list_b[j], P)
        idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
        sort!(idxbs)
        lu = zeros(Int, graph_b.no, graph_b.no, length(idxbs))
        cat_lu = zeros(Int, graph_b.no, graph_b.no, length(idxbs))
        push!(all_cats_b, HP_Category_CA(j, cats_b[j], connected_b[j], idxbs, shiftb, lu, cat_lu))
        push!(all_cats_bra_b, HP_Category_Bra(j, connected_b[j], idxbs, shiftb))
        shiftb += length(idxbs)
    end
    
    for k in 1:len_cat_b
        graph_b = make_cat_graphs(fock_list_b[k], P)
        lu = ActiveSpaceSolvers.RASCI.dfs_ca(graph_b, 1, graph_b.max, all_cats_b[k].lookup, all_cats_b, all_cats_bra_b, rev_bs, rev_bs)
        all_cats_b[k].lookup .= lu
    end
#=}}}=#
    return as, bs, rev_as, rev_bs, all_cats_a, all_cats_b
end

"""
    sigma_one(prob::RASCIAnsatz, spin_pairs::Vector{Spin_Pair}, cats_a::Vector{HP_Category_CA}, cats_b::Vector{HP_Category_CA}, ints::InCoreInts, C)

Compute only beta spin contributions to sigma, σ = H|v>
"""
function sigma_one(prob::RASCIAnsatz, spin_pairs::Vector{Spin_Pair}, cats_a::Vector{HP_Category_CA}, cats_b::Vector{HP_Category_CA}, ints::InCoreInts, C)
    sigma_one = Dict{Int, Array{Float64,3}}()#={{{=#
    v = Dict{Int, Array{Float64, 3}}()
    n_spin_pairs = length(spin_pairs)
    nroots = size(C,2)

    F = zeros(Float64, prob.dimb)
    start = 1

    #sign to switch from (a,b) to (b,a) for optimizing _sum_spin_pairs! function
    sgnK = 1 
    if (prob.na) % 2 != 0 
        sgnK = -sgnK
    end
    
    for m in 1:n_spin_pairs
        #Beta then alpha to speed things up
        sigma_one[m] = zeros(length(cats_b[spin_pairs[m].pair[2]].idxs), length(cats_a[spin_pairs[m].pair[1]].idxs), nroots)
        #Alpha then Beta
        #sigma_one[m] = zeros(length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs), nroots)
        tmp = C[start:start+spin_pairs[m].dim-1, :]
        v[m] = reshape(tmp, (length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs), nroots))
        
        v[m] = sgnK.*permutedims(v[m], (2,1,3))
        start += spin_pairs[m].dim
    end
    
    gkl = get_gkl(ints, prob) 
    for Ib in 1:prob.dimb
        fill!(F,0.0)
        comb_kl = 0
        comb_ij = 0
        cat_Ib = find_cat(Ib, cats_b)
        #pair_Ib = find_spin_pair(spin_pairs, cat_Ib.idx, "beta")
        #Ib_local = Ib-spin_pairs[pair_Ib].bshift
        Ib_local = Ib-cat_Ib.shift
        for k in 1:prob.no, l in 1:prob.no
            Kb = cat_Ib.lookup[l,k,Ib_local]
            Kb != 0 || continue
            sign_kl = sign(Kb)
            Kb = abs(Kb)
            @inbounds F[Kb] += sign_kl*gkl[k,l]
            catb_kl = find_cat(Kb, cats_b)
            comb_kl = (k-1)*prob.no + l
            #pair_Kb = find_spin_pair(spin_pairs, catb_kl.idx, "beta") 
            #Kb_local = Kb-spin_pairs[pair_Kb].bshift
            Kb_local = Kb-catb_kl.shift

            for i in 1:prob.no, j in 1:prob.no
                comb_ij = (i-1)*prob.no + j
                if comb_ij < comb_kl
                    continue
                end
                Jb = catb_kl.lookup[j,i,Kb_local]
                Jb != 0 || continue
                sign_ij = sign(Jb)
                Jb = abs(Jb)
                if comb_kl == comb_ij
                    delta = 1
                else
                    delta = 0
                end
                if sign_kl == sign_ij
                    F[Jb] += (ints.h2[i,j,k,l]*1/(1+delta))
                else
                    F[Jb] -= (ints.h2[i,j,k,l]*1/(1+delta))
                end
            end
        end
        _sum_spin_pairs!(sigma_one, v, F, Ib, cats_a, cats_b, spin_pairs)
    end
    
    start = 1
    sig = zeros(Float64, prob.dim, nroots)
    for m in 1:n_spin_pairs
        #Alpha then beta
        tmp = reshape(sgnK.*permutedims(sigma_one[m], (2,1,3)), (size(sigma_one[m],1)*size(sigma_one[m],2), nroots))
        #tmp = reshape(sigma_one[m], (size(sigma_one[m],1)*size(sigma_one[m],2), nroots))
        sig[start:start+spin_pairs[m].dim-1, :] .= tmp
        start += spin_pairs[m].dim
    end
    return sig#=}}}=#
end

"""
    sigma_two(prob::RASCIAnsatz, spin_pairs::Vector{Spin_Pair}, cats_a::Vector{HP_Category_CA}, cats_b::Vector{HP_Category_CA}, ints::InCoreInts, C)

Compute only alpha spin contributions to sigma, σ = H|v>
"""
function sigma_two(prob::RASCIAnsatz, spin_pairs::Vector{Spin_Pair}, cats_a::Vector{HP_Category_CA}, cats_b::Vector{HP_Category_CA}, ints::InCoreInts, C)
    sigma_two = Dict{Int, Array{Float64,3}}()#={{{=#
    v = Dict{Int, Array{Float64, 3}}()
    n_spin_pairs = length(spin_pairs)
    nroots = size(C,2)
    F = zeros(Float64, prob.dima)
    start = 1

    for m in 1:n_spin_pairs
        sigma_two[m] = zeros(length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs), nroots)
        tmp = C[start:start+spin_pairs[m].dim-1, :]
        v[m] = reshape(tmp, (length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs), nroots))
        start += spin_pairs[m].dim
    end

    gkl = get_gkl(ints, prob) 
    
    for Ia in 1:prob.dima
        fill!(F,0.0)
        comb_kl = 0
        comb_ij = 0
        cat_Ia = find_cat(Ia, cats_a)
        #pair_Ia = find_spin_pair(spin_pairs, cat_Ia.idx, "alpha")
        #Ia_local = Ia-spin_pairs[pair_Ia].ashift
        Ia_local = Ia-cat_Ia.shift
        for k in 1:prob.no, l in 1:prob.no
            Ka = cat_Ia.lookup[l,k,Ia_local]
            Ka != 0 || continue
            sign_kl = sign(Ka)
            Ka = abs(Ka)
            @inbounds F[Ka] += sign_kl*gkl[k,l]
            cata_kl = find_cat(Ka, cats_a)
            comb_kl = (k-1)*prob.no + l
            Ka_local = Ka-cata_kl.shift

            for i in 1:prob.no, j in 1:prob.no
                comb_ij = (i-1)*prob.no + j
                if comb_ij < comb_kl
                    continue
                end
                Ja = cata_kl.lookup[j,i,Ka_local]
                Ja != 0 || continue
                sign_ij = sign(Ja)
                Ja = abs(Ja)
                if comb_kl == comb_ij
                    delta = 1
                else
                    delta = 0
                end
                if sign_kl == sign_ij
                    F[Ja] += (ints.h2[i,j,k,l]*1/(1+delta))
                else
                    F[Ja] -= (ints.h2[i,j,k,l]*1/(1+delta))
                end
            end
        end
        _sum_spin_pairs!(sigma_two, v, F, Ia, cats_a, cats_b, spin_pairs, sigma="two")
    end

    starti = 1
    sig = zeros(Float64, prob.dim, nroots)
    for m in 1:n_spin_pairs
        tmp = reshape(sigma_two[m], (size(sigma_two[m],1)*size(sigma_two[m],2), nroots))
        sig[starti:starti+spin_pairs[m].dim-1, :] .= tmp
        starti += spin_pairs[m].dim
    end
    return sig#=}}}=#
end

"""
    sigma_three(prob::RASCIAnsatz, spin_pairs::Vector{Spin_Pair}, cats_a::Vector{HP_Category_CA}, cats_b::Vector{HP_Category_CA}, ints::InCoreInts, C)

Compute both alpha and beta single excitation contributions to sigma, σ = H|v>
"""
function sigma_three(prob::RASCIAnsatz, spin_pairs::Vector{Spin_Pair}, cats_a::Vector{HP_Category_CA}, cats_b::Vector{HP_Category_CA}, ints::InCoreInts, C)
    sigma_three = Dict{Int, Array{Float64,3}}()#={{{=#
    v = Dict{Int, Array{Float64, 3}}()
    n_spin_pairs = length(spin_pairs)
    nroots = size(C,2)
    
    hkl = zeros(Float64, prob.no, prob.no)
   
    start = 1
    for m in 1:n_spin_pairs
        sigma_three[m] = zeros(length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs), nroots)
        tmp = C[start:start+spin_pairs[m].dim-1, :]
        v[m] = reshape(tmp, (length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs), nroots))
        start += spin_pairs[m].dim
    end

    #need to make function to make ras orbs
    ras1, ras2, ras3 = make_rasorbs(prob.ras_spaces[1], prob.ras_spaces[2], prob.ras_spaces[3], prob.no)
    
    for m in 1:length(spin_pairs)
        cat_Ia = cats_a[spin_pairs[m].pair[1]]
        cat_Ib = cats_b[spin_pairs[m].pair[2]]
        for Ia in cats_a[spin_pairs[m].pair[1]].idxs
            Ia_local = Ia-cat_Ia.shift
            for k in 1:prob.no, l in 1:prob.no
                Ja = cat_Ia.lookup[l,k,Ia_local]
                Ja != 0 || continue
                sign_kl = sign(Ja)
                Ja = abs(Ja)
                hkl .= ints.h2[:,:,k,l]
                cata_kl = cats_a[cat_Ia.cat_lookup[l,k,Ia_local]]
                Ja_local = Ja-cata_kl.shift
                Ibs = cats_b[spin_pairs[m].pair[2]].idxs
                for Ib in Ibs
                #for Ib in cats_b[spin_pairs[m].pair[2]].idxs
                    Ib_local = Ib-cat_Ib.shift
                    @views sig = sigma_three[m][Ia_local, Ib_local, :]
                    # RAS1 -> RAS1, RAS2 -> RAS2, and RAS3 -> RAS3
                    sp = find_spin_pair(spin_pairs, (cata_kl.idx, cat_Ib.idx))
                    if sp != 0
                        contract!(nroots, ras1, ras1, cat_Ib, cat_Ib, Ib_local, Ja_local, sp, hkl, sign_kl, v, sig)
                        contract!(nroots, ras2, ras2, cat_Ib, cat_Ib, Ib_local, Ja_local, sp, hkl, sign_kl, v, sig)
                        contract!(nroots, ras3, ras3, cat_Ib, cat_Ib, Ib_local, Ja_local, sp, hkl, sign_kl, v, sig)
                    end

                    # RAS1 -> RAS2
                    tmp_ij = (cat_Ib.hp[1] + 1, cat_Ib.hp[2])
                    catb_ij = find_cat(tmp_ij, cats_b)
                    if catb_ij != 0
                        sp = find_spin_pair(spin_pairs, (cata_kl.idx, catb_ij.idx))
                        if sp != 0
                            contract!(nroots, ras1, ras2, cat_Ib, catb_ij, Ib_local, Ja_local, sp, hkl, sign_kl, v, sig)
                        end
                    end

                    # RAS1 -> RAS3
                    tmp_ij = (cat_Ib.hp[1] + 1, cat_Ib.hp[2]+1)
                    catb_ij = find_cat(tmp_ij, cats_b)
                    if catb_ij != 0
                        sp = find_spin_pair(spin_pairs, (cata_kl.idx, catb_ij.idx))
                        if sp != 0
                            contract!(nroots, ras1, ras3, cat_Ib, catb_ij, Ib_local, Ja_local, sp, hkl, sign_kl, v, sig)
                        end
                    end

                    # RAS2 -> RAS3
                    tmp_ij = (cat_Ib.hp[1], cat_Ib.hp[2]+1)
                    catb_ij = find_cat(tmp_ij, cats_b)
                    if catb_ij != 0
                        sp = find_spin_pair(spin_pairs, (cata_kl.idx, catb_ij.idx))
                        if sp != 0
                            contract!(nroots, ras2, ras3, cat_Ib, catb_ij, Ib_local, Ja_local, sp, hkl, sign_kl, v, sig)
                        end   
                    end
                   
                    # RAS2 -> RAS1
                    tmp_ij = (cat_Ib.hp[1] - 1, cat_Ib.hp[2])
                    catb_ij = find_cat(tmp_ij, cats_b)
                    if catb_ij != 0
                        sp = find_spin_pair(spin_pairs, (cata_kl.idx, catb_ij.idx))
                        if sp != 0
                            contract!(nroots, ras2, ras1, cat_Ib, catb_ij, Ib_local, Ja_local, sp, hkl, sign_kl, v, sig)
                        end
                    end

                    # RAS3 -> RAS1
                    tmp_ij = (cat_Ib.hp[1] - 1, cat_Ib.hp[2]-1)
                    catb_ij = find_cat(tmp_ij, cats_b)
                    if catb_ij != 0
                        sp = find_spin_pair(spin_pairs, (cata_kl.idx, catb_ij.idx))
                        if sp != 0
                            contract!(nroots, ras3, ras1, cat_Ib, catb_ij, Ib_local, Ja_local, sp, hkl, sign_kl, v, sig)
                        end
                    end

                    # RAS3 -> RAS2
                    tmp_ij = (cat_Ib.hp[1], cat_Ib.hp[2]-1)
                    catb_ij = find_cat(tmp_ij, cats_b)
                    if catb_ij != 0
                        sp = find_spin_pair(spin_pairs, (cata_kl.idx, catb_ij.idx))
                        if sp != 0
                            contract!(nroots, ras3, ras2, cat_Ib, catb_ij, Ib_local, Ja_local, sp, hkl, sign_kl, v, sig)
                        end   
                    end
                end
            end
        end
    end
    

    start = 1
    sig = zeros(Float64, prob.dim, nroots)
    for m in 1:n_spin_pairs
        tmp = reshape(sigma_three[m], (size(sigma_three[m],1)*size(sigma_three[m],2), nroots))
        sig[start:start+spin_pairs[m].dim-1, :] .= tmp
        start += spin_pairs[m].dim
    end
    return sig#=}}}=#
end

"""
    _sum_spin_pairs!(sig::Dict{Int, Array{T, 3}}, v::Dict{Int,Array{T,3}}, F::Vector{T}, I::Int, cats_a::Vector{HP_Category_CA}, cats_b::Vector{HP_Category_CA}, spin_pairs::Vector{Spin_Pair}; sigma="one") where {T}

Speeds up sigma_one and sigma_two contractions
"""
function _sum_spin_pairs!(sig::Dict{Int, Array{T, 3}}, v::Dict{Int,Array{T,3}}, F::Vector{T}, I::Int, cats_a::Vector{HP_Category_CA}, cats_b::Vector{HP_Category_CA}, spin_pairs::Vector{Spin_Pair}; sigma="one") where {T}
    n_roots = size(v[1],3)#={{{=#

    if sigma == "one"
        current_cat = find_cat(I, cats_b)
        Ib_local = I-current_cat.shift
        for catb in cats_b    
            for cats in catb.connected
                m = find_spin_pair(spin_pairs, (cats_a[cats].idx, current_cat.idx))
                if m != 0
                    @views tmp_sig = sig[m][Ib_local,:,:]
                end

                for Ia in cats_a[cats].idxs
                    n = find_spin_pair(spin_pairs, (cats_a[cats].idx, catb.idx))
                    Ia_v_local = Ia-cats_a[cats].shift
                    for Jb in catb.idxs
                        Jb_local = Jb-catb.shift
                        @views tmp = v[n][Jb_local, Ia_v_local, :]
                        #@views tmp = v[n][Ia_v_local, Jb_local, :]
                        if cats_a[cats].idx in current_cat.connected
                            Ia_local = Ia-cats_a[cats].shift
                            @inbounds @simd for si in 1:n_roots
                                tmp_sig[Ia_local,si] += F[Jb]*tmp[si]
                                #sig[m][Ia_local,Ib_local,si] += F[Jb]*v[n][Ia_v_local,Jb_local, si]
                            end
                        end
                    end
                end
            end
        end

        # Alpha first then beta (slower version){{{
        #for catb in cats_b    
        #    for cats in catb.connected
        #        m = find_spin_pair(spin_pairs, (cats_a[cats].idx, current_cat.idx))
        #        if m != 0
        #            @views tmp_sig = sig[m][:, Ib_local, :]
        #        end

        #        for Ia in cats_a[cats].idxs
        #            n = find_spin_pair(spin_pairs, (cats_a[cats].idx, catb.idx))
        #            Ia_v_local = Ia-cats_a[cats].shift
        #            for Jb in catb.idxs
        #                Jb_local = Jb-catb.shift
        #                @views tmp = v[n][Ia_v_local, Jb_local, :]
        #                if cats_a[cats].idx in current_cat.connected
        #                    Ia_local = Ia-cats_a[cats].shift
        #                    @inbounds @simd for si in 1:n_roots
        #                        tmp_sig[Ia_local,si] += F[Jb]*tmp[si]
        #                        #sig[m][Ia_local,Ib_local,si] += F[Jb]*v[n][Ia_v_local,Jb_local, si]
        #                    end
        #                end
        #            end
        #        end
        #    end
        #end}}}
    else
        current_cat = find_cat(I, cats_a)
        Ia_local = I-current_cat.shift
        for cata in cats_a
            for cats in cata.connected
                m = find_spin_pair(spin_pairs, (current_cat.idx, cats_b[cats].idx))
                if m != 0
                    @views tmp_sig = sig[m][Ia_local,:, :]
                end

                for Ib in cats_b[cats].idxs
                    n = find_spin_pair(spin_pairs, (cata.idx, cats_b[cats].idx))
                    Ib_v_local = Ib-cats_b[cats].shift
                    for Ja in cata.idxs
                        Ja_local = Ja-cata.shift
                        @views tmp = v[n][Ja_local, Ib_v_local, :]
                        if cats_b[cats].idx in current_cat.connected
                            Ib_local = Ib-cats_b[cats].shift
                            @inbounds @simd for si in 1:n_roots
                                tmp_sig[Ib_local,si] += F[Ja]*tmp[si]
                                #sig[m][Ia_local,Ib_local,si] += F[Ja]*v[n][Ja_local, Ib_v_local,si]
                            end
                        end
                    end
                end
            end
        end
    end#=}}}=#
end

"""
    compute_1rdm(prob::RASCIAnsatz, C::Vector)

Computes the 1-particle reduced density matrix for each spin, <ψ|p'q|ψ>
"""
function compute_1rdm(prob::RASCIAnsatz, C::Vector)
    spin_pairs, cats_a, cats_b = ActiveSpaceSolvers.RASCI.make_spin_pairs(prob)#={{{=#
    rdm1a = zeros(prob.no, prob.no)
    rdm1b = zeros(prob.no, prob.no)
    v = Dict{Int, Array{Float64, 2}}()
    
    start = 1
    for m in 1:length(spin_pairs)
        tmp = C[start:start+spin_pairs[m].dim-1]
        v[m] = reshape(tmp, (length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs)))
        start += spin_pairs[m].dim
    end

    ##alpha l'k
    for m in 1:length(spin_pairs)
        cat_Ia = cats_a[spin_pairs[m].pair[1]]
        for Ia in cats_a[spin_pairs[m].pair[1]].idxs
            Ia_local = Ia-cat_Ia.shift
            for k in 1:prob.no, l in 1:prob.no
                Ja = cat_Ia.lookup[l,k,Ia_local]
                Ja != 0 || continue
                sign_kl = sign(Ja)
                Ja = abs(Ja)
                cata_kl = find_cat(Ja, cats_a)
                n = find_spin_pair(spin_pairs, (cata_kl.idx, spin_pairs[m].pair[2]))
                n != 0 || continue
                Ja_local = Ja-cata_kl.shift
                rdm1a[l,k] += sign_kl*dot(v[n][Ja_local,:], v[m][Ia_local,:])
            end
        end
    end
    
    ##beta l'k
    for m in 1:length(spin_pairs)
        cat_Ib = cats_b[spin_pairs[m].pair[2]]
        for Ib in cats_b[spin_pairs[m].pair[2]].idxs
            Ib_local = Ib-cat_Ib.shift
            for k in 1:prob.no, l in 1:prob.no
                Jb = cat_Ib.lookup[l,k,Ib_local]
                Jb != 0 || continue
                sign_kl = sign(Jb)
                Jb = abs(Jb)
                catb_kl = find_cat(Jb, cats_b)
                n = find_spin_pair(spin_pairs, (spin_pairs[m].pair[1], catb_kl.idx))
                n != 0 || continue
                Jb_local = Jb-catb_kl.shift
                rdm1b[l,k] += sign_kl*dot(v[n][:, Jb_local], v[m][:, Ib_local])
            end
        end
    end

    return rdm1a, rdm1b#=}}}=#
end

"""
    compute_1rdm_2rdm(prob::RASCIAnsatz, C::Vector)

Computes both the 1-particle and 2-particle reduced density matrices, <ψ|p'q'sr|ψ>
"""
function compute_1rdm_2rdm_old(prob::RASCIAnsatz, C::Vector)
    println("in older rdm222")
    cats_a_bra, cats_a = ActiveSpaceSolvers.RASCI.fill_lu_HP(prob, prob, spin="alpha", type="ccaa")#={{{=#
    cats_b_bra, cats_b = ActiveSpaceSolvers.RASCI.fill_lu_HP(prob, prob, spin="beta", type="ccaa")
    spin_pairs = ActiveSpaceSolvers.RASCI.make_spin_pairs(prob, cats_a, cats_b)
    spin_pairs_bra = ActiveSpaceSolvers.RASCI.make_spin_pairs(prob, cats_a_bra, cats_b_bra)

    spin_pairs_ca, cats_a_ca, cats_b_ca = ActiveSpaceSolvers.RASCI.make_spin_pairs(prob)

    rdm1a, rdm1b = compute_1rdm(prob, C)
    rdm2aa = zeros(prob.no, prob.no, prob.no, prob.no)
    rdm2bb = zeros(prob.no, prob.no, prob.no, prob.no)
    rdm2ab = zeros(prob.no, prob.no, prob.no, prob.no)
    
    v = Dict{Int, Array{Float64, 2}}()
    
    start = 1
    for m in 1:length(spin_pairs)
        tmp = C[start:start+spin_pairs[m].dim-1]
        v[m] = reshape(tmp, (length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs)))
        start += spin_pairs[m].dim
    end

    ## alpha alpha p'r'sq
    for m in 1:length(spin_pairs)
        cat_Ia = cats_a[spin_pairs[m].pair[1]]
        for Ia in cats_a[spin_pairs[m].pair[1]].idxs
            Ia_local = Ia-cat_Ia.shift
            for p in 1:prob.no, q in 1:prob.no, r in 1:prob.no, s in 1:prob.no
                #lookup[a,aa,c,cc,idx]
                Ja = cat_Ia.lookup[q,s,r,p,Ia_local]
                Ja != 0 || continue
                sign_kl = sign(Ja)
                Ja = abs(Ja)
                cata_kl = find_cat(Ja, cats_a_bra)
                n = find_spin_pair(spin_pairs_bra, (cata_kl.idx, spin_pairs[m].pair[2]))
                n != 0 || continue
                Ja_local = Ja-cata_kl.shift
                rdm2aa[p,q,r,s] += sign_kl*dot(v[n][Ja_local,:], v[m][Ia_local,:])
            end
        end
    end
    
    ## beta beta p'r'sq
    for m in 1:length(spin_pairs)
        cat_Ib = cats_b[spin_pairs[m].pair[2]]
        for Ib in cats_b[spin_pairs[m].pair[2]].idxs
            Ib_local = Ib-cat_Ib.shift
            for p in 1:prob.no, q in 1:prob.no, r in 1:prob.no, s in 1:prob.no
                Jb = cat_Ib.lookup[q,s,r,p,Ib_local]
                Jb != 0 || continue
                sign_kl = sign(Jb)
                Jb = abs(Jb)
                catb_kl = find_cat(Jb, cats_b_bra)
                n = find_spin_pair(spin_pairs_bra, (spin_pairs[m].pair[1], catb_kl.idx))
                n != 0 || continue
                Jb_local = Jb-catb_kl.shift
                rdm2bb[p,q,r,s] += sign_kl*dot(v[n][:, Jb_local], v[m][:, Ib_local])
            end
        end
    end
    

    #alpha beta  p'r'sq
    for m in 1:length(spin_pairs_ca)
        cat_Ia = cats_a_ca[spin_pairs_ca[m].pair[1]]
        cat_Ib = cats_b_ca[spin_pairs_ca[m].pair[2]]
        for Ia in cats_a_ca[spin_pairs_ca[m].pair[1]].idxs
            Ia_local = Ia-cat_Ia.shift
            for q in 1:prob.no, p in 1:prob.no
                Ja = cat_Ia.lookup[q,p,Ia_local]
                Ja != 0 || continue
                sign_pq = sign(Ja)
                Ja = abs(Ja)
                cata_pq = find_cat(Ja, cats_a_ca)
                for Ib in cats_b_ca[spin_pairs[m].pair[2]].idxs
                    Ib_local = Ib-cat_Ib.shift
                    for s in 1:prob.no, r in 1:prob.no
                        Jb = cat_Ib.lookup[s,r,Ib_local]
                        Jb != 0 || continue
                        sign_rs = sign(Jb)
                        Jb = abs(Jb)
                        catb_sr = find_cat(Jb, cats_b_ca)
                        n = find_spin_pair(spin_pairs_ca, (cata_pq.idx, catb_sr.idx))
                        n != 0 || continue
                        Ja_local = Ja-cata_pq.shift
                        Jb_local = Jb-catb_sr.shift
                        rdm2ab[p,q,r,s] += sign_pq*sign_rs*v[n][Ja_local, Jb_local]*v[m][Ia_local, Ib_local]
                    end
                end
            end
        end
    end
    return rdm1a, rdm1b, rdm2aa, rdm2bb, rdm2ab#=}}}=#
end

"""
    compute_1rdm_2rdm(prob::RASCIAnsatz, C::Vector)

Computes both the 1-particle and 2-particle reduced density matrices, <ψ|p'q'sr|ψ>
"""
function compute_1rdm_2rdm(prob::RASCIAnsatz, C::Vector)
    #println("in newer rdm2")
    spin_pairs, cats_a, cats_b = ActiveSpaceSolvers.RASCI.make_spin_pairs(prob)

    rdm1a, rdm1b = compute_1rdm(prob, C)
    rdm2aa = zeros(prob.no, prob.no, prob.no, prob.no)
    rdm2bb = zeros(prob.no, prob.no, prob.no, prob.no)
    rdm2ab = zeros(prob.no, prob.no, prob.no, prob.no)
    
    v = Dict{Int, Array{Float64, 2}}()
    
    start = 1
    for m in 1:length(spin_pairs)
        tmp = C[start:start+spin_pairs[m].dim-1]
        v[m] = reshape(tmp, (length(cats_a[spin_pairs[m].pair[1]].idxs), length(cats_b[spin_pairs[m].pair[2]].idxs)))
        start += spin_pairs[m].dim
    end
    
    categories = ActiveSpaceSolvers.RASCI.generate_spin_categories(prob)
    fock_list_a, del_at_a = make_fock_from_categories(categories, prob, "alpha")
    #compute configs
    as = compute_config_dict(fock_list_a, prob, "alpha")
    rev_as = Dict(value => key for (key, value) in as)

    #println("starting rdm2aa")
    for m in 1:length(spin_pairs)
        cat_Ia = cats_a[spin_pairs[m].pair[1]]
        for I in cats_a[spin_pairs[m].pair[1]].idxs
            config = as[I]
            Ia_local = I-cat_Ia.shift
            for s in config
                sgna, config_a = apply_a_dumb(config, s)
                if isempty(config_a) == false
                    for r in config_a
                        sgnaa, config_aa = apply_a_dumb(config_a, r)
                        for q in 1:prob.no
                            sgnc, config_c = apply_c_dumb(config_aa, q)
                            for p in 1:prob.no
                                sgncc, config_cc = apply_c_dumb(config_c, p)
                                if haskey(rev_as, config_cc)
                                    cat_Ja = find_cat(rev_as[config_cc], cats_a)
                                    n = find_spin_pair(spin_pairs, (cat_Ja.idx, spin_pairs[m].pair[2]))
                                    if n != 0
                                        Ja_local = rev_as[config_cc]-cat_Ja.shift
                                        sgn = sgna*sgnaa*sgnc*sgncc
                                        rdm2aa[p,s,q,r] += sgn*dot(v[n][Ja_local,:], v[m][Ia_local,:])
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    fock_list_b, del_at_b = make_fock_from_categories(categories, prob, "beta")
    #compute configs
    bs = compute_config_dict(fock_list_b, prob, "beta")
    rev_bs = Dict(value => key for (key, value) in bs)

    for m in 1:length(spin_pairs)
        cat_Ib = cats_b[spin_pairs[m].pair[2]]
        for I in cats_b[spin_pairs[m].pair[2]].idxs
            config = bs[I]
            Ib_local = I-cat_Ib.shift
            for s in config
                sgna, config_a = apply_a_dumb(config, s)
                if isempty(config_a) == false
                    for r in config_a
                        sgnaa, config_aa = apply_a_dumb(config_a, r)
                        for q in 1:prob.no
                            sgnc, config_c = apply_c_dumb(config_aa, q)
                            for p in 1:prob.no
                                sgncc, config_cc = apply_c_dumb(config_c, p)
                                if haskey(rev_bs, config_cc)
                                    cat_Ja = find_cat(rev_bs[config_cc], cats_b)
                                    n = find_spin_pair(spin_pairs, (spin_pairs[m].pair[1], cat_Ja.idx))
                                    if n != 0
                                        Ja_local = rev_bs[config_cc]-cat_Ja.shift
                                        sgn = sgna*sgnaa*sgnc*sgncc
                                        rdm2bb[p,s,q,r] += sgn*dot(v[n][:, Ja_local], v[m][:, Ib_local])
                                    end
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    #println("starting rdm2ab")
    
    #alpha beta  p'r'sq
    for m in 1:length(spin_pairs)
        cat_Ia = cats_a[spin_pairs[m].pair[1]]
        cat_Ib = cats_b[spin_pairs[m].pair[2]]
        for Ia in cats_a[spin_pairs[m].pair[1]].idxs
            Ia_local = Ia-cat_Ia.shift
            for q in 1:prob.no, p in 1:prob.no
                Ja = cat_Ia.lookup[q,p,Ia_local]
                Ja != 0 || continue
                sign_pq = sign(Ja)
                Ja = abs(Ja)
                cata_pq = find_cat(Ja, cats_a)
                for Ib in cats_b[spin_pairs[m].pair[2]].idxs
                    Ib_local = Ib-cat_Ib.shift
                    for s in 1:prob.no, r in 1:prob.no
                        Jb = cat_Ib.lookup[s,r,Ib_local]
                        Jb != 0 || continue
                        sign_rs = sign(Jb)
                        Jb = abs(Jb)
                        catb_sr = find_cat(Jb, cats_b)
                        n = find_spin_pair(spin_pairs, (cata_pq.idx, catb_sr.idx))
                        n != 0 || continue
                        Ja_local = Ja-cata_pq.shift
                        Jb_local = Jb-catb_sr.shift
                        rdm2ab[p,q,r,s] += sign_pq*sign_rs*v[n][Ja_local, Jb_local]*v[m][Ia_local, Ib_local]
                    end
                end
            end
        end
    end
    return rdm1a, rdm1b, rdm2aa, rdm2bb, rdm2ab#=}}}=#
end

"""
    fill_lu_HP(bra::RASCIAnsatz, ket::RASCIAnsatz; spin="alpha", type="c")

Fills all types of HP categories, used mainly when TDMs are computed
"""
function fill_lu_HP(bra::RASCIAnsatz, ket::RASCIAnsatz; spin="alpha", type="c")
    categories = ActiveSpaceSolvers.RASCI.generate_spin_categories(ket)#={{{=#
    all_cats = Vector{HP_Category}()
    all_cats_bra = Vector{ActiveSpaceSolvers.RASCI.HP_Category_Bra}()
    cats_a = deepcopy(categories)
    cats_b = deepcopy(categories)
    fock_list_a, del_at_a = make_fock_from_categories(categories, ket, "alpha")
    deleteat!(cats_a, del_at_a)
    len_cat_a = length(cats_a)
        
    fock_list_b, del_at_b = make_fock_from_categories(categories, ket, "beta")
    deleteat!(cats_b, del_at_b)
    len_cat_b = length(cats_b)

    if spin == "alpha"
        connected = make_spincategory_connections(cats_a, cats_b, ket)
        #compute configs
        as = compute_config_dict(fock_list_a, ket, "alpha")
        #this reverses the config dictionary to get the index as the key 
        rev_as = Dict(value => key for (key, value) in as)
        max_a = length(as)

        #calculate bra config dictonary
        categories_bra = ActiveSpaceSolvers.RASCI.generate_spin_categories(bra)
        cats_a_bra = deepcopy(categories_bra)
        fock_list_a_bra, del_at_a_bra = make_fock_from_categories(categories_bra, bra, "alpha")
        deleteat!(cats_a_bra, del_at_a_bra)
        
        cats_b_bra = deepcopy(categories_bra)
        fock_list_b_bra, del_at_b_bra = make_fock_from_categories(categories_bra, bra, "beta")
        deleteat!(cats_b_bra, del_at_b_bra)
        
        connected_bra = make_spincategory_connections(cats_a_bra, cats_b_bra, bra)
        as_bra = compute_config_dict(fock_list_a_bra, bra, "alpha")
        #this reverses the config dictionary to get the index as the key 
        rev_as_bra = Dict(value => key for (key, value) in as_bra)
        max_a_bra = length(as_bra)

        if type == "no_op"
            shift = 0
            for j in 1:len_cat_a
                idxas = Vector{Int}()
                graph_a = make_cat_graphs(fock_list_a[j], ket)
                idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
                sort!(idxas)
                push!(all_cats, HP_Category_Bra(j, connected[j], idxas, shift))
                shift += length(idxas)
            end
            shift = 0
            for m in 1:length(cats_a_bra)
                idxas_bra = Vector{Int}()
                graph_a_bra = make_cat_graphs(fock_list_a_bra[m], bra)
                idxas_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a_bra, 1, graph_a_bra.max, idxas_bra, rev_as_bra) 
                sort!(idxas_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxas_bra, shift))
                shift += length(idxas_bra)
            end
            return all_cats_bra, all_cats        
        end
            

        if type == "c"
            #HP_Category_C
            shift = 0
            for j in 1:len_cat_a
                idxas = Vector{Int}()
                graph_a = make_cat_graphs(fock_list_a[j], ket)
                idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
                sort!(idxas)
                lu = zeros(Int, graph_a.no, length(idxas))
                push!(all_cats, HP_Category_C(j, cats_a[j], connected[j], idxas, shift, lu))
                shift += length(idxas)
            end
            
            shift = 0
            for m in 1:length(cats_a_bra)
                idxas_bra = Vector{Int}()
                graph_a_bra = make_cat_graphs(fock_list_a_bra[m], bra)
                idxas_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a_bra, 1, graph_a_bra.max, idxas_bra, rev_as_bra) 
                sort!(idxas_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxas_bra, shift))
                shift += length(idxas_bra)
            end
            
            for k in 1:len_cat_a
                graph_a = make_cat_graphs(fock_list_a[k], ket)
                lu = ActiveSpaceSolvers.RASCI.dfs_c(graph_a, 1, graph_a.max, all_cats[k].lookup, all_cats, all_cats_bra, rev_as, rev_as_bra)
                #all_cats[k].lookup .= lu
            end

            return all_cats_bra, all_cats        
        end

        if type == "a"
            #HP_Category_A
            shift = 0
            for j in 1:len_cat_a
                idxas = Vector{Int}()
                graph_a = make_cat_graphs(fock_list_a[j], ket)
                idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
                sort!(idxas)
                lu = zeros(Int, graph_a.no,length(idxas))
                push!(all_cats, HP_Category_A(j, cats_a[j], connected[j], idxas, shift, lu))
                shift += length(idxas)
            end
            shift = 0
            for m in 1:length(cats_a_bra)
                idxas_bra = Vector{Int}()
                graph_a_bra = make_cat_graphs(fock_list_a_bra[m], bra)
                idxas_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a_bra, 1, graph_a_bra.max, idxas_bra, rev_as_bra) 
                sort!(idxas_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxas_bra, shift))
                shift += length(idxas_bra)
            end
            
            for k in 1:len_cat_a
                graph_a = make_cat_graphs(fock_list_a[k], ket)
                lu = ActiveSpaceSolvers.RASCI.dfs_a(graph_a, 1, graph_a.max, all_cats[k].lookup, all_cats, all_cats_bra, rev_as, rev_as_bra)
                #all_cats[k].lookup .= lu
            end
            
            return all_cats_bra, all_cats        
        end

        if type == "ca"
            #HP_Category_CA
            shift = 0
            for j in 1:len_cat_a
                idxas = Vector{Int}()
                graph_a = make_cat_graphs(fock_list_a[j], ket)
                idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
                sort!(idxas)
                lu = zeros(Int, graph_a.no, graph_a.no, length(idxas))
                cat_lu = zeros(Int, graph_a.no, graph_a.no, length(idxas))
                push!(all_cats, HP_Category_CA(j, cats_a[j], connected[j], idxas, shift, lu, cat_lu))
                shift += length(idxas)
            end

            shift = 0
            for m in 1:length(cats_a_bra)
                idxas_bra = Vector{Int}()
                graph_a_bra = make_cat_graphs(fock_list_a_bra[m], bra)
                idxas_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a_bra, 1, graph_a_bra.max, idxas_bra, rev_as_bra) 
                sort!(idxas_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxas_bra, shift))
                shift += length(idxas_bra)
            end
            
            for k in 1:len_cat_a
                graph_a = make_cat_graphs(fock_list_a[k], ket)
                lu = ActiveSpaceSolvers.RASCI.dfs_ca(graph_a, 1, graph_a.max, all_cats[k].lookup, all_cats, all_cats_bra, rev_as, rev_as_bra)
            end

            return all_cats_bra, all_cats        
        end

        if type == "cc"
            #HP_Category_CC
            shift = 0
            for j in 1:len_cat_a
                idxas = Vector{Int}()
                graph_a = make_cat_graphs(fock_list_a[j], ket)
                idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
                sort!(idxas)
                lu = zeros(Int, graph_a.no, graph_a.no, length(idxas))
                push!(all_cats, HP_Category_CC(j, cats_a[j], connected[j], idxas, shift, lu))
                shift += length(idxas)
            end
            
            shift = 0
            for m in 1:length(cats_a_bra)
                idxas_bra = Vector{Int}()
                graph_a_bra = make_cat_graphs(fock_list_a_bra[m], bra)
                idxas_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a_bra, 1, graph_a_bra.max, idxas_bra, rev_as_bra) 
                sort!(idxas_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxas_bra, shift))
                shift += length(idxas_bra)
            end
            for k in 1:len_cat_a
                graph_a = make_cat_graphs(fock_list_a[k], ket)
                lu = ActiveSpaceSolvers.RASCI.dfs_cc(graph_a, 1, graph_a.max, all_cats[k].lookup, all_cats, all_cats_bra, rev_as, rev_as_bra)
                #all_cats[k].lookup .= lu
            end
        
            return all_cats_bra, all_cats        
        end

        if type == "cca"
            #HP_Category_CCA
            shift = 0
            for j in 1:len_cat_a
                idxas = Vector{Int}()
                graph_a = make_cat_graphs(fock_list_a[j], ket)
                idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
                sort!(idxas)
                lu = zeros(Int, graph_a.no, graph_a.no, graph_a.no, length(idxas))
                push!(all_cats, HP_Category_CCA(j, cats_a[j], connected[j], idxas, shift, lu))
                shift += length(idxas)
            end

            shift = 0
            for m in 1:length(cats_a_bra)
                idxas_bra = Vector{Int}()
                graph_a_bra = make_cat_graphs(fock_list_a_bra[m], bra)
                idxas_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a_bra, 1, graph_a_bra.max, idxas_bra, rev_as_bra) 
                sort!(idxas_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxas_bra, shift))
                shift += length(idxas_bra)
            end
            
            for k in 1:len_cat_a
                graph_a = make_cat_graphs(fock_list_a[k], ket)
                lu = ActiveSpaceSolvers.RASCI.dfs_cca(graph_a, 1, graph_a.max, all_cats[k].lookup, all_cats, all_cats_bra, rev_as, rev_as_bra)
                #all_cats[k].lookup .= lu
            end

            return all_cats_bra, all_cats        
        end

        if type == "ccaa"
            #HP_Category_CCAA
            shift = 0
            for j in 1:len_cat_a
                idxas = Vector{Int}()
                graph_a = make_cat_graphs(fock_list_a[j], ket)
                idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
                sort!(idxas)
                lu = zeros(Int, graph_a.no, graph_a.no, graph_a.no, graph_a.no, length(idxas))
                push!(all_cats, HP_Category_CCAA(j, cats_a[j], connected[j], idxas, shift, lu))
                shift += length(idxas)
            end

            shift = 0
            for m in 1:length(cats_a_bra)
                idxas_bra = Vector{Int}()
                graph_a_bra = make_cat_graphs(fock_list_a_bra[m], bra)
                idxas_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a_bra, 1, graph_a_bra.max, idxas_bra, rev_as_bra) 
                sort!(idxas_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxas_bra, shift))
                shift += length(idxas_bra)
            end
            
            for k in 1:len_cat_a
                graph_a = make_cat_graphs(fock_list_a[k], ket)
                lu = ActiveSpaceSolvers.RASCI.dfs_ccaa(graph_a, 1, graph_a.max, all_cats[k].lookup, all_cats, all_cats_bra, rev_as, rev_as_bra)
            end
            return all_cats_bra, all_cats        
        end
    else
        connected = make_spincategory_connections(cats_b, cats_a, ket)
        #compute configs
        bs = compute_config_dict(fock_list_b, ket, "beta")
        rev_bs = Dict(value => key for (key, value) in bs)
        max_b = length(bs)
        
        #calculate bra config dictonary
        categories_bra = ActiveSpaceSolvers.RASCI.generate_spin_categories(bra)
        cats_a_bra = deepcopy(categories_bra)
        fock_list_a_bra, del_at_a_bra = make_fock_from_categories(categories_bra, bra, "alpha")
        deleteat!(cats_a_bra, del_at_a_bra)
        
        cats_b_bra = deepcopy(categories_bra)
        fock_list_b_bra, del_at_b_bra = make_fock_from_categories(categories_bra, bra, "beta")
        deleteat!(cats_b_bra, del_at_b_bra)
        
        connected_bra = make_spincategory_connections(cats_b_bra, cats_a_bra, bra)
        bs_bra = compute_config_dict(fock_list_b_bra, bra, "beta")
        #this reverses the config dictionary to get the index as the key 
        rev_bs_bra = Dict(value => key for (key, value) in bs_bra)
        max_b_bra = length(bs_bra)
        
        if type == "no_op"
            shift = 0
            for j in 1:len_cat_b
                idxbs = Vector{Int}()
                graph_b = make_cat_graphs(fock_list_b[j], ket)
                idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
                sort!(idxbs)
                push!(all_cats, HP_Category_Bra(j, connected[j], idxbs, shift))
                shift += length(idxbs)
            end
            
            shift = 0
            for m in 1:length(cats_b_bra)
                idxbs_bra = Vector{Int}()
                graph_b_bra = make_cat_graphs(fock_list_b_bra[m], bra)
                idxbs_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b_bra, 1, graph_b_bra.max, idxbs_bra, rev_bs_bra) 
                sort!(idxbs_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxbs_bra, shift))
                shift += length(idxbs_bra)
            end
            return all_cats_bra, all_cats        
        end

        
        if type == "c"
            #HP_Category_C
            shift = 0
            for j in 1:len_cat_b
                idxbs = Vector{Int}()
                graph_b = make_cat_graphs(fock_list_b[j], ket)
                idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
                sort!(idxbs)
                lu = zeros(Int, graph_b.no, length(idxbs))
                push!(all_cats, HP_Category_C(j, cats_b[j], connected[j], idxbs, shift, lu))
                shift += length(idxbs)
            end
            
            shift = 0
            for m in 1:length(cats_b_bra)
                idxbs_bra = Vector{Int}()
                graph_b_bra = make_cat_graphs(fock_list_b_bra[m], bra)
                idxbs_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b_bra, 1, graph_b_bra.max, idxbs_bra, rev_bs_bra) 
                sort!(idxbs_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxbs_bra, shift))
                shift += length(idxbs_bra)
            end
            
            for k in 1:len_cat_b
                graph_b = make_cat_graphs(fock_list_b[k], ket)
                lu = ActiveSpaceSolvers.RASCI.dfs_c(graph_b, 1, graph_b.max, all_cats[k].lookup, all_cats, all_cats_bra, rev_bs, rev_bs_bra)
            end

            return all_cats_bra, all_cats        
        end

        if type == "a"
            #HP_Category_A
            shift = 0
            for j in 1:len_cat_b
                idxbs = Vector{Int}()
                graph_b = make_cat_graphs(fock_list_b[j], ket)
                idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
                sort!(idxbs)
                lu = zeros(Int, graph_b.no,length(idxbs))
                push!(all_cats, HP_Category_A(j, cats_b[j], connected[j], idxbs, shift, lu))
                shift += length(idxbs)
            end

            shift = 0
            for m in 1:length(cats_b_bra)
                idxbs_bra = Vector{Int}()
                graph_b_bra = make_cat_graphs(fock_list_b_bra[m], bra)
                idxbs_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b_bra, 1, graph_b_bra.max, idxbs_bra, rev_bs_bra) 
                sort!(idxbs_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxbs_bra, shift))
                shift += length(idxbs_bra)
            end
            
            for k in 1:len_cat_b
                graph_b = make_cat_graphs(fock_list_b[k], ket)
                lu = ActiveSpaceSolvers.RASCI.dfs_a(graph_b, 1, graph_b.max, all_cats[k].lookup, all_cats, all_cats_bra, rev_bs, rev_bs_bra)
            end
            return all_cats_bra, all_cats        
        end

        if type == "ca"
            #HP_Category_CA
            shift = 0
            for j in 1:len_cat_b
                idxbs = Vector{Int}()
                graph_b = make_cat_graphs(fock_list_b[j], ket)
                idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
                sort!(idxbs)
                lu = zeros(Int, graph_b.no, graph_b.no, length(idxbs))
                cat_lu = zeros(Int, graph_b.no, graph_b.no, length(idxbs))
                push!(all_cats, HP_Category_CA(j, cats_b[j], connected[j], idxbs, shift, lu, cat_lu))
                shift += length(idxbs)
            end

            shift = 0
            for m in 1:length(cats_b_bra)
                idxbs_bra = Vector{Int}()
                graph_b_bra = make_cat_graphs(fock_list_b_bra[m], bra)
                idxbs_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b_bra, 1, graph_b_bra.max, idxbs_bra, rev_bs_bra) 
                sort!(idxbs_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxbs_bra, shift))
                shift += length(idxbs_bra)
            end
            
            for k in 1:len_cat_b
                graph_b = make_cat_graphs(fock_list_b[k], ket)
                lu = ActiveSpaceSolvers.RASCI.dfs_ca(graph_b, 1, graph_b.max, all_cats[k].lookup, all_cats, all_cats_bra, rev_bs, rev_bs_bra)
            end
            return all_cats_bra, all_cats        
        end

        if type == "cc"
            #HP_Category_CC
            shift = 0
            for j in 1:len_cat_b
                idxbs = Vector{Int}()
                graph_b = make_cat_graphs(fock_list_b[j], ket)
                idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
                sort!(idxbs)
                lu = zeros(Int, graph_b.no, graph_b.no, length(idxbs))
                push!(all_cats, HP_Category_CC(j, cats_b[j], connected[j], idxbs, shift, lu))
                shift += length(idxbs)
            end

            shift = 0
            for m in 1:length(cats_b_bra)
                idxbs_bra = Vector{Int}()
                graph_b_bra = make_cat_graphs(fock_list_b_bra[m], bra)
                idxbs_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b_bra, 1, graph_b_bra.max, idxbs_bra, rev_bs_bra) 
                sort!(idxbs_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxbs_bra, shift))
                shift += length(idxbs_bra)
            end
            
            for k in 1:len_cat_b
                graph_b = make_cat_graphs(fock_list_b[k], ket)
                lu = ActiveSpaceSolvers.RASCI.dfs_cc(graph_b, 1, graph_b.max, all_cats[k].lookup, all_cats, all_cats_bra, rev_bs, rev_bs_bra)
            end
            return all_cats_bra, all_cats        
        end

        if type == "cca"
            #HP_Category_CCA
            shift =0 
            for j in 1:len_cat_b
                idxbs = Vector{Int}()
                graph_b = make_cat_graphs(fock_list_b[j], ket)
                idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
                sort!(idxbs)
                lu = zeros(Int, graph_b.no, graph_b.no, graph_b.no, length(idxbs))
                push!(all_cats, HP_Category_CCA(j, cats_b[j], connected[j], idxbs, shift, lu))
                shift += length(idxbs)
            end

            shift = 0
            for m in 1:length(cats_b_bra)
                idxbs_bra = Vector{Int}()
                graph_b_bra = make_cat_graphs(fock_list_b_bra[m], bra)
                idxbs_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b_bra, 1, graph_b_bra.max, idxbs_bra, rev_bs_bra) 
                sort!(idxbs_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxbs_bra, shift))
                shift += length(idxbs_bra)
            end
            
            for k in 1:len_cat_b
                graph_b = make_cat_graphs(fock_list_b[k], ket)
                lu = ActiveSpaceSolvers.RASCI.dfs_cca(graph_b, 1, graph_b.max, all_cats[k].lookup, all_cats, all_cats_bra, rev_bs, rev_bs_bra)
            end
            return all_cats_bra, all_cats        
        end

       if type == "ccaa"
           #HP_Category_CCAA
           shift = 0
           for j in 1:len_cat_b
               idxbs = Vector{Int}()
               graph_b = make_cat_graphs(fock_list_b[j], ket)
               idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
               sort!(idxbs)
               lu = zeros(Int, graph_b.no, graph_b.no, graph_b.no, graph_b.no, length(idxbs))
               push!(all_cats, HP_Category_CCAA(j, cats_b[j], connected[j], idxbs, shift, lu))
               shift += length(idxbs)
           end

           shift = 0
           for m in 1:length(cats_b_bra)
                idxbs_bra = Vector{Int}()
                graph_b_bra = make_cat_graphs(fock_list_b_bra[m], bra)
                idxbs_bra = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b_bra, 1, graph_b_bra.max, idxbs_bra, rev_bs_bra) 
                sort!(idxbs_bra)
                push!(all_cats_bra, HP_Category_Bra(m, connected_bra[m], idxbs_bra, shift))
                shift += length(idxbs_bra)
            end
           
            for k in 1:len_cat_b
                graph_b = make_cat_graphs(fock_list_b[k], ket)
                lu = ActiveSpaceSolvers.RASCI.dfs_ccaa(graph_b, 1, graph_b.max, all_cats[k].lookup, all_cats, all_cats_bra, rev_bs, rev_bs_bra)
            end
            return all_cats_bra, all_cats        
       end
    end#=}}}=#
end

"""
    bubble_sort(arr)

Normal orders occupation strings and returns number of swaps used to determine the sign
"""
function bubble_sort(arr)
    len = length(arr) #={{{=#
    count = 0
    # Traverse through all array elements
    for i = 1:len-1
        for j = 2:len
        # Last i elements are already in place
        # Swap if the element found is greater
            if arr[j-1] > arr[j] 
                count += 1
                tmp = arr[j-1]
                arr[j-1] = arr[j]
                arr[j] = tmp
            end
        end
    end
    return count, arr#=}}}=#
end

"""
"""
function apply_a_dumb(config, orb)
    spot = first(findall(x->x==orb, config))
    new = Vector(config)
    
    splice!(new, spot)

    sign = 1 
    if spot % 2 != 1
        sign = -1
    end
    return sign, new
end

function apply_c_dumb(config, orb)
    new = Vector(config)
    insert_here = 1
    
    if isempty(config)
        new = [orb]
        sign_c = 1
        
    else
        for i in 1:length(config)
            if config[i] > orb
                insert_here = i
                break
            else
                insert_here += 1
            end
        end

        insert!(new, insert_here, orb)

        sign_c = 1
        if insert_here % 2 != 1
            sign_c = -1
        end
    end
    return sign_c, new
end

"""
    apply_a(config, orb, config_dict_ket, config_dict_bra, cats_ket::Vector{<:HP_Category}, cats_bra::Vector{<:HP_Category})

Apply annhilation operator, used in dfs_a() in type_RASCI_OlsenGraph.jl
"""
function apply_a(config, orb, config_dict_ket, config_dict_bra, cats_ket::Vector{<:HP_Category}, cats_bra::Vector{<:HP_Category})
    idx_org = config_dict_ket[config]#={{{=#
    cat_org = find_cat(idx_org, cats_ket)
    idx_local = findfirst(item -> item == idx_org, cat_org.idxs)

    spot = first(findall(x->x==orb, config))
    new = Vector(config)
    
    splice!(new, spot)
    if haskey(config_dict_bra, new)
        idx = config_dict_bra[new]
    else
        return 1, 0, 0, 0
    end

    cat = find_cat(idx, cats_bra)
    if cat == 0
        return 1, 0, 0, 0
    end

    sign = 1 
    if spot % 2 != 1
        sign = -1
    end
    return sign, new, idx_local, idx#=}}}=#
end

"""
    apply_c(config, orb, config_dict_ket, config_dict_bra, cats_ket::Vector{<:HP_Category}, cats_bra::Vector{<:HP_Category})

Apply creation operator, used in dfs_c() in type_RASCI_OlsenGraph.jl
"""
function apply_c(config, orb, config_dict_ket, config_dict_bra, cats_ket::Vector{<:HP_Category}, cats_bra::Vector{<:HP_Category})
    idx_org = config_dict_ket[config]#={{{=#
    cat_org = find_cat(idx_org, cats_ket)
    idx_local = findfirst(item -> item == idx_org, cat_org.idxs)

    new = Vector(config)
    insert_here = 1
    
    if isempty(config)
        new = [orb]
        sign_c = 1
        
        if haskey(config_dict_bra, new);
            idx = config_dict_bra[new]
        else
            return 1, 0, 0,0
        end
        cat = find_cat(idx, cats_bra)
        if cat == 0
            return 1, 0,0,0
        end

    else
        for i in 1:length(config)
            if config[i] > orb
                insert_here = i
                break
            else
                insert_here += 1
            end
        end

        insert!(new, insert_here, orb)
        if haskey(config_dict_bra, new);
            idx = config_dict_bra[new]
        else
            return 1, 0, 0,0
        end
        cat = find_cat(idx, cats_bra)
        if cat == 0
            return 1, 0,0,0
        end

        sign_c = 1
        if insert_here % 2 != 1
            sign_c = -1
        end
    end
    return sign_c, new, idx_local, idx#=}}}=#
end

"""
    apply_ca(config, orb, orb2, config_dict_ket, config_dict_bra, cats_ket::Vector{<:HP_Category}, cats_bra::Vector{<:HP_Category})

Apply creation-annhilation pair operator, used in dfs_ca() in type_RASCI_OlsenGraph.jl
"""
function apply_ca(config, orb, orb2, config_dict_ket, config_dict_bra, cats_ket::Vector{<:HP_Category}, cats_bra::Vector{<:HP_Category})
    idx_org = config_dict_ket[config]#={{{=#
    cat_org = find_cat(idx_org, cats_ket)
    idx_local = findfirst(item -> item == idx_org, cat_org.idxs)
    
    spot = first(findall(x->x==orb, config))
    new = Vector(config)
    splice!(new, spot)
    
    sign_a = 1 
    if spot % 2 != 1
        sign_a = -1
    end
    
    if orb2 in new
        return 1, 0, 0,0
    end

    insert_here = 1
    new2 = Vector(new)
    sign_c = 1
    
    if isempty(new)
        new2 = [orb2]
        
        if haskey(config_dict_bra, new2);
            idx = config_dict_bra[new2]
        else
            return 1, 0, 0,0
        end
        cat = find_cat(idx, cats_bra)
        if cat == 0
            return 1, 0,0,0
        end

    else
        for i in 1:length(new)
            if new[i] > orb2
                insert_here = i
                break
            else
                insert_here += 1
            end
        end

        insert!(new2, insert_here, orb2)

        sign_c = 1
        if insert_here % 2 != 1
            sign_c = -1
        end

        if haskey(config_dict_bra, new2);
            idx = config_dict_bra[new2]
        else
            return 1, 0, 0,0
        end
        cat = find_cat(idx, cats_bra)
        if cat == 0
            return 1, 0,0,0
        end
    end

    return sign_a*sign_c, new2, idx_local, idx#=}}}=#
end

"""
    apply_cc(config, orb, orb2, config_dict_ket, config_dict_bra, cats_ket::Vector{<:HP_Category}, cats_bra::Vector{<:HP_Category})

Apply creation-creation pair operator, used in dfs_cc() in type_RASCI_OlsenGraph.jl
"""
function apply_cc(config, orb, orb2, config_dict_ket, config_dict_bra, cats_ket::Vector{<:HP_Category}, cats_bra::Vector{<:HP_Category})
    idx_org = config_dict_ket[config]#={{{=#
    cat_org = find_cat(idx_org, cats_ket)
    idx_local = findfirst(item -> item == idx_org, cat_org.idxs)

    new = Vector(config)
    insert_here = 1
    sign_c = 1
    
    if isempty(config)
        new = [orb]
        sign_c = 1

    else
        for i in 1:length(config)
            if config[i] > orb
                insert_here = i
                break
            else
                insert_here += 1
            end
        end

        insert!(new, insert_here, orb)

        if insert_here % 2 != 1
            sign_c = -1
        end
    end

    insert_here = 1
    new2 = Vector(new)
    for i in 1:length(new)
        if new[i] > orb2
            insert_here = i
            break
        else
            insert_here += 1
        end
    end

    insert!(new2, insert_here, orb2)

    sign_cc = 1
    if insert_here % 2 != 1
        sign_cc = -1
    end
    
    if haskey(config_dict_bra, new2);
        idx = config_dict_bra[new2]
    else
        return 1, 0, 0,0
    end
    cat = find_cat(idx, cats_bra)
    if cat == 0
        return 1, 0,0,0
    end

    return sign_c*sign_cc, new2, idx_local, idx#=}}}=#
end

"""
    apply_cca(config, orb_a, orb, orb2, config_dict_ket, config_dict_bra, cats_ket::Vector{<:HP_Category}, cats_bra::Vector{<:HP_Category})

Apply creation-creation-annhilation operators, used in dfs_cca() in type_RASCI_OlsenGraph.jl
"""
function apply_cca(config, orb_a, orb, orb2, config_dict_ket, config_dict_bra, cats_ket::Vector{<:HP_Category}, cats_bra::Vector{<:HP_Category})
    idx_org = config_dict_ket[config]#={{{=#
    cat_org = find_cat(idx_org, cats_ket)
    idx_local = findfirst(item -> item == idx_org, cat_org.idxs)

    #apply annhilation
    spot = first(findall(x->x==orb_a, config))
    new_a = Vector(config)
    splice!(new_a, spot)
    sign_a = 1
    if spot % 2 != 1
        sign_a = -1
    end
    
    #apply first creation
    new = Vector(new_a)
    insert_here = 1
    sign_c = 1
    
    if isempty(new_a)
        new = [orb]
        sign_c = 1

    else
        for i in 1:length(new_a)
            if new_a[i] > orb
                insert_here = i
                break
            else
                insert_here += 1
            end
        end

        insert!(new, insert_here, orb)

        if insert_here % 2 != 1
            sign_c = -1
        end
    end
    
    #apply second creation
    insert_here = 1
    new2 = Vector(new)
    for i in 1:length(new)
        if new[i] > orb2
            insert_here = i
            break
        else
            insert_here += 1
        end
    end

    insert!(new2, insert_here, orb2)

    sign_cc = 1
    if insert_here % 2 != 1
        sign_cc = -1
    end
    
    if haskey(config_dict_bra, new2);
        idx = config_dict_bra[new2]
    else
        return 1, 0, 0,0
    end
    cat = find_cat(idx, cats_bra)
    if cat == 0
        return 1, 0,0,0
    end

    return sign_a*sign_c*sign_cc, new2, idx_local, idx#=}}}=#
end

"""
    apply_ccaa(config, orb_a, orb_aa, orb, orb2, config_dict_ket, config_dict_bra, cats_ket::Vector{<:HP_Category}, cats_bra::Vector{<:HP_Category})

Apply creation-creation-annhilation-annhilation operators, used in dfs_ccaa() in type_RASCI_OlsenGraph.jl, only used for 2rdms
"""
function apply_ccaa(config, orb_a, orb_aa, orb, orb2, config_dict_ket, config_dict_bra, cats_ket::Vector{<:HP_Category}, cats_bra::Vector{<:HP_Category})
    idx_org = config_dict_ket[config]#={{{=#
    cat_org = find_cat(idx_org, cats_ket)
    idx_local = findfirst(item -> item == idx_org, cat_org.idxs)

    #apply first annhilation
    spot = findfirst(x->x==orb_a, config)
    #spot = first(findall(x->x==orb_a, config))
    new_a = Vector(config)
    splice!(new_a, spot)
    sign_a = 1
    if spot % 2 != 1
        sign_a = -1
    end
    
    #apply second annhilation
    spota = findfirst(x->x==orb_aa, new_a)
    #spota = first(findall(x->x==orb_aa, new_a))
    new_aa = Vector(new_a)
    splice!(new_aa, spota)
    sign_aa = 1
    if spota % 2 != 1
        sign_aa = -1
    end
    
    #apply first creation
    new = Vector(new_aa)
    insert_here = 1
    sign_c = 1
    
    if isempty(new_aa)
        new = [orb]
        sign_c = 1

    else
        for i in 1:length(new_aa)
            if new_aa[i] > orb
                insert_here = i
                break
            else
                insert_here += 1
            end
        end

        insert!(new, insert_here, orb)

        if insert_here % 2 != 1
            sign_c = -1
        end
    end
    
    #apply second creation
    insert_here = 1
    new2 = Vector(new)
    for i in 1:length(new)
        if new[i] > orb2
            insert_here = i
            break
        else
            insert_here += 1
        end
    end

    insert!(new2, insert_here, orb2)

    sign_cc = 1
    if insert_here % 2 != 1
        sign_cc = -1
    end
    
    if haskey(config_dict_bra, new2);
        idx = config_dict_bra[new2]
    else
        return 1, 0, 0,0
    end
    cat = find_cat(idx, cats_bra)
    if cat == 0
        return 1, 0,0,0
    end

    return sign_a*sign_aa*sign_c*sign_cc, new2, idx_local, idx#=}}}=#
end

"""
    apply_single_excitation!(config, a_orb, c_orb, config_dict, categories::Vector{HP_Category_CA})

"""
function apply_single_excitation!(config, a_orb, c_orb, config_dict, categories::Vector{HP_Category_CA})
    #println("Config: ", config)
    #println(a_orb, " : ", c_orb)
    idx_org = config_dict[config]
    cat_org = find_cat(idx_org, categories)
    idx_local = findfirst(item -> item == idx_org, cat_org.idxs)

    spot = first(findall(x->x==a_orb, config))#={{{=#
    new = Vector(config)
    splice!(new, spot)
    
    sign_a = 1 
    if spot % 2 != 1
        sign_a = -1
    end
    
    if c_orb in new
        return 1, 0, 0,0
    end

    insert_here = 1
    new2 = Vector(new)

    if isempty(new)
        new2 = [c_orb]
        sign_c = 1
        #println("New config: ", new2)
        
        if haskey(config_dict, new2);
            idx = config_dict[new2]
        else
            return 1, 0, 0,0
        end
        cat = find_cat(idx, categories)
        if cat == 0
            return 1, 0,0,0
        end

    else
        for i in 1:length(new)
            if new[i] > c_orb
                insert_here = i
                break
            else
                insert_here += 1
            end
        end

        insert!(new2, insert_here, c_orb)
        #println("New config: ", new2)

        if haskey(config_dict, new2);
            idx = config_dict[new2]
        else
            return 1, 0, 0,0
        end
        cat = find_cat(idx, categories)
        if cat == 0
            return 1, 0,0,0
        end


        sign_c = 1
        if insert_here % 2 != 1
            sign_c = -1
        end
    end

    return sign_c*sign_a, new2, idx_local, idx#=}}}=#
    #return sign_c*sign_a, new2, idx#=}}}=#
end

"""
    get_gkl(ints::InCoreInts, prob::RASCIAnsatz)

Used in sigma_one and sigma_two to get an instance of the two electron integrals
"""
function get_gkl(ints::InCoreInts, prob::RASCIAnsatz)
    hkl = zeros(prob.no, prob.no)#={{{=#
    hkl .= ints.h1
    gkl = zeros(prob.no, prob.no)
    for k in 1:prob.no
        for l in 1:prob.no
            gkl[k,l] += hkl[k,l]
            x = 0
            for j in 1:prob.no
                if j < k
                    x += ints.h2[k,j,j,l]
                end
            end
            gkl[k,l] -= x
            if k >= l 
                if k == l
                    delta = 1
                else
                    delta = 0
                end
                gkl[k,l] -= ints.h2[k,k,k,l]*1/(1+delta)
            end
        end
    end#=}}}=#
    return gkl
end

"""
    make_rasorbs(rasi_orbs, rasiii_orbs, norbs, frozen_core=false)
"""
function make_rasorbs(rasi_orbs, rasii_orbs, rasiii_orbs, norbs, frozen_core=false)
    if frozen_core==false#={{{=#
        i_orbs = [1:1:rasi_orbs;]
        start2 = rasi_orbs+1
        end2 = start2+rasii_orbs-1
        ii_orbs = [start2:1:end2;]
        start = norbs-rasiii_orbs+1
        iii_orbs = [start:1:norbs;]
        return i_orbs, ii_orbs, iii_orbs
    end#=}}}=#
end


function contract_same_ras!(nroots::Int, ras_orbs::Vector{Int}, current_cat::HP_Category_CA, Ib::Int, Ja_local::Int, spin_pair::Int, hkl::Array{Float64, 2}, sign_kl::Int, v::Dict{Int, Array{Float64, 3}}, sig::SubArray{Float64, 1, Array{Float64, 3}, Tuple{Int64, Int64, Base.Slice{Base.OneTo{Int64}}}, true})
    for i in ras_orbs#={{{=#
        for j in ras_orbs
            Jb = current_cat.lookup[j, i, Ib]
            Jb != 0 || continue
            sign_ij = sign(Jb)
            Jb = abs(Jb)
            Jb_local = Jb-current_cat.shift
            @views v2 = v[spin_pair][Ja_local,Jb_local, :]
            sgn = sign_ij*sign_kl
            h = hkl[i,j]
            for si in 1:nroots
                #@inbounds sig[si] += h*v[spin_pair][Ja_local,Jb_local, si]*sgn
                @inbounds sig[si] += h*v2[si]*sgn
            end
        end
    end
end#=}}}=#

function contract!(nroots::Int, ras_orbs1::Vector{Int}, ras_orbs2::Vector{Int}, current_cat::HP_Category_CA, next_cat::HP_Category_CA, Ib::Int, Ja_local::Int, spin_pair::Int, hkl::Array{Float64, 2}, sign_kl::Int, v::Dict{Int, Array{Float64, 3}}, sig::SubArray{Float64, 1, Array{Float64, 3}, Tuple{Int64, Int64, Base.Slice{Base.OneTo{Int64}}}, true})
    for i in ras_orbs2#={{{=#
        for j in ras_orbs1
            Jb = current_cat.lookup[j, i, Ib]
            Jb != 0 || continue
            sign_ij = sign(Jb)
            Jb = abs(Jb)
            Jb_local = Jb-next_cat.shift
            @views v2 = v[spin_pair][Ja_local,Jb_local, :]
            sgn = sign_ij*sign_kl
            h = hkl[i,j]
            for si in 1:nroots
                #@inbounds sig[si] += h*v[spin_pair][Ja_local,Jb_local, si]*sgn
                @inbounds sig[si] += h*v2[si]*sgn
            end
        end
    end
end#=}}}=#
        
                

        



