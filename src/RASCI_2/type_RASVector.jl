include("type_SubspaceDeterminantString.jl")
using OrderedCollections
using TensorOperations
using InCoreIntegrals
    
struct RasBlock
    focka::Tuple{Int, Int, Int}
    fockb::Tuple{Int, Int, Int}
end

struct RASVector{T}
    data::OrderedDict{RasBlock, Array{T,3}}
    #                 Tuple{(ras1_na, ras2_na, ras3_na), (ras1_nb, ras2_nb, ras3_nb)}, Array{alpha, beta, roots}
end

"""
    RASVector(ras_spaces::Vector{Int, Int, Int, Int}, na, nb, no, max_h, max_p, max_h2, max_p2)

Constructor to create RASCI Vector that allowes problems like DDCI where you want multiple cases of holes/particles
#Arguments
- `ras_spaces`: vector of Ints for number of orbs in each ras subspace
- `na` or `nb`: number of alpha or beta electrons
- `max_h`: maximum number of holes
- `max_p`: maximum number of particles
- `max_h2`: an additional case of maximum number of holes (e.g. max_h=1, max_p=2 then max_h2=2 and max_p2=1)
- `max_p2`: an additional case of maximum number of particles
#Returns
- `RASVector`
"""
#function RASVector(v::Array{T}, ras_spaces::Tuple{Int, Int, Int}, na, nb, no; max_h=0, max_p=0, max_h2=nothing, max_p2=nothing) where T{{{
#    a_blocks, fock_as = make_blocks(ras_spaces, na, max_h, max_p)
#    b_blocks, fock_bs = make_blocks(ras_spaces, nb, max_h, max_p)
#    rasvec = OrderedDict{Tuple{Int, Int, Int, Int}, Array{3, T}}()
#    nroots = size(v, 2)
#    
#    start = 1
#    for i in 1:length(a_blocks)
#        dima = binomial(ras_spaces[1], fock_as[i][1])*binomial(ras_spaces[2], fock_as[i][2])*binomial(ras_spaces[3], fock_as[i][3])
#        for j in 1:length(b_blocks)
#            dimb = binomial(ras_spaces[1], fock_bs[j][1])*binomial(ras_spaces[2], fock_bs[j][2])*binomial(ras_spaces[3], fock_bs[j][3])
#            if a_blocks[i][1]+b_blocks[j][1]<= max_h
#                if a_blocks[i][2]+b_blocks[j][2] <= max_p
#                    rasvec[(a_blocks[i][1], b_blocks[j][1], a_blocks[i][2], b_blocks[j][2])] = reshape(v[start:start+dima*dimb-1, :], dima, dimb, nroots)
#                    start += dima*dimb
#                end
#            end
#        end
#    end
#    
#    if max_h2 && max_p2 != nothing
#        a_blocks2, fock_as2 = make_blocks(ras_spaces, na, max_h2, max_p2)
#        b_blocks2, fock_bs2 = make_blocks(ras_spaces, nb, max_h2, max_p2)
#        for i in 1:length(a_blocks2)
#            dima = binomial(ras_spaces[1], fock_as2[i][1])*binomial(ras_spaces[2], fock_as2[i][2])*binomial(ras_spaces[3], fock_as2[i][3])
#            for j in 1:length(b_blocks2)
#                dimb = binomial(ras_spaces[1], fock_bs2[j][1])*binomial(ras_spaces[2], fock_bs2[j][2])*binomial(ras_spaces[3], fock_bs2[j][3])
#                if a_blocks2[i][1]+b_blocks2[j][1]<= max_h2
#                    if a_blocks2[i][2]+b_blocks2[j][2] <= max_p2
#                        rasvec[(a_blocks2[i][1], b_blocks2[j][1], a_blocks2[i][2], b_blocks2[j][2])] = reshape(v[start:start+dima*dimb-1, :], dima, dimb, nroots)
#                        start += dima*dimb
#                    end
#                end
#            end
#        end
#    end
#    return RASVector(rasvec)
#end}}}

function RASVector(v::Array{T}, ras_spaces::Tuple{Int, Int, Int}, na::Int, nb::Int, no::Int; max_h=0, max_p=0, max_h2=nothing, max_p2=nothing) where T
    a_blocks, fock_as = make_blocks(ras_spaces, na, max_h, max_p)
    b_blocks, fock_bs = make_blocks(ras_spaces, nb, max_h, max_p)
    rasvec = OrderedDict{RasBlock, Array{T,3}}()
    nroots = size(v, 2)
    
    start = 1
    for i in 1:length(a_blocks)
        dima = binomial(ras_spaces[1], fock_as[i][1])*binomial(ras_spaces[2], fock_as[i][2])*binomial(ras_spaces[3], fock_as[i][3])
        for j in 1:length(b_blocks)
            dimb = binomial(ras_spaces[1], fock_bs[j][1])*binomial(ras_spaces[2], fock_bs[j][2])*binomial(ras_spaces[3], fock_bs[j][3])
            if a_blocks[i][1]+b_blocks[j][1]<= max_h
                if a_blocks[i][2]+b_blocks[j][2] <= max_p
                    block1 = RasBlock(fock_as[i], fock_bs[j])
                    rasvec[block1] = reshape(v[start:start+dima*dimb-1, :], dima, dimb, nroots)
                    start += dima*dimb
                end
            end
        end
    end
    
    if max_h2 != nothing && max_p2 != nothing
        a_blocks2, fock_as2 = make_blocks(ras_spaces, na, max_h2, max_p2)
        b_blocks2, fock_bs2 = make_blocks(ras_spaces, nb, max_h2, max_p2)
        for i in 1:length(a_blocks2)
            dima = binomial(ras_spaces[1], fock_as2[i][1])*binomial(ras_spaces[2], fock_as2[i][2])*binomial(ras_spaces[3], fock_as2[i][3])
            for j in 1:length(b_blocks2)
                dimb = binomial(ras_spaces[1], fock_bs2[j][1])*binomial(ras_spaces[2], fock_bs2[j][2])*binomial(ras_spaces[3], fock_bs2[j][3])
                if a_blocks2[i][1]+b_blocks2[j][1]<= max_h2
                    if a_blocks2[i][2]+b_blocks2[j][2] <= max_p2
                        block1 = RasBlock(fock_as2[i], fock_bs2[j])
                        rasvec[block1] = reshape(v[start:start+dima*dimb-1, :], dima, dimb, nroots)
                        start += dima*dimb
                    end
                end
            end
        end
    end
    return RASVector(rasvec)
end

"""
makes fock list and hp categories that are allowed based on ras spaces, number of electrons,
max holes and max particles
#Returns
- `categories`: list of Tuple(h,p) thhat are allowed
- `fock_list`: list of Tuple(ras1 ne, ras2 ne, ras3 ne) for number of electrons in each ras subspace
"""
function make_blocks(ras_spaces::Tuple{Int, Int, Int}, ne::Int, max_h::Int, max_p::Int)
    categories = []
    for h in 1:max_h+1
        holes = h-1
        for p in 1:max_p+1
            particles = p-1
            cat = (holes, particles)
            push!(categories, cat)
        end
    end
    fock_list = []
    cat_delete = []
    
    if ne < ras_spaces[1]
        start = (ne, 0, 0)
    elseif ne > ras_spaces[1]+ras_spaces[2]
        start = (ras_spaces[1], ras_spaces[2], ne-(ras_spaces[1]+ras_spaces[2]))
    else
        start = (ras_spaces[1], ne-ras_spaces[1], 0)
    end

    for i in 1:length(categories)
        fock = (start[1]-categories[i][1],ne-((start[3]+categories[i][2])+(start[1]-categories[i][1])) ,start[3]+categories[i][2])
        push!(fock_list, fock)

        if any(fock.<0)
            push!(cat_delete, i)
            continue
        end

        if fock[1]>ras_spaces[1] || fock[2]>ras_spaces[2] || fock[3]>ras_spaces[3]
            push!(cat_delete, i)
        end
    end
    
    deleteat!(fock_list, cat_delete)
    deleteat!(categories, cat_delete)
    return categories, fock_list
end

"""
need to implement this
"""
function dim(vec::RASVector)
    dim = 1
    return dim
end

"""
Local fock sector index to global RASCI space index shift.
Spin dependent shift.
"""
function shift(v::RASVector, current_fock, spin="alpha")
    if spin == "alpha"#={{{=#
        shift = 1
        for (fock, vec) in v
            if (fock[1]) == current_fock
                return shift
            else
                shift += size(vec, 1)
            end
        end
        return shift
    else
        shift = 1
        for (fock, vec) in v
            if (fock[2]) == current_fock
                return shift
            else
                shift += size(vec, 2)
            end
        end
        return shift
    end
end#=}}}=#

function get_dima(v::RASVector)
    dima = 0
    for (fock, vec) in v
        dima += size(vec, 1)
    end
    return dima
end

function get_dimb(v::RASVector)
    dimb = 0
    for (fock, vec) in v
        dimb += size(vec, 2)
    end
    return dimb
end

function apply_annihilation!(det1::Vector{Int}, det2::Vector{Int}, det3::Vector{Int}, orb_a)
    sign_a = 1#={{{=#
    if orb_a in det1
        spot = findfirst(det1.==orb_a)
        splice!(det1, spot)
        
        if spot % 2 != 1
            sign_a = -1
        end

    elseif orb_a in det2
        spot =  findfirst(det2.==orb_a)
        splice!(det2, spot)
        
        if spot % 2 != 1
            sign_a = -1
        end
        sign_a*(-1)^length(det1)
        
    elseif orb_a in det3
        spot =  findfirst(det3.==orb_a)
        splice!(det3, spot)
        
        if spot % 2 != 1
            sign_a = -1
        end
        sign_a*(-1)^length(det1)*(-1)^length(det2)
    else
        return 0, 0, 0, 0
    end

    return sign_a, det1, det2, det3#=}}}=#
end

function apply_creation!(det1::Vector{Int}, det2::Vector{Int}, det3::Vector{Int}, ras1, ras2, ras3, orb_c)
    insert_here = 1#={{{=#
    sign_c = 1
    if orb_c in ras1
        if orb_c in det1
            return 0, 0
        end

        if isempty(det1)
            det1 = [orb_c]
        else
            for i in 1:length(det1)
                if det1[i] > orb_c
                    insert_here = i
                    break
                else
                    insert_here += 1
                end
            end
            insert!(det1, insert_here, orb_c)
        end
        
        if insert_here % 2 != 1
            sign_c = -1
        end

    elseif orb_c in ras2
        if orb_c in det2
            return 0, 0
        end

        if isempty(det2)
            det2 = [orb_c]
        else
            for i in 1:length(det2)
                if det2[i] > orb_c
                    insert_here = i
                    break
                else
                    insert_here += 1
                end
            end
            insert!(det2, insert_here, orb_c)
        end
        
        if insert_here % 2 != 1
            sign_c = -1
        end
        #must count swaps from config in ras1
        sign_c*(-1)^length(det1)

    elseif orb_c in ras3
        if orb_c in det3
            return 0, 0
        end

        if isempty(det3)
            det3 = [orb_c]
        else
            for i in 1:length(det3)
                if det3[i] > orb_c
                    insert_here = i
                    break
                else
                    insert_here += 1
                end
            end
            insert!(det3, insert_here, orb_c)
        end
        if insert_here % 2 != 1
            sign_c = -1
        end
        #must count swaps from config in ras1 and ras2
        sign_c*(-1)^length(det1)*(-1)^length(det2)
    else
        return 0, 0
    end
        
    det1_c = SubspaceDeterminantString(length(ras1), length(det1), det1)
    det2_c = SubspaceDeterminantString(length(ras2), length(det2), det2.-length(ras1))
    det3_c = SubspaceDeterminantString(length(ras3), length(det3), det3.-length(ras1).-length(ras2))

    idx = calc_full_ras_index(det1_c, det2_c, det3_c)
    return sign_c, idx#=}}}=#
end

function initalize_lu(v::RASVector, no::Int)
    lua = OrderedDict{RasBlock, Array{Int,3}}()
    lub = OrderedDict{RasBlock, Array{Int,3}}()
    for (block, vec) in v.data
        lua[block] = zeros(no, no, size(vec,1))
        lub[block] = zeros(no, no, size(vec,2))
    end
    return lua, lub
end

function fill_lu(v::RASVector, ras_spaces::Tuple{Int, Int, Int})
    single_excit = make_single_excit(ras_spaces)#={{{=#
    ras1 = range(start=1, stop=ras_spaces[1])
    ras2 = range(start=ras_spaces[1]+1,stop=ras_spaces[1]+ras_spaces[2])
    ras3 = range(start=ras_spaces[1]+ras_spaces[2]+1, stop=ras_spaces[1]+ras_spaces[2]+ras_spaces[3])
    norbs = sum(ras_spaces)
    a_lu, b_lu = initalize_lu(v, norbs)

    #alpha lu table
    for (block1, vec) in v.data
        det1 = SubspaceDeterminantString(ras_spaces[1], block1.focka[1])
        for i in 1:det1.max
            det2 = SubspaceDeterminantString(ras_spaces[2], block1.focka[2])
            for j in 1:det2.max
                det3 = SubspaceDeterminantString(ras_spaces[3], block1.focka[3])
                for n in 1:det3.max
                    println("\n")
                    display(det1.config)
                    display(det2.config)
                    display(det3.config)
                    idx = calc_full_ras_index(det1, det2, det3)
                    for (se_a, delta) in single_excit
                        block2 = RasBlock(block1.focka.+delta, block1.fockb)
                        haskey(v.data, block2) || continue
                        #println("se_a: ", se_a[1], "se_c: ", se_a[2])
                        for k in se_a[1]
                            #println("det1: ", det1.config, "det2: ", det2.config.+det1.no, "det3: ", det3.config.+det1.no.+det2.no)
                            #println("k: ", k)
                            sgn_a, det1_config, det2_config, det3_config = apply_annihilation!(det1.config, det2.config.+det1.no, det3.config.+det1.no.+det2.no, k)
                            sgn_a != 0 || continue
                            for l in se_a[2]
                                if l in det1_config || l in det2_config || l in det3_config
                                    continue
                                else
                                    #println("\ndet1: ", det1_config, "det2: ", det2_config, "det3: ", det3_config)
                                    #println("l: ", l)
                                    sgn_c, idx_new = apply_creation!(det1_config, det2_config, det3_config, ras1, ras2, ras3, l)
                                    sgn_c != 0 || continue
                                    a_lu[block1][k, l, idx] = sgn_a*sgn_c*idx_new
                                end
                            end
                        end
                    end
                    incr!(det3)
                end
                incr!(det2)
            end
            incr!(det1)
        end
    end
    
    #beta lu table
    for (block1, vec) in v.data
        det1 = SubspaceDeterminantString(ras_spaces[1], block1.fockb[1])
        for i in 1:det1.max
            det2 = SubspaceDeterminantString(ras_spaces[2], block1.fockb[2])
            for j in 1:det2.max
                det3 = SubspaceDeterminantString(ras_spaces[3], block1.fockb[3])
                for n in 1:det3.max
                    idx = calc_full_ras_index(det1, det2, det3)
                    for (se_a, delta) in single_excit
                        block2 = RasBlock(block1.focka, block1.fockb.+delta)
                        haskey(v.data, block2) || continue
                        for k in se_a[1]
                            sgn_a, fock_a, det1_config, det2_config, det3_config = apply_annhilation!(det1.config, det2.config, det3.config, k)
                            sgn_a != 0 || continue
                            for l in se_a[2]
                                sgn_c, idx_new = apply_creation!(det1_config, det2_config, det3_config, ras1, ras2, ras3, l)
                                sgn_c != 0 || continue
                                b_lu[block1][k, l, idx] = sgn_a*sgn_c*idx_new
                            end
                        end
                    end
                    incr!(det3)
                end
                incr!(det2)
            end
            incr!(det1)
        end
    end#=}}}=#
    return a_lu, b_lu
end


"""
automate the types of single excitations, these will always be the same
for any rasci problem
#Returns
- `single_exc`: OrderedDict{Tuple{Vector{Int}, Vector{Int}}, Tuple{Int, Int}}() 
    keys are a tuple of annhilation orbs to creation orbs in another ras subspace
    values are the Hole-Particle change from that single excitation
"""
function make_single_excit(ras_spaces::Tuple{Int, Int, Int})
    #={{{=#
    i_orbs = range(1, ras_spaces[1])
    ii_orbs = range(ras_spaces[1]+1, ras_spaces[1]+ras_spaces[2])
    iii_orbs = range(ras_spaces[1]+ras_spaces[2]+1, ras_spaces[1]+ras_spaces[2]+ras_spaces[3])

    single_exc = OrderedDict{Tuple{Vector{Int}, Vector{Int}}, Tuple{Int, Int, Int}}()
    single_exc[(i_orbs, i_orbs)] = (0, 0, 0) #ras1->ras1
    single_exc[(ii_orbs, ii_orbs)] = (0, 0, 0) #ras2->ras2
    single_exc[(iii_orbs, iii_orbs)] = (0, 0, 0) #ras3->ras3
    single_exc[(i_orbs, ii_orbs)] = (-1, 1, 0) #ras1->ras2
    single_exc[(i_orbs, iii_orbs)] = (-1, 0, 1) #ras1->ras3
    single_exc[(ii_orbs, i_orbs)] = (1, -1, 0) #ras2->ras1
    single_exc[(ii_orbs, iii_orbs)] = (0, -1, 1) #ras2->ras3
    single_exc[(iii_orbs, i_orbs)] = (1, 0, -1) #ras3->ras1
    single_exc[(iii_orbs, ii_orbs)] = (0, 1, -1) #ras3->ras2
    return single_exc#=}}}=#
end

function initalize_sig(v::RASVector)
    sig = OrderedDict{RasBlock, Array{3,T}}()
    for (block, vec) in v
        sig[block] = zeros(size(vec))
    end
    return sig
end

"""
Sigma one (beta)
"""
function sigma_one(v::RASVector, ints::InCoreInts, ras_spaces::Tuple{Int, Int, Int}, b_lu::Dict{RasBlock, Array{3,Int}})
    sig1 = initalize_sig(v)
    single_excit = make_single_excit(ras_spaces)
    gkl = get_gkl(ints, sum(ras_spaces)) 

    for (block1, vec) in v
        for ((k_range, l_range), delta1) in single_exc
            block2 = RasBlock(block1.focka, block1.fockb+detla1)
            haskey(v,block2) || continue

            for Ib in 1:size(vec,2)
                F_kl = zeros(size(v[block2], 2))
                for k in k_range, l in l_range
                    Jb = b_lu[block1][k,l,Ib]
                    sign_kl = sign(Jb)
                    Jb = abs(Jb)
                    F_kl[Jb] += sign_kl*gkl[k,l]
                end
                @views sig_Ib = sig1[block1][:,Ib,:]
                @views C = v[block2]
                @tensor begin
                    sig_Ib[Ia, r] += F_kl[Jb]*C[Ia, Jb, r]
                end
            end

            for ((i_range, j_range), delta2) in single_exc
                block3 = RasBlock(block2.focka, block2.fockb+delta2)
                haskey(v,block3) || continue

                for Ib in 1:size(vec,2)
                    comb_kl = 0
                    comb_ij = 0

                    F_ij = zeros(size(v[block3], 2))

                    for k in k_range, l in l_range
                        comb_kl = (k-1)*no + l
                        for i in i_range, j in j_range
                            comb_ij = (i-1)*no + j
                            comb_ij >= comb_kl || continue
                            
                            Jb = lu[block1][k,l,Ib]   # Ja is local to block2
                            sign_kl = sign(Jb)
                            Jb = abs(Jb)
                            Kb = lu[block2][i,j,Jb]   # Ka is local to block3
                            sign_ij = sign(Kb)
                            Kb = abs(Kb) 

                            if comb_kl == comb_ij
                                delta = 1
                            else
                                delta = 0
                            end
                            if sign_kl == sign_ij
                                F_ij[Kb] += (ints.h2[i,j,k,l]*1/(1+delta))
                            else
                                F_ij[Kb] -= (ints.h2[i,j,k,l]*1/(1+delta))
                            end
                        end
                    end

                    # v[block2] is (block2.dima , block2.dimb, nroots)
                    # s[block1] is (block1.dima , block1.dimb, nroots)

                    @views sig_Ib = sig1[block1][:,Ib,:]
                    @views C = v[block3]
                    @tensor begin
                        sig_Ib[Ia,r] += F_ij[Jb] * C[Ia,Jb,r]
                    end
                end
            end
        end
    end
end


"""
Sigma two (alpha)
"""
function sigma_two(v::RASVector, ints::InCoreInts, ras_spaces::Tuple{Int, Int, Int}, a_lu::Dict{RasBlock, Array{3,Int}})
    sig2 = initalize_sig(v)
    single_excit = make_single_excit(ras_spaces)
    gkl = get_gkl(ints, sum(ras_spaces)) 

    for (block1, vec) in v
        for ((k_range, l_range), delta1) in single_exc

            block2 = RasBlock(block1.focka + delta1, block1.fockb)
            haskey(v,block2) || continue

            for Ia in 1:size(vec,1)
                F_kl = zeros(size(v[block2], 1))
                for k in k_range, l in l_range
                    Ja = a_lu[block1][k,l,Ia]
                    sign_kl = sign(Ja)
                    Ja = abs(Ja)
                    F_kl[Ja] += sign_kl*gkl[k,l]
                end
                @views sig_Ia = sig2[block1][Ia,:,:]
                @views C = v[block2]
                @tensor begin
                    sig_Ia[Ib, r] += F_kl[Ja]*C[Ja, Ib, r]
                end
            end

            for ((i_range, j_range), delta2) in single_exc

                block3 = RasBlock(block2.focka + delta2, block2.fockb)
                haskey(v,block3) || continue

                for Ia in 1:size(vec,1)
                    comb_kl = 0
                    comb_ij = 0

                    F_ij = zeros(size(v[block3], 1))

                    for k in k_range, l in l_range
                        comb_kl = (k-1)*no + l
                        for i in i_range, j in j_range
                            comb_ij = (i-1)*no + j
                            comb_ij >= comb_kl || continue
                            
                            Ja = lu[block1][k,l,Ia]   # Ja is local to block2
                            sign_kl = sign(Ja)
                            Ja = abs(Ja)
                            Ka = lu[block2][i,j,Ja]   # Ka is local to block3
                            sign_ij = sign(Ka)
                            Ka = abs(Ka)
                            
                            if comb_kl == comb_ij
                                delta = 1
                            else
                                delta = 0
                            end
                            if sign_kl == sign_ij
                                F_ij[Ka] += (ints.h2[i,j,k,l]*1/(1+delta))
                            else
                                F_ij[Ka] -= (ints.h2[i,j,k,l]*1/(1+delta))
                            end
                        end
                    end

                    # v[block2] is (block2.dima , block2.dimb, nroots)
                    # s[block1] is (block1.dima , block1.dimb, nroots)

                    @views sig_Ia = sig2[block1][Ia,:,:]
                    @views C = v[block3]
                    @tensor begin
                        sig_Ia[Ib,r] += F_ij[Ka] * C[Ka,Ib,r]
                    end
                end
            end
        end
    end
end
    
    #F = zeros(Float64, dima)
    #
    #for Ia in 1:dima
    #    comb_kl = 0
    #    comb_ij = 0
    #    fill!(F,0.0)
    #    for (se_a, delta_fock) in single_excit
    #        #if haskey(v, (fock[1]+delta_fock))
    #            for k in se_a[1], l in se_a[2]
    #                Ja = a_lu[][k, l, Ia]
    #                sign_a = sign(Ja)
    #                F[Ja] += sign_a*gkl[k,l]
    #                comb_kl = (k-1)*no + l
    #                for (se_a2, hpa2) in single_excit
    #                    for i in se_a2[1], j in se_a2[2]
    #                        comb_ij = (i-1)*no + j
    #                        if comb_ij < comb_kl
    #                            continue
    #                        end
    #                        Jaa = lu[i, j, Ja]
    #                        sign_aa = sign(Jaa)

    #                        if comb_kl == comb_ij
    #                            delta = 1
    #                        else
    #                            delta = 0
    #                        end
    #                        if sign_kl == sign_ij
    #                            F[Jaa] += (ints.h2[i,j,k,l]*1/(1+delta))
    #                        else
    #                            F[Jaa] -= (ints.h2[i,j,k,l]*1/(1+delta))
    #                        end

    #                        _sum_spin_pairs(sig2, C, F, Ia, hpa, hpa2)
    #                    end
    #                end
    #            end
    #        #end
    #    end
    #end
    #return sig2

"""
Sigma three is the mixed spin block (both alpha and beta single excitations)
"""
function sigma_three(v::RASVector, ints::InCoreInts, ras_spaces::Tuple{Int, Int, Int}, a_lu::Dict{RasBlock, Array{3, Int}}, b_lu::Dict{RasBlock, Array{3, Int}})
    sig3 = initalize_sig(v)
    single_excit = make_single_excit(ras_spaces)
    no = sum(ras_spaces) 
    hkl = zeros(Float64, no, no)
    nroots = 1 #this will chnage just need access to it for the contraction

    for (block1, vec) in v
        nroots = size(vec,3)
        for (se_a, delta_a) in single_excit
            for (se_b, delta_b) in single_excit
                block2 = RasBlock(block1.focka+detla_a, block1.fockb+delta_b)
                haskey(v, block2) || continue
                for aconfig in 1:size(vec, 1)
                    for k in se_a[1], l in se_a[2]
                        Ja = a_lu[block1][k, l, aconfig]
                        sign_a = sign(Ja)
                        Ja = abs(Ja)
                        hkl .= ints.h2[:,:,k,l]
                        for bconfig in 1:size(vec, 2)
                            @views sig = sig3[block1][aconfig, b_config, :]
                            for i in se_b[1], j in se_b[2]
                                Jb = b_lu[block1][i, j, bconfig]
                                sign_b = sign(Jb)
                                Jb = abs(Jb)
                                h = hkl[i,j]
                                @views v = v[block2][Ja, Jb, :]
                                sgn = sign_a*sign_b
                                for si in 1:nroots
                                    @inbounds sig[si] += sgn*h*v[si]
                                end
                            end
                        end
                    end
                end
            end
        end
    end
    return sig3
end

"""
    get_gkl(ints::InCoreInts, no::Int)

Used in sigma_one and sigma_two to get an instance of the two electron integrals
"""
function get_gkl(ints::InCoreInts, no::Int)
    hkl = zeros(no, no)#={{{=#
    hkl .= ints.h1
    gkl = zeros(no, no)
    for k in 1:no
        for l in 1:no
            gkl[k,l] += hkl[k,l]
            x = 0
            for j in 1:no
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


