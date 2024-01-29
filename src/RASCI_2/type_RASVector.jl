include("type_SubspaceDeterminantString.jl")
using OrderedCollections

struct RASVector{T}
    #data::OrderedDict{Tuple{Int, Int, Int, Int}, Array{3, T}}
    #                 Tuple{nhα, nhβ, npα, npβ}, Array{alpha, beta, roots}
    data::OrderedDict{Tuple{Tuple{Int, Int, Int}, Tuple{Int, Int, Int}}, Array{3, T}}
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

function RASVector(v::Array{T}, ras_spaces::Tuple{Int, Int, Int}, na, nb, no; max_h=0, max_p=0, max_h2=nothing, max_p2=nothing) where T
    a_blocks, fock_as = make_blocks(ras_spaces, na, max_h, max_p)
    b_blocks, fock_bs = make_blocks(ras_spaces, nb, max_h, max_p)
    rasvec = OrderedDict{Tuple{Tuple{Int, Int, Int}, Tuple{Int, Int, Int}}, Array{3, T}}()
    nroots = size(v, 2)
    
    start = 1
    for i in 1:length(a_blocks)
        dima = binomial(ras_spaces[1], fock_as[i][1])*binomial(ras_spaces[2], fock_as[i][2])*binomial(ras_spaces[3], fock_as[i][3])
        for j in 1:length(b_blocks)
            dimb = binomial(ras_spaces[1], fock_bs[j][1])*binomial(ras_spaces[2], fock_bs[j][2])*binomial(ras_spaces[3], fock_bs[j][3])
            if a_blocks[i][1]+b_blocks[j][1]<= max_h
                if a_blocks[i][2]+b_blocks[j][2] <= max_p
                    rasvec[(fock_as[i], fock_bs[j])] = reshape(v[start:start+dima*dimb-1, :], dima, dimb, nroots)
                    start += dima*dimb
                end
            end
        end
    end
    
    if max_h2 && max_p2 != nothing
        a_blocks2, fock_as2 = make_blocks(ras_spaces, na, max_h2, max_p2)
        b_blocks2, fock_bs2 = make_blocks(ras_spaces, nb, max_h2, max_p2)
        for i in 1:length(a_blocks2)
            dima = binomial(ras_spaces[1], fock_as2[i][1])*binomial(ras_spaces[2], fock_as2[i][2])*binomial(ras_spaces[3], fock_as2[i][3])
            for j in 1:length(b_blocks2)
                dimb = binomial(ras_spaces[1], fock_bs2[j][1])*binomial(ras_spaces[2], fock_bs2[j][2])*binomial(ras_spaces[3], fock_bs2[j][3])
                if a_blocks2[i][1]+b_blocks2[j][1]<= max_h2
                    if a_blocks2[i][2]+b_blocks2[j][2] <= max_p2
                        rasvec[(fock_as2[i], fock_bs2[j])] = reshape(v[start:start+dima*dimb-1, :], dima, dimb, nroots)
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

function fill_lu(v::RASVector, ras_spaces::Tuple{Int, Int, Int})
    single_excit = make_single_excit(ras_spaces)#={{{=#
    ras1 = range(start=1, stop=ras_spaces[1])
    ras2 = range(start=ras_spaces[1]+1,stop=ras_spaces[1]+ras_spaces[2])
    ras3 = range(start=ras_spaces[1]+ras_spaces[2]+1, stop=ras_spaces[1]+ras_spaces[2]+ras_spaces[3])
    a_lu = []
    b_lu = []
    norbs = sum(ras_spaces)

    #alpha lu table
    for (fock, vec) in v
        lu = zeros(Int, (norbs, norbs, size(vec,1))) #this doesnt have to be full norbs but can't figure out how to initalize
        det1 = SubspaceDeterminantString(ras_spaces[1], fock[1][1])
        for i in 1:det1.max
            det2 = SubspaceDeterminantString(ras_spaces[2], fock[1][2])
            for j in 1:det2.max
                det3 = SubspaceDeterminantString(ras_spaces[3], fock[1][3])
                for n in 1:det3.max
                    idx = calc_full_ras_index(det1, det2, det3)
                    for (se_a, delta_fock) in single_excit
                        if haskey(v, (fock[1]+delta_fock)) #this need to look over all other beta fock sectors
                            for k in se_a[1]
                                for l in se_a[2]
                                    sgn_a, fock_a, det1_config, det2_config, det3_config = apply_annhilation!(det1.config, det2.config, det3.config, k)
                                    sgn_a != 0 || continue
                                    sgn_c, idx_new = apply_creation!(det1_config, det2_config, det3_config, ras1, ras2, ras3, l)
                                    sgn_c != 0 || continue
                                    lu[k, l, idx] = sgn_a*sgn_c*idx_new
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
        push!(a_lu, lu)
    end
    
    #beta lu table
    for (fock, vec) in v
        lu = zeros(Int, (norbs, norbs, size(vec,2))) #this doesnt have to be full norbs but can't figure out how to initalize
        det1 = SubspaceDeterminantString(ras_spaces[1], fock[2][1])
        for i in 1:det1.max
            det2 = SubspaceDeterminantString(ras_spaces[2], fock[2][2])
            for j in 1:det2.max
                det3 = SubspaceDeterminantString(ras_spaces[3], fock[2][3])
                for n in 1:det3.max
                    idx = calc_full_ras_index(det1, det2, det3)
                    for (se_a, delta_fock) in single_excit
                        if haskey(v, (fock[2]+delta_fock), spin="beta") #this needs to look over all other alpha fock sectors
                            for k in se_a[1]
                                for l in se_a[2]
                                    sgn_a, fock_a, det1_config, det2_config, det3_config = apply_annhilation!(det1.config, det2.config, det3.config, k)
                                    sgn_a != 0 || continue
                                    sgn_c, idx_new = apply_creation!(det1_config, det2_config, det3_config, ras1, ras2, ras3, l)
                                    sgn_c != 0 || continue
                                    lu[k, l, idx] = sgn_a*sgn_c*idx_new
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
        push!(b_lu, lu)
    end#=}}}=#
    return a_lu, b_lu
end

function haskey(v::RASVector, fock_curr::Tuple{Int, Int, Int}; spin="alpha")
    if spin == "alpha"
        for (fock, vec) in v
            if (fock_curr, fock[2]) == fock
                return true
            else
                continue
            end
        end
    else
        for (fock, vec) in v
            if (fock[1], fock_curr) == fock
                return true
            else
                conitnue
            end
        end
    end
    return false
end


"""
need to code this
"""
function _sum_spin_pairs!(sig::RASVector, v::RASVector, F::Array{Float}, I::Int, hp_s::Tuple{Int, Int}, hp_d::Tuple{Int, Int})
end

function apply_annhiliation!(det1::Vector{Int}, det2::Vector{Int}, det3::Vector{Int}, orb_a)
    sign_a = 1#={{{=#
    if orb_a in det1
        spot =  findfirst(det1==orb_a)
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
        return 0, 0
    end

    return sign_a#=}}}=#
end

function apply_creation!(det1::Vector{Int}, det2::Vector{Int}, det3::Vector{Int}, ras1, ras2, ras3, orb_c, current_fock, v::RASVector)
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
        
    det1_c = SubspaceDeterminantString(length(ras1), length(det1_c), det1)
    det2_c = SubspaceDeterminantString(length(ras2), length(det2_c), det2)
    det3_c = SubspaceDeterminantString(length(ras3), length(det3_c), det3)

    idx = calc_full_ras_index(det1_c, det2_c, det3_c)
    return sign_c, idx#=}}}=#
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
    i_orbs = [1:1:ras_spaces[1];]#={{{=#
    start2 = ras_spaces[1]+1
    end2 = start2+ras_spaces[2]-1
    ii_orbs = [start2:1:end2;]
    start = norbs-ras_spaces[3]+1
    iii_orbs = [start:1:norbs;]

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
    
    #single_exc[(i_orbs, i_orbs)] = (0, 0) #ras1->ras1
    #single_exc[(ii_orbs, ii_orbs)] = (0, 0) #ras2->ras2
    #single_exc[(iii_orbs, iii_orbs)] = (0, 0) #ras3->ras3
    #single_exc[(i_orbs, ii_orbs)] = (1, 0) #ras1->ras2
    #single_exc[(i_orbs, iii_orbs)] = (1, 1) #ras1->ras3
    #single_exc[(ii_orbs, i_orbs)] = (-1, 0) #ras2->ras1
    #single_exc[(ii_orbs, iii_orbs)] = (0, 1) #ras2->ras3
    #single_exc[(iii_orbs, i_orbs)] = (-1, -1) #ras3->ras1
    #single_exc[(iii_orbs, ii_orbs)] = (0, -1) #ras3->ras2
    return single_exc#=}}}=#
end

"""
Sigma two (alpha)
"""
function sigma_two(v::RASVector, ints::InCoreInts, ras_spaces::Tuple{Int, Int, Int})
    sig2 = RASVector(v.data, zeros{3, T}()) #NEED TO FIGURE OUT HOW TO INITALIZE SIG
    single_excit = make_single_excit(ras_spaces)
    dima = get_dima(v)
    gkl = get_gkl(ints, prob) 
    
    F = zeros(Float64, dima)
    for Ia in 1:dima
        comb_kl = 0
        comb_ij = 0
        fill!(F,0.0)
        for (se_a, delta_fock) in single_excit
            #if haskey(v, (fock[1]+delta_fock))
                for k in se_a[1], l in se_a[2]
                    Ja = a_lu[][k, l, Ia]
                    sign_a = sign(Ja)
                    F[Ja] += sign_a*gkl[k,l]
                    comb_kl = (k-1)*no + l
                    for (se_a2, hpa2) in single_excit
                        for i in se_a2[1], j in se_a2[2]
                            comb_ij = (i-1)*no + j
                            if comb_ij < comb_kl
                                continue
                            end
                            Jaa = lu[i, j, Ja]
                            sign_aa = sign(Jaa)

                            if comb_kl == comb_ij
                                delta = 1
                            else
                                delta = 0
                            end
                            if sign_kl == sign_ij
                                F[Jaa] += (ints.h2[i,j,k,l]*1/(1+delta))
                            else
                                F[Jaa] -= (ints.h2[i,j,k,l]*1/(1+delta))
                            end

                            _sum_spin_pairs(sig2, C, F, Ia, hpa, hpa2)
                        end
                    end
                end
            #end
        end
    end
    return sig2
end

"""
Sigma three is the mixed spin block (both alpha and beta single excitations)
"""
function sigma_three(v::RASVector, ints::InCoreInts, ras_spaces::Tuple{Int, Int, Int}, no::Int, nroots::Int)
    sig3 = RASVector(RASVector.data, zeros{3, T}())
    single_excit = make_single_excit(ras_spaces)
    
    hkl = zeros(Float64, no, no)

    for (fock, vec) in v
        for (se_a, delta_a) in single_excit
            for (se_b, delta_b) in single_excit
                new_fock = (fock[1] + delta_a, fock[2]+delta_b)
                if haskey(v, new_fock)
                    for aconfig in 1:size(vec, 1)
                        for k in se_a[1], l in se_a[2]
                            Ja = lu[k, l, aconfig]
                            sign_a = sign(Ja)
                            hkl .= ints.h2[:,:,k,l]
                            for bconfig in 1:size(vec, 2)
                                @views sig = sig3[fock][aconfig, b_config, :]
                                for i in se_b[1], j in se_b[2]
                                    Jb = lu[i, j, bconfig]
                                    sign_b = sign(Jb)
                                    h = hkl[i,j]
                                    @views v = v[new_fock][Ja, Jb, :]
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


