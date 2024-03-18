using ActiveSpaceSolvers
using OrderedCollections
using TensorOperations
using InCoreIntegrals
using StaticArrays
    
struct RasBlock
    focka::Tuple{Int, Int, Int}
    fockb::Tuple{Int, Int, Int}
end

struct RASVector{T}
    data::OrderedDict{ActiveSpaceSolvers.RASCI_2.RasBlock, Array{T,3}}
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

function RASVector(v, prob::RASCIAnsatz_2)
    a_blocks, fock_as = make_blocks(prob.ras_spaces, prob.na, prob.max_h, prob.max_p)#={{{=#
    b_blocks, fock_bs = make_blocks(prob.ras_spaces, prob.nb, prob.max_h, prob.max_p)
    rasvec = OrderedDict{ActiveSpaceSolvers.RASCI_2.RasBlock, Array{Float64,3}}()
    nroots = size(v, 2)
    
    start = 1
    for i in 1:length(a_blocks)
        dima = binomial(prob.ras_spaces[1], fock_as[i][1])*binomial(prob.ras_spaces[2], fock_as[i][2])*binomial(prob.ras_spaces[3], fock_as[i][3])
        for j in 1:length(b_blocks)
            dimb = binomial(prob.ras_spaces[1], fock_bs[j][1])*binomial(prob.ras_spaces[2], fock_bs[j][2])*binomial(prob.ras_spaces[3], fock_bs[j][3])
            if a_blocks[i][1]+b_blocks[j][1]<= prob.max_h
                if a_blocks[i][2]+b_blocks[j][2] <= prob.max_p
                    block1 = RasBlock(fock_as[i], fock_bs[j])
                    rasvec[block1] = reshape(v[start:start+dima*dimb-1, :], dima, dimb, nroots)
                    start += dima*dimb
                end
            end
        end
    end
    
    if prob.max_h2 != 0 && prob.max_p2 != 0
        a_blocks2, fock_as2 = make_blocks(prob.ras_spaces, prob.na, prob.max_h2, prob.max_p2)
        b_blocks2, fock_bs2 = make_blocks(prob.ras_spaces, prob.nb, prob.max_h2, prob.max_p2)
        for i in 1:length(a_blocks2)
            dima = binomial(prob.ras_spaces[1], fock_as2[i][1])*binomial(prob.ras_spaces[2], fock_as2[i][2])*binomial(prob.ras_spaces[3], fock_as2[i][3])
            for j in 1:length(b_blocks2)
                dimb = binomial(prob.ras_spaces[1], fock_bs2[j][1])*binomial(prob.ras_spaces[2], fock_bs2[j][2])*binomial(prob.ras_spaces[3], fock_bs2[j][3])
                if a_blocks2[i][1]+b_blocks2[j][1]<= prob.max_h2
                    if a_blocks2[i][2]+b_blocks2[j][2] <= prob.max_p2
                        block1 = RasBlock(fock_as2[i], fock_bs2[j])
                        rasvec[block1] = reshape(v[start:start+dima*dimb-1, :], dima, dimb, nroots)
                        start += dima*dimb
                    end
                end
            end
        end
    end
    return RASVector(rasvec)
end#=}}}=#

"""
makes fock list and hp categories that are allowed based on ras spaces, number of electrons,
max holes and max particles
#Returns
- `categories`: list of Tuple(h,p) thhat are allowed
- `fock_list`: list of Tuple(ras1 ne, ras2 ne, ras3 ne) for number of electrons in each ras subspace
"""
function make_blocks(ras_spaces::SVector{3, Int}, ne::Int, max_h::Int8, max_p::Int8)
    categories = []#={{{=#
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
end#=}}}=#

function get_dim(v::RASVector)
    dim = 0#={{{=#
    for (block, vec) in v.data
        dim += size(vec,1)*size(vec,2)
    end
    return dim
end#=}}}=#

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
    dima = 0#={{{=#
    tmp = []
    for (fock, vec) in v.data
        if fock.focka in tmp
            continue
        else
            push!(tmp, fock.focka)
            dima += size(vec, 1)
        end
    end
    return dima#=}}}=#
end

function get_dimb(v::RASVector)
    dimb = 0#={{{=#
    tmp = []
    for (fock, vec) in v.data
        if fock.fockb in tmp
            continue
        else
            push!(tmp, fock.fockb)
            dimb += size(vec, 2)
        end
    end
    return dimb#=}}}=#
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
        sign_a = sign_a*(-1)^length(det1)
        
    elseif orb_a in det3
        spot =  findfirst(det3.==orb_a)
        splice!(det3, spot)
        
        if spot % 2 != 1
            sign_a = -1
        end
        sign_a = sign_a*(-1)^length(det1)*(-1)^length(det2)
    else
        return 0, 0, 0, 0
    end

    return sign_a, det1, det2, det3#=}}}=#
end

function apply_creation!(det1::Vector{Int}, det2::Vector{Int}, det3::Vector{Int}, ras1, ras2, ras3, orb_c)
    if orb_c in ras1#={{{=#
        insert_here = 1
        sign_c = 1
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
        det1_c = SubspaceDeterminantString(length(ras1), length(det1), det1)
        det2_c = SubspaceDeterminantString(length(ras2), length(det2), det2.-length(ras1))
        det3_c = SubspaceDeterminantString(length(ras3), length(det3), det3.-length(ras1).-length(ras2))

        idx = calc_full_ras_index(det1_c, det2_c, det3_c)
        return sign_c, idx

    elseif orb_c in ras2
        insert_here = 1
        sign_c = 1
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
        sign_c = sign_c*(-1)^length(det1)
        
        det1_c = SubspaceDeterminantString(length(ras1), length(det1), det1)
        det2_c = SubspaceDeterminantString(length(ras2), length(det2), det2.-length(ras1))
        det3_c = SubspaceDeterminantString(length(ras3), length(det3), det3.-length(ras1).-length(ras2))

        idx = calc_full_ras_index(det1_c, det2_c, det3_c)
        return sign_c, idx

    elseif orb_c in ras3
        insert_here = 1
        sign_c = 1
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
        sign_c = sign_c*(-1)^length(det1)*(-1)^length(det2)
        
        det1_c = SubspaceDeterminantString(length(ras1), length(det1), det1)
        det2_c = SubspaceDeterminantString(length(ras2), length(det2), det2.-length(ras1))
        det3_c = SubspaceDeterminantString(length(ras3), length(det3), det3.-length(ras1).-length(ras2))
        idx = calc_full_ras_index(det1_c, det2_c, det3_c)
        return sign_c, idx

    else
        return 0, 0
    end#=}}}=#
end

function initalize_lu(v::RASVector, no::Int)
    lu = Dict{Tuple{Int, Int, Int}, Array{Int,3}}()#={{{=#
    for (block, vec) in v.data
        if haskey(lu, block.focka) == false
            lu[block.focka] = zeros(no, no, size(vec,1))
        end
        if haskey(lu, block.fockb) == false
            lu[block.fockb] = zeros(no, no, size(vec,2))
        end
    end
    return lu
end#=}}}=#

function fill_lu(v::RASVector, ras_spaces::SVector{3, Int})
    single_excit = make_single_excit(ras_spaces)#={{{=#
    ras1 = range(start=1, stop=ras_spaces[1])
    ras2 = range(start=ras_spaces[1]+1,stop=ras_spaces[1]+ras_spaces[2])
    ras3 = range(start=ras_spaces[1]+ras_spaces[2]+1, stop=ras_spaces[1]+ras_spaces[2]+ras_spaces[3])
    norbs = sum(ras_spaces)
    lu = initalize_lu(v, norbs)

    #general lu table
    for (fock1, lu_data) in lu
        for ((k_range, l_range), delta) in single_excit
            fock2 = fock1.+delta
            haskey(lu, fock2) || continue
            
            #conf_scr1_2 = zeros(Int, fock2[1])
            #conf_scr2_2 = zeros(Int, fock2[2])
            #conf_scr3_2 = zeros(Int, fock2[3])

            det1 = SubspaceDeterminantString(ras_spaces[1], fock1[1])
            for i in 1:det1.max
                det2 = SubspaceDeterminantString(ras_spaces[2], fock1[2])
                for j in 1:det2.max
                    det3 = SubspaceDeterminantString(ras_spaces[3], fock1[3])
                    for n in 1:det3.max
                        idx = calc_full_ras_index(det1, det2, det3)
                        #println(det1.config, det2.config.+det1.no, det3.config.+det1.no.+det2.no)
                        #display(idx)

                        for k in k_range
                            tmp_det1 = deepcopy(det1.config)
                            tmp_det2 = deepcopy(det2.config)
                            tmp_det3 = deepcopy(det3.config)

                            sgn_a, det1_config, det2_config, det3_config = apply_annihilation!(tmp_det1, tmp_det2.+det1.no, tmp_det3.+det1.no.+det2.no, k)
                            
                            sgn_a != 0 || continue
                            for l in l_range
                                if l in det1_config || l in det2_config || l in det3_config
                                    continue
                                else
                                    sgn_c, idx_new = apply_creation!(det1_config, det2_config, det3_config, ras1, ras2, ras3, l)
                                    sgn_c != 0 || continue
                                    lu[fock1][k, l, idx] = sgn_a*sgn_c*idx_new
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
    end

    return lu
end#=}}}=#

function get_configs(ras_spaces, fock)
    tmp = []#={{{=#
    det1 = SubspaceDeterminantString(ras_spaces[1], fock[1])
    for i in 1:det1.max
        det2 = SubspaceDeterminantString(ras_spaces[2], fock[2])
        for j in 1:det2.max
            det3 = SubspaceDeterminantString(ras_spaces[3], fock[3])
            for n in 1:det3.max
                idx = calc_full_ras_index(det1, det2, det3)
                config = [det1.config;det2.config.+det1.no;det3.config.+det1.no.+det2.no]
                push!(tmp, config)
                incr!(det3)
            end
            incr!(det2)
        end
        incr!(det1)
    end
    return tmp#=}}}=#
end




"""
automate the types of single excitations, these will always be the same
for any rasci problem
#Returns
- `single_exc`: OrderedDict{Tuple{Vector{Int}, Vector{Int}}, Tuple{Int, Int}}() 
    keys are a tuple of annhilation orbs to creation orbs in another ras subspace
    values are the Hole-Particle change from that single excitation
"""
function make_single_excit(ras_spaces::SVector{3, Int})
    #={{{=#
#i_orbs = range(1, ras_spaces[1])
    #ii_orbs = range(ras_spaces[1]+1, ras_spaces[1]+ras_spaces[2])
    #iii_orbs = range(ras_spaces[1]+ras_spaces[2]+1, ras_spaces[1]+ras_spaces[2]+ras_spaces[3])

    i_orbs = range(start=1, stop=ras_spaces[1])
    ii_orbs = range(start=ras_spaces[1]+1,stop=ras_spaces[1]+ras_spaces[2])
    iii_orbs = range(start=ras_spaces[1]+ras_spaces[2]+1, stop=ras_spaces[1]+ras_spaces[2]+ras_spaces[3])

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
    sig = OrderedDict{RasBlock, Array{Float64, 3}}()#={{{=#
    for (block, vec) in v.data
        sig[block] = zeros(size(vec))
    end
    return sig
end#=}}}=#

function get_other_spin_blocks(fock::Tuple{Int,Int,Int}, v::RASVector; spin=1)
    tmp = []#={{{=#
     
    if spin == 1 #alpha, need ot find beta blocks
        for (block, vec) in v.data
            if block.focka == fock
                push!(tmp, block)
            else
                continue
            end
        end
    else
        for (block, vec) in v.data
            if block.fockb == fock
                push!(tmp, block)
            else
                continue
            end
        end
    end
    return tmp
end#=}}}=#

function get_fockas(v::RASVector)
    tmp = []#={{{=#
    for (block, vec) in v.data
        push!(tmp, block.focka)
    end
    return tmp
end#=}}}=#

function get_dim_fock(fock::Tuple{Int,Int,Int}, ras_spaces::SVector{3, Int})
    dim1 = get_nchk(ras_spaces[1],fock[1])#={{{=#
    dim2 = get_nchk(ras_spaces[2],fock[2])
    dim3 = get_nchk(ras_spaces[3],fock[3])
    return dim1*dim2*dim3
end#=}}}=#

"""
Sigma one (beta)
"""
function sigma_one(v::RASVector, ints::InCoreInts, ras_spaces::SVector{3, Int}, lu::Dict{Tuple{Int,Int,Int}, Array{Int,3}}) where T
    sig1 = initalize_sig(v)#={{{=#
    single_excit = make_single_excit(ras_spaces)
    gkl = get_gkl(ints, sum(ras_spaces)) 
    no = sum(ras_spaces)
    nroots =1
    
    for (block1, vec) in v.data
        nroots = size(vec,3)
        for ((k_range, l_range), delta1) in single_excit
            block2 = RasBlock(block1.focka, block1.fockb.+delta1)
            haskey(v.data,block2) || continue
            F = zeros(size(v.data[block2], 2))
            for Ib in 1:size(vec,2)
                fill!(F, 0.0)
                for l in l_range, k in k_range  
                    Jb = lu[block1.fockb][k,l,Ib]
                    Jb != 0 || continue
                    sign_kl = sign(Jb)
                    Jb = abs(Jb)
                    F[Jb] += sign_kl*gkl[l,k]
                    comb_kl = (l-1)*no + k
                    for ((i_range, j_range), delta2) in single_excit
                        block3 = RasBlock(block1.focka, block1.fockb.+delta1.+delta2)
                        haskey(v.data,block3) || continue
                        #when block3 == block2 can do double excitations and contract with same F array
                        if block3 == block2
                            for j in j_range, i in i_range
                                comb_ij = (j-1)*no + i
                                comb_ij >= comb_kl || continue

                                Kb = lu[block1.fockb.+delta1][i,j,Jb]   # Ka is local to block3
                                Kb != 0 || continue
                                sign_ij = sign(Kb)
                                Kb = abs(Kb)

                                if comb_kl == comb_ij
                                    delta = 1
                                else
                                    delta = 0
                                end
                                if sign_kl == sign_ij
                                    F[Kb] += (ints.h2[j,i,l,k]*1/(1+delta))
                                else
                                    F[Kb] -= (ints.h2[j,i,l,k]*1/(1+delta))
                                end
                            end
                        end
                    end
                end

                @views sig_Ib = sig1[block1][:,Ib,:]
                @views C = v.data[block2]
                @tensor begin
                    sig_Ib[Ia, r] += F[Jb]*C[Ia, Jb, r]
                end
            end
            
            #collecting double excitations where block2 != block3
            #don't need to compute single excitations bec all were found in above block
            for ((i_range, j_range), delta2) in single_excit
                block3 = RasBlock(block1.focka, block1.fockb.+delta1.+delta2)
                haskey(v.data,block3) || continue
                block3 != block2 || continue
                F_ij = zeros(size(v.data[block3], 2))
                for Ib in 1:size(vec,2)
                    fill!(F_ij, 0.0)
                    for l in l_range, k in k_range
                        Jb = lu[block1.fockb][k,l,Ib]
                        Jb != 0 || continue
                        sign_kl = sign(Jb)
                        Jb = abs(Jb)
                        comb_kl = (l-1)*no + k
                        for j in j_range, i in i_range
                            comb_ij = (j-1)*no + i
                            comb_ij >= comb_kl || continue

                            Kb = lu[block1.fockb.+delta1][i,j,Jb]   # Kb is local to block3
                            Kb != 0 || continue
                            sign_ij = sign(Kb)
                            Kb = abs(Kb)

                            if comb_kl == comb_ij
                                delta = 1
                            else
                                delta = 0
                            end
                            if sign_kl == sign_ij
                                F_ij[Kb] += (ints.h2[j,i,l,k]*1/(1+delta))
                            else
                                F_ij[Kb] -= (ints.h2[j,i,l,k]*1/(1+delta))
                            end
                        end
                    end
                    
                    @views sig_Ib = sig1[block1][:,Ib,:]
                    @views C = v.data[block3]
                    @tensor begin
                        sig_Ib[Ia, r] += F_ij[Jb]*C[Ia, Jb, r]
                    end
                end
            end
        end
    end

    starti = 1
    dim = get_dim(v)
    sig = zeros(Float64, dim, nroots)
    for (block, vec) in sig1
        tmp = reshape(vec, (size(vec,1)*size(vec,2), nroots))
        sig[starti:starti+(size(vec,1)*size(vec,2))-1, :] .= tmp
        starti += (size(vec,1)*size(vec,2))
    end
    return sig
end#=}}}=#
    
"""
Sigma two (alpha)
"""
function sigma_two(v::RASVector, ints::InCoreInts, ras_spaces::SVector{3, Int}, lu::Dict{Tuple{Int,Int,Int}, Array{Int,3}})
    sig2 = initalize_sig(v)#={{{=#
    single_excit = make_single_excit(ras_spaces)
    gkl = get_gkl(ints, sum(ras_spaces)) 
    no = sum(ras_spaces)
    nroots =1

    for (block1, vec) in v.data
        nroots = size(vec,3)
        for ((k_range, l_range), delta1) in single_excit
            block2 = RasBlock(block1.focka.+delta1, block1.fockb)
            haskey(v.data,block2) || continue
            F = zeros(size(v.data[block2], 1))
            for Ia in 1:size(vec,1)
                fill!(F, 0.0)
                for l in l_range, k in k_range  
                    Ja = lu[block1.focka][k,l,Ia]
                    Ja != 0 || continue
                    sign_kl = sign(Ja)
                    Ja = abs(Ja)
                    F[Ja] += sign_kl*gkl[l,k]
                    comb_kl = (l-1)*no + k
                    tmp = deepcopy(F)
                    for ((i_range, j_range), delta2) in single_excit
                        block3 = RasBlock(block1.focka.+delta1.+delta2, block1.fockb)
                        haskey(v.data,block3) || continue
                        #when block3 == block2 can do double excitations and contract with same F array
                        if block3 == block2
                            for j in j_range, i in i_range
                                comb_ij = (j-1)*no + i
                                comb_ij >= comb_kl || continue

                                Ka = lu[block1.focka.+delta1][i,j,Ja]   # Ka is local to block3
                                Ka != 0 || continue
                                sign_ij = sign(Ka)
                                Ka = abs(Ka)

                                if comb_kl == comb_ij
                                    delta = 1
                                else
                                    delta = 0
                                end
                                if sign_kl == sign_ij
                                    F[Ka] += (ints.h2[j,i,l,k]*1/(1+delta))
                                else
                                    F[Ka] -= (ints.h2[j,i,l,k]*1/(1+delta))
                                end
                            end
                        end
                    end
                end

                @views sig_Ia = sig2[block1][Ia,:,:]
                @views C = v.data[block2]
                @tensor begin
                    sig_Ia[Ib, r] += F[Ja]*C[Ja, Ib, r]
                end
            end
            
            #collecting double excitations where block2 != block3
            #don't need to compute single excitations bec all were found in above block
            for ((i_range, j_range), delta2) in single_excit
                block3 = RasBlock(block1.focka.+delta1.+delta2, block1.fockb)
                haskey(v.data,block3) || continue
                block3 != block2 || continue
                F_ij = zeros(size(v.data[block3], 1))
                for Ia in 1:size(vec,1)
                    fill!(F_ij, 0.0)
                    for l in l_range, k in k_range
                        Ja = lu[block1.focka][k,l,Ia]
                        Ja != 0 || continue
                        sign_kl = sign(Ja)
                        Ja = abs(Ja)
                        comb_kl = (l-1)*no + k
                        for j in j_range, i in i_range
                            comb_ij = (j-1)*no + i
                            comb_ij >= comb_kl || continue

                            Ka = lu[block1.focka.+delta1][i,j,Ja]   # Ka is local to block3
                            Ka != 0 || continue
                            sign_ij = sign(Ka)
                            Ka = abs(Ka)

                            if comb_kl == comb_ij
                                delta = 1
                            else
                                delta = 0
                            end
                            if sign_kl == sign_ij
                                F_ij[Ka] += (ints.h2[j,i,l,k]*1/(1+delta))
                            else
                                F_ij[Ka] -= (ints.h2[j,i,l,k]*1/(1+delta))
                            end
                        end
                    end
                    
                    @views sig_Ia = sig2[block1][Ia,:,:]
                    @views C = v.data[block3]
                    @tensor begin
                        sig_Ia[Ib, r] += F_ij[Ja]*C[Ja, Ib, r]
                    end
                end
            end
        end
    end

    starti = 1
    dim = get_dim(v)
    sig = zeros(Float64, dim, nroots)
    for (block, vec) in sig2
        tmp = reshape(vec, (size(vec,1)*size(vec,2), nroots))
        sig[starti:starti+(size(vec,1)*size(vec,2))-1, :] .= tmp
        starti += (size(vec,1)*size(vec,2))
    end
    return sig
end#=}}}=#

"""
Sigma three is the mixed spin block (both alpha and beta single excitations)
"""
function sigma_three(v::RASVector, ints::InCoreInts, ras_spaces::SVector{3, Int}, lu::Dict{Tuple{Int,Int,Int}, Array{Int,3}})
    sig3 = initalize_sig(v)#={{{=#
    single_excit = make_single_excit(ras_spaces)
    no = sum(ras_spaces) 
    hkl = zeros(Float64, no, no)
    nroots = 1 #this will chnage just need access to it for the contraction

    for (block1, vec) in v.data
        nroots = size(vec,3)
        for ((k_range, l_range), delta_a) in single_excit
            for ((i_range, j_range), delta_b) in single_excit
                block2 = RasBlock(block1.focka.+delta_a, block1.fockb.+delta_b)
                haskey(v.data, block2) || continue
                for aconfig in 1:size(vec, 1)
                    for l in l_range, k in k_range
                        Ja = lu[block1.focka][k, l, aconfig]
                        Ja != 0 || continue     
                        sign_a = sign(Ja)
                        Ja = abs(Ja)
                        hkl .= ints.h2[:,:,l,k]
                        for bconfig in 1:size(vec, 2)
                            @views sig = sig3[block1][aconfig, bconfig, :]
                            for j in j_range, i in i_range
                                Jb = lu[block1.fockb][i, j, bconfig]
                                Jb != 0 || continue     
                                sign_b = sign(Jb)
                                Jb = abs(Jb)
                                h = hkl[j,i]
                                @views v2 = v.data[block2][Ja, Jb, :]
                                sgn = sign_a*sign_b
                                for si in 1:nroots
                                    @inbounds sig[si] += sgn*h*v2[si]
                                end
                            end
                        end
                    end
                end
            end
        end
    end

    
    starti = 1
    dim = get_dim(v)
    sig = zeros(Float64, dim, nroots)
    for (block, vec) in sig3
        tmp = reshape(vec, (size(vec,1)*size(vec,2), nroots))
        sig[starti:starti+(size(vec,1)*size(vec,2))-1, :] .= tmp
        starti += (size(vec,1)*size(vec,2))
    end
    return sig
end#=}}}=#

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

function get_fock_delta(orb_a::Int, orb_c::Int, ras_spaces::SVector{3, Int})
    i_orbs = range(start=1, stop=ras_spaces[1])#={{{=#
    ii_orbs = range(start=ras_spaces[1]+1,stop=ras_spaces[1]+ras_spaces[2])
    iii_orbs = range(start=ras_spaces[1]+ras_spaces[2]+1, stop=ras_spaces[1]+ras_spaces[2]+ras_spaces[3])
    
    delta = (0,0,0)
    
    if orb_a in i_orbs && orb_c in ii_orbs
        delta = (-1, 1, 0) #ras1->ras2
        return delta
    elseif orb_a in i_orbs && orb_c in iii_orbs
        delta = (-1, 0, 1) #ras1->ras3
        return delta
    elseif orb_a in ii_orbs && orb_c in i_orbs
        delta = (1, -1, 0) #ras2->ras1
        return delta
    elseif orb_a in ii_orbs && orb_c in iii_orbs
        delta = (0, -1, 1) #ras2->ras3
        return delta
    elseif orb_a in iii_orbs && orb_c in i_orbs
        delta = (1, 0, -1) #ras3->ras1
        return delta
    elseif orb_a in iii_orbs && orb_c in ii_orbs
        delta = (0, 1, -1) #ras3->ras2
        return delta
    end
    return delta
end#=}}}=#

function compute_S2_expval(C::Matrix, P::RASCIAnsatz_2)
    ###{{{
    #S2 = (S+S- + S-S+)1/2 + Sz.Sz
    #   = 1/2 sum_ij(ai'bi bj'ai + bj'aj ai'bi) + Sz.Sz
    #   do swaps and you can end up adding the two together to get rid
    #   of the 1/2 factor so 
    #   = (-1) sum_ij(ai'aj|alpha>bj'bi|beta> + Sz.Sz
    ###

    nr = size(C,2)
    s2 = zeros(nr)
    v = RASVector(C, P)
    lu = fill_lu(v, P.ras_spaces)
    v = v.data

    for (block1, vec) in v
        as = get_configs(P.ras_spaces, block1.focka)
        bs = get_configs(P.ras_spaces, block1.fockb)
        for Ia in 1:length(as)
            config_a = as[Ia]
            for Ib in 1:length(bs)
                config_b = bs[Ib]
                #Sz.Sz (α) 
                count_a = (P.na-1)*P.na
                for i in 1:count_a
                    for r in 1:nr
                        s2[r] += 0.25*vec[Ia, Ib, r]*vec[Ia, Ib, r]
                    end
                end

                #Sz.Sz (β)
                count_b = (P.nb-1)*P.nb
                for i in 1:count_b
                    for r in 1:nr
                        s2[r] += 0.25*vec[Ia, Ib, r]*vec[Ia, Ib, r]
                    end
                end

                #Sz.Sz (α,β)
                for ai in config_a
                    for bj in config_b
                        if ai != bj
                            for r in 1:nr
                                s2[r] -= .5 * vec[Ia, Ib, r]*vec[Ia, Ib, r] 
                            end
                        end
                    end
                end

                ##Sp.Sm + Sm.Sp Diagonal Part
                for ai in config_a
                    if ai in config_b
                    else
                        for r in 1:nr
                            s2[r] += .75 * vec[Ia, Ib, r]*vec[Ia, Ib, r] 
                        end
                    end
                end

                for bi in config_b
                    if bi in config_a
                    else
                        for r in 1:nr
                            s2[r] += .75 * vec[Ia, Ib, r]*vec[Ia, Ib, r] 
                        end
                    end
                end

                #(Sp.Sm + Sm.Sp)1/2 Off Diagonal Part
                for ai in config_a
                    for bj in config_b
                        if ai ∉ config_b
                            if bj ∉ config_a

                                #Sp.Sm + Sm.Sp
                                delta_a = get_fock_delta(ai, bj, P.ras_spaces)
                                delta_b = get_fock_delta(bj, ai, P.ras_spaces)
                                block2 = RasBlock(block1.focka.+delta_a, block1.fockb.+delta_b)
                                haskey(v, block2) || continue
                                La = lu[block1.focka][ai,bj,Ia]
                                La != 0 || continue
                                sign_a = sign(La)
                                La = abs(La)
                                Lb = lu[block1.fockb][bj,ai,Ib]
                                Lb != 0 || continue
                                sign_b = sign(Lb)
                                Lb = abs(Lb)
                                for r in 1:nr
                                    s2[r] -= sign_a*sign_b*vec[Ia, Ib,r]*v[block2][La, Lb, r]
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

function apply_S2_matrix(prob, v)
end

function compute_1rdm(prob::RASCIAnsatz_2, C::Vector)
    rdm1a = zeros(prob.no, prob.no)
    rdm1b = zeros(prob.no, prob.no)
    return rdm1a, rdm1b
end

function compute_1rdm_2rdm(prob::RASCIAnsatz_2, C::Vector)
    rdm1a, rdm1b = comput_1rdm(prob, C)
    rdm2aa = zeros(prob.no, prob.no, prob.no, prob.no)
    rdm2bb = zeros(prob.no, prob.no, prob.no, prob.no)
    rdm2ab = zeros(prob.no, prob.no, prob.no, prob.no)
    return rdm1a, rdm1b, rdm2aa, rdm2bb, rdm2ab
end


