using ActiveSpaceSolvers
using OrderedCollections
using TensorOperations
using InCoreIntegrals
using StaticArrays

function breakup_config(det::Vector{Int}, ras1, ras2, ras3)
    det1 = Vector{Int}()#={{{=#
    det2 = Vector{Int}()
    det3 = Vector{Int}()
    for i in det
        if i in ras1
            push!(det1, i)
        elseif i in ras2
            push!(det2, i)
        else
            push!(det3, i)
        end
    end
    return det1, det2, det3
end#=}}}=#


function apply_annihilation(det::Vector{Int}, orb_a)
    sign_a =1#={{{=#
    if orb_a in det
        spot = findfirst(det.==orb_a)
        splice!(det, spot)

        if spot % 2 != 1
            sign_a = -1
        end
        #if length(det) == 1
        #    return sign_a, [det]
        #else
        #    return sign_a, det
        #end
        return sign_a, det
    else
        return 0, 0
    end
end#=}}}=#

function apply_creation(det::Vector{Int}, orb_c)
    insert_here = 1#={{{=#
    sign_c = 1
    if orb_c in det
        return 0, 0
    end

    if isempty(det)
        det = [orb_c]
    else
        for i in 1:length(det)
            if det[i] > orb_c
                insert_here = i
                break
            else
                insert_here += 1
            end
        end
        insert!(det, insert_here, orb_c)
    end

    if insert_here % 2 != 1
        sign_c = -1
    end
    return sign_c, det
end#=}}}=#

function apply_annihilation!(det1::Vector{Int}, det2::Vector{Int}, det3::Vector{Int}, orb_a)
    sign_a =1#={{{=#
    if orb_a in det1
        spot = findfirst(det1.==orb_a)
        splice!(det1, spot)
        
        if spot % 2 != 1
            sign_a = -1
        end
        println(det1, det2, det3)

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
    ras1, ras2, ras3 = make_ras_spaces(ras_spaces)
    norbs = sum(ras_spaces)
    lookup = initalize_lu(v, norbs)
    
    #general lu table
    for (fock1, lu_data) in lookup
        #conf_scr1_2 = zeros(Int, fock2[1])
        #conf_scr2_2 = zeros(Int, fock2[2])
        #conf_scr3_2 = zeros(Int, fock2[3])

        idx = 0
        det3 = SubspaceDeterminantString(ras_spaces[3], fock1[3])
        for n in 1:det3.max
            det2 = SubspaceDeterminantString(ras_spaces[2], fock1[2])
            for j in 1:det2.max
                det1 = SubspaceDeterminantString(ras_spaces[1], fock1[1])
                for i in 1:det1.max
                    #idx = calc_full_ras_index(det1, det2, det3)
                    idx += 1
                    config = [det1.config;det2.config.+det1.no;det3.config.+det1.no.+det2.no]

                    for k in config
                        #sgn_a, det1_config, det2_config, det3_config = apply_annihilation!(tmp_det1, tmp_det2.+det1.no, tmp_det3.+det1.no.+det2.no, k)
                        tmp = deepcopy(config)
                        sgn_a, deta = apply_annihilation(tmp, k)

                        for l in 1:sum(ras_spaces)
                            tmp2 = deepcopy(deta)
                            if l in tmp2
                                continue
                            end

                            sgn_c, detc = apply_creation(tmp2, l)
                            #sgn_c != 0 || continue
                            delta_kl = get_fock_delta(k, l, ras_spaces)
                            haskey(lookup, fock1.+delta_kl) || continue
                            
                            d1, d2, d3 = breakup_config(detc, ras1, ras2, ras3)

                            det1_c = SubspaceDeterminantString(length(ras1), length(d1), d1)
                            det2_c = SubspaceDeterminantString(length(ras2), length(d2), d2.-length(ras1))
                            det3_c = SubspaceDeterminantString(length(ras3), length(d3), d3.-length(ras1).-length(ras2))
                            idx_new = calc_full_ras_index(det1_c, det2_c, det3_c)
                            
                            #sgn_c, idx_new = apply_creation!(tmp_c1, tmp_c2, tmp_c3, ras1, ras2, ras3, l)
                            #sgn_c, idx_new = apply_creation!(det1_config, det2_config, det3_config, ras1, ras2, ras3, l)
                            lookup[fock1][k, l, idx] = sgn_a*sgn_c*idx_new
                        end
                    end
                    incr!(det1)
                end
                incr!(det2)
            end
            incr!(det3)
        end
    end
    return lookup
end#=}}}=#

function get_configs(ras_spaces, fock)
    tmp = []#={{{=#
    det3 = SubspaceDeterminantString(ras_spaces[3], fock[3])
    for n in 1:det3.max
        det2 = SubspaceDeterminantString(ras_spaces[2], fock[2])
        for j in 1:det2.max
            det1 = SubspaceDeterminantString(ras_spaces[1], fock[1])
            for i in 1:det1.max
                #idx = calc_full_ras_index(det1, det2, det3)
                config = [det1.config;det2.config.+det1.no;det3.config.+det1.no.+det2.no]
                #push!(tmp, (idx,config))
                push!(tmp, config)
                incr!(det1)
            end
            incr!(det2)
        end
        incr!(det3)
    end
    return tmp#=}}}=#
end

function find_fock_delta_ccaa(a, aa, c, cc, ras1, ras2, ras3)
    delta = (0,0,0)#={{{=#
    if a in ras1
        delta =  (delta[1]-1, delta[2], delta[3])
    elseif a in ras2
        delta =  (delta[1], delta[2]-1, delta[3])
    else
        delta =  (delta[1], delta[2], delta[3]-1)
    end
    
    if aa in ras1
        delta =  (delta[1]-1, delta[2], delta[3])
    elseif aa in ras2
        delta =  (delta[1], delta[2]-1, delta[3])
    else
        delta =  (delta[1], delta[2], delta[3]-1)
    end
    
    if c in ras1
        delta =  (delta[1]+1, delta[2], delta[3])
    elseif c in ras2
        delta =  (delta[1], delta[2]+1, delta[3])
    else
        delta =  (delta[1], delta[2], delta[3]+1)
    end
    
    if cc in ras1
        delta =  (delta[1]+1, delta[2], delta[3])
    elseif cc in ras2
        delta =  (delta[1], delta[2]+1, delta[3])
    else
        delta =  (delta[1], delta[2], delta[3]+1)
    end

    return delta#=}}}=#
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

    #i_orbs = range(start=1, stop=ras_spaces[1])
    #ii_orbs = range(start=ras_spaces[1]+1,stop=ras_spaces[1]+ras_spaces[2])
    #iii_orbs = range(start=ras_spaces[1]+ras_spaces[2]+1, stop=ras_spaces[1]+ras_spaces[2]+ras_spaces[3])
    i_orbs, ii_orbs, iii_orbs = make_ras_spaces(ras_spaces)

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

function make_excitation_classes_ccaa(ras_spaces::SVector{3, Int})
    i_orbs, ii_orbs, iii_orbs = make_ras_spaces(ras_spaces)#={{{=#

    ranges = [i_orbs, ii_orbs, iii_orbs]
    
    double_exc = OrderedDict{Tuple{Vector{Int}, Vector{Int}, Vector{Int}, Vector{Int}}, Tuple{Int, Int, Int}}()
    for (pidx, p) in enumerate(ranges)
        for (qidx, q) in enumerate(ranges)
            for (ridx, r) in enumerate(ranges)
                for (sidx, s) in enumerate(ranges)
                    tmp = [0,0,0]
                    tmp[pidx] += 1
                    tmp[qidx] += 1
                    tmp[ridx] -= 1
                    tmp[sidx] -= 1
                    tmp2 = Tuple(Float64(x) for x in tmp)
                    double_exc[(p,q,r,s)] = tmp2
                end
            end
        end
    end
    return double_exc
end#=}}}=#

function make_excitation_classes_cca(ras_spaces::SVector{3, Int})
    i_orbs, ii_orbs, iii_orbs = make_ras_spaces(ras_spaces)#={{{=#

    ranges = [i_orbs, ii_orbs, iii_orbs]
    
    double_exc = OrderedDict{Tuple{Vector{Int}, Vector{Int}, Vector{Int}}, Tuple{Int, Int, Int}}()
    for (pidx, p) in enumerate(ranges)
        for (qidx, q) in enumerate(ranges)
            for (ridx, r) in enumerate(ranges)
                tmp = [0,0,0]
                tmp[pidx] += 1
                tmp[qidx] += 1
                tmp[ridx] -= 1
                tmp2 = Tuple(Float64(x) for x in tmp)
                double_exc[(p,q,r)] = tmp2 
            end
        end
    end
    return double_exc
end#=}}}=#

function make_excitation_classes_cc(ras_spaces::SVector{3, Int})
    i_orbs, ii_orbs, iii_orbs = make_ras_spaces(ras_spaces)#={{{=#
    ranges = [i_orbs, ii_orbs, iii_orbs]
    
    double_exc = OrderedDict{Tuple{Vector{Int}, Vector{Int}}, Tuple{Int, Int, Int}}()
    for (pidx, p) in enumerate(ranges)
        for (qidx, q) in enumerate(ranges)
            tmp = [0,0,0]
            tmp[pidx] += 1
            tmp[qidx] += 1
            tmp2 = Tuple(Float64(x) for x in tmp)
            double_exc[(q,p)] = tmp2 
        end
    end
    return double_exc
end#=}}}=#

function make_excitation_classes_ca(ras_spaces::SVector{3, Int})
    i_orbs, ii_orbs, iii_orbs = make_ras_spaces(ras_spaces)#={{{=#
    ranges = [i_orbs, ii_orbs, iii_orbs]
    
    double_exc = OrderedDict{Tuple{Vector{Int}, Vector{Int}}, Tuple{Int, Int, Int}}()
    for (pidx, p) in enumerate(ranges)
        for (qidx, q) in enumerate(ranges)
            tmp = [0,0,0]
            tmp[pidx] += 1
            tmp[qidx] -= 1
            tmp2 = Tuple(Float64(x) for x in tmp)
            double_exc[(p,q)] = tmp2 
        end
    end
    return double_exc
end#=}}}=#

function make_excitation_classes_c(ras_spaces::SVector{3, Int})
    i_orbs, ii_orbs, iii_orbs = make_ras_spaces(ras_spaces)#={{{=#

    ranges = [i_orbs, ii_orbs, iii_orbs]
    
    double_exc = OrderedDict{Vector{Int},Tuple{Int, Int, Int}}()
    for (pidx, p) in enumerate(ranges)
        tmp = [0,0,0]
        tmp[pidx] += 1
        tmp2 = Tuple(Float64(x) for x in tmp)
        double_exc[(p)] = tmp2 
    end
    return double_exc
end#=}}}=#

function make_excitation_classes_a(ras_spaces::SVector{3, Int})
    i_orbs, ii_orbs, iii_orbs = make_ras_spaces(ras_spaces)#={{{=#

    ranges = [i_orbs, ii_orbs, iii_orbs]
    
    double_exc = OrderedDict{Vector{Int},Tuple{Int, Int, Int}}()
    for (pidx, p) in enumerate(ranges)
        tmp = [0,0,0]
        tmp[pidx] -= 1
        tmp2 = Tuple(Float64(x) for x in tmp)
        double_exc[(p)] = tmp2 
    end
    return double_exc
end#=}}}=#

function initalize_sig(v::RASVector)
    sig = OrderedDict{RasBlock, Array{Float64, 3}}()#={{{=#
    for (block, vec) in v.data
        sig[block] = zeros(size(vec))
    end
    return sig
end#=}}}=#

function initalize_sig_ba(v::RASVector)
    sig = OrderedDict{RasBlock, Array{Float64, 3}}()#={{{=#
    for (block, vec) in v.data
        sig[block] = zeros(size(vec,2), size(vec,1), size(vec,3))
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

function _fill_Ckl!(Ckl::OrderedDict{RasBlock, Array{Float64, 3}}, focka::Tuple{Int,Int,Int}, v::RASVector, L::Vector{Int}, nroots::Int)
    empty!(Ckl)#={{{=#
    nI = length(L)

    for (blocks, vec) in v.data
        if blocks.focka == focka
            Ckl[blocks] = zeros(size(vec))
        end
    end
    
    for (sub_block, vec2) in Ckl
        for si in 1:nroots
            for Jb in 1:size(vec2,2)
                for Li in 1:nI
                    Ckl[sub_block][Li, Jb, si] = v.data[sub_block][abs(L[Li]), Jb, si]*sign(L[Li])
                end
            end
        end
    end
end#=}}}=#

function get_ckl_dim(ras_spaces::SVector{3,Int}, fock::Tuple{Int,Int,Int}, k_range, l_range)
    dim1 = 0#={{{=#
    dim2 = 0
    ras1, ras2, ras3 = make_ras_spaces(ras_spaces)

    if length(k_range) == 0 || length(l_range) == 0
        return dim1, dim2
    end

    if k_range[1] in ras1
        if l_range[1] in ras1
            dim1=binom_coeff_calc(ras_spaces[1], fock[1])*binom_coeff_calc(ras_spaces[2]+1, fock[2]+1)*binom_coeff_calc(ras_spaces[3]+1, fock[3]+1)
            dim2=binom_coeff_calc(ras_spaces[1]-1, fock[1])*binom_coeff_calc(ras_spaces[2]+1, fock[2]+1)*binom_coeff_calc(ras_spaces[3]+1, fock[3]+1)

        elseif l_range[1] in ras2
            dim2 = binom_coeff_calc(ras_spaces[1],fock[1])*(binom_coeff_calc(ras_spaces[2]+1,fock[2]+1)-binom_coeff_calc(ras_spaces[2], fock[2]))*binom_coeff_calc(ras_spaces[3]+1, fock[3]+1)
        else #l_range in ras3
            dim2 = binom_coeff_calc(ras_spaces[1],fock[1])*(binom_coeff_calc(ras_spaces[3]+1,fock[3]+1)-binom_coeff_calc(ras_spaces[3], fock[3]))*binom_coeff_calc(ras_spaces[2]+1, fock[2]+1)
        end
        return dim1, dim2

    elseif k_range[1] in ras2
        if l_range[1] in ras1
            dim2 = binom_coeff_calc(ras_spaces[2],fock[2])*(binom_coeff_calc(ras_spaces[1]+1,fock[1]+1)-binom_coeff_calc(ras_spaces[1], fock[1]))*binom_coeff_calc(ras_spaces[3]+1, fock[3]+1)
        elseif l_range[1] in ras2
            dim1=binom_coeff_calc(ras_spaces[2],fock[2])*binom_coeff_calc(ras_spaces[1]+1,fock[1]+1)*binom_coeff_calc(ras_spaces[3]+1,fock[3]+1)
            dim2=binom_coeff_calc(ras_spaces[2]-1,fock[2])*binom_coeff_calc(ras_spaces[1]+1,fock[1]+1)*binom_coeff_calc(ras_spaces[3]+1,fock[3]+1)
        else #l_range in ras3
            dim2 = binom_coeff_calc(ras_spaces[2],fock[2])*(binom_coeff_calc(ras_spaces[3]+1,fock[3]+1)-binom_coeff_calc(ras_spaces[3], fock[3]))*binom_coeff_calc(ras_spaces[1]+1,fock[1]+1)
        end
        return dim1, dim2
    else
        if l_range[1] in ras1
            dim2 = binom_coeff_calc(ras_spaces[3],fock[3])*(binom_coeff_calc(ras_spaces[1]+1,fock[1]+1)-binom_coeff_calc(ras_spaces[1], fock[1]))*binom_coeff_calc(ras_spaces[2]+1,fock[2]+1)
        elseif l_range[1] in ras2
            dim2 = binom_coeff_calc(ras_spaces[3],fock[3])*(binom_coeff_calc(ras_spaces[2]+1,fock[2]+1)-binom_coeff_calc(ras_spaces[2], fock[2]))*binom_coeff_calc(ras_spaces[1]+1,fock[1]+1)
        else #l_range in ras3
            dim1=binom_coeff_calc(ras_spaces[3],fock[3])*binom_coeff_calc(ras_spaces[1]+1,fock[1]+1)*binom_coeff_calc(ras_spaces[2]+1,fock[2]+1)
            dim2=binom_coeff_calc(ras_spaces[3]-1,fock[3])*binom_coeff_calc(ras_spaces[1]+1,fock[1]+1)*binom_coeff_calc(ras_spaces[2]+1,fock[2]+1)
        end
        return dim1, dim2
    end#=}}}=#
end

function get_ckl_dim_old(ras_spaces::SVector{3,Int}, fock::Tuple{Int,Int,Int}, k_range, l_range)
    dim1 = 0#={{{=#
    dim2 = 0
    ras1, ras2, ras3 = make_ras_spaces(ras_spaces)

    if length(k_range) == 0 || length(l_range) == 0
        return dim1, dim2
    end

    if k_range[1] in ras1
        if l_range[1] in ras1
            dim1=binomial(ras_spaces[1]-1,fock[1]-1)*binomial(ras_spaces[2],fock[2])*binomial(ras_spaces[3],fock[3])
            dim2=binomial(ras_spaces[1]-2,fock[1]-1)*binomial(ras_spaces[2],fock[2])*binomial(ras_spaces[3],fock[3])
        elseif l_range[1] in ras2
            dim2 = binomial(ras_spaces[1]-1,fock[1]-1)*(binomial(ras_spaces[2],fock[2])-binomial(ras_spaces[2]-1, fock[2]-1))*binomial(ras_spaces[3], fock[3])
        else #l_range in ras3
            dim2 = binomial(ras_spaces[1]-1,fock[1]-1)*(binomial(ras_spaces[3],fock[3])-binomial(ras_spaces[3]-1, fock[3]-1))*binomial(ras_spaces[2], fock[2])
        end
        return dim1, dim2

    elseif k_range[1] in ras2
        if l_range[1] in ras1
            dim2 = binomial(ras_spaces[2]-1,fock[2]-1)*(binomial(ras_spaces[1],fock[1])-binomial(ras_spaces[1]-1, fock[1]-1))*binomial(ras_spaces[3], fock[3])
        elseif l_range[1] in ras2
            dim1=binomial(ras_spaces[2]-1,fock[2]-1)*binomial(ras_spaces[1],fock[1])*binomial(ras_spaces[3],fock[3])
            dim2=binomial(ras_spaces[2]-2,fock[2]-1)*binomial(ras_spaces[1],fock[1])*binomial(ras_spaces[3],fock[3])
        else #l_range in ras3
            dim2 = binomial(ras_spaces[2]-1,fock[2]-1)*(binomial(ras_spaces[3],fock[3])-binomial(ras_spaces[3]-1, fock[3]-1))*binomial(ras_spaces[1],fock[1])
        end
        return dim1, dim2
    else
        if l_range[1] in ras1
            dim2 = binomial(ras_spaces[3]-1,fock[3]-1)*(binomial(ras_spaces[1],fock[1])-binomial(ras_spaces[1]-1, fock[1]-1))*binomial(ras_spaces[2],fock[2])
        elseif l_range[1] in ras2
            dim2 = binomial(ras_spaces[3]-1,fock[3]-1)*(binomial(ras_spaces[2],fock[2])-binomial(ras_spaces[2]-1, fock[2]-1))*binomial(ras_spaces[1],fock[1])
        else #l_range in ras3
            dim1=binomial(ras_spaces[3]-1,fock[3]-1)*binomial(ras_spaces[1],fock[1])*binomial(ras_spaces[2],fock[2])
            dim2=binomial(ras_spaces[3]-2,fock[3]-1)*binomial(ras_spaces[1],fock[1])*binomial(ras_spaces[2],fock[2])
        end
        return dim1, dim2
    end#=}}}=#
end

function binom_coeff_calc(orb::Int, e::Int)
    if orb <= 0#={{{=#
        return 0
    elseif e <= 0
        return 0
    else
        bc = binom_coeff[orb, e]
        if bc < 0
            return 0
        else
            return bc
        end
    end
end
#=}}}=#

function _mult!(Ckl::Array{T,3}, FJb::Array{T,1}, VI::Array{T,2}) where {T}
    #={{{=#
    nI = size(Ckl)[1]
    n_roots::Int = size(Ckl)[3]
    ket_max = size(FJb)[1]
    tmp = 0.0
    for si in 1:n_roots
        @views V = VI[:,si]
        for Jb in 1:ket_max
            tmp = FJb[Jb]
            if abs(tmp) > 1e-14
                @inbounds @simd for I in 1:nI
                    VI[I,si] += tmp*Ckl[I,Jb,si]
                end
            end
        end
    end
end
#=}}}=#

function get_Ckl!(Ckl::Array{T,3}, v::Array{T,3}, L::Vector{Int}, count::Int, nroots::Int) where T
    for si in 1:nroots#={{{=#
        for Jb in 1:size(v,2)
            for Li in 1:count
                Ckl[Li,Jb,si] = v[abs(L[Li]), Jb, si]*sign(L[Li])
            end
        end
    end
end#=}}}=#

function scatter!(sig, VI::Array{T,2}, count::Int, nroots::Int) where T
    for si in 1:nroots#={{{=#
        for Li in 1:count
            sig[Li,si] += VI[Li,si]
        end
    end
end#=}}}=#

function scatter_Ib!(sig, VI::Array{T,2}, count::Int, nroots::Int) where T
    for si in 1:nroots#={{{=#
        for Ib in 1:size(sig,2)
            for Li in 1:count
                sig[Li,Ib,si] += VI[Li,si]
            end
        end
    end
end#=}}}=#

function get_beta!(i_range::Vector{Int}, j_range::Vector{Int}, lu::Array{Int,3}, hkl::Array{Float64,2}, nroots::Int, sign_a::Int, sigIa, v2)
    for j in j_range, i in i_range#={{{=#
        #R = findall(!iszero, lu[i,j,:])
        #L = lu[i,j,R]
        R = Vector{Int}()
        L = Vector{Int}()
        for (Iidx, I) in enumerate(lu[i,j,:])
            if I != 0
                push!(R, Iidx)
                push!(L, I)
            end
        end
        length(R) != 0 || continue
        h = hkl[j,i]
        for Li in 1:length(R)
            for si in 1:nroots
                @inbounds sigIa[R[Li], si] += sign_a*sign(L[Li])*h*v2[abs(L[Li]), si]
            end
        end
    end
end#=}}}=#

"""
Sigma one (beta)
"""
function sigma_one(v::RASVector, ints::InCoreInts, ras_spaces::SVector{3, Int}, lu::Dict{Tuple{Int,Int,Int}, Array{Int,3}}) where T
    sig1 = initalize_sig_ba(v)#={{{=#
    single_excit = make_single_excit(ras_spaces)
    gkl = get_gkl(ints, sum(ras_spaces)) 
    no = sum(ras_spaces)
    first_entry = first(v.data)
    na = sum(first_entry[1].focka)
    nroots = size(first_entry[2],3)
    
    #sign to switch from (a,b) to (b,a) for optimizing
    sgnK = 1 
    if (na) % 2 != 0 
        sgnK = -sgnK
    end
    
    v_perm = initalize_sig_ba(v)

    for (block1, vec) in v.data
        v_perm[block1] .= sgnK.*permutedims(v.data[block1], (2,1,3))
    end
    
    for (block1, vec) in v_perm
        for ((k_range, l_range), delta1) in single_excit
            if length(k_range) == 0 || length(l_range)==0
                continue
            end
            block2 = RasBlock(block1.focka, block1.fockb.+delta1)
            haskey(v_perm,block2) || continue
            F = zeros(size(v_perm[block2], 1))
            #F = zeros(size(v.data[block2], 2))
            for Ib in 1:size(vec,1)
                fill!(F, 0.0)
                for l in l_range, k in k_range  
                    Jb = lu[block1.fockb][k,l,Ib]
                    Jb != 0 || continue
                    sign_kl = sign(Jb)
                    Jb = abs(Jb)
                    F[Jb] += sign_kl*gkl[l,k]
                    comb_kl = (l-1)*no + k
                    @views lu_Jb = lu[block1.fockb.+delta1][:,:,Jb]
                    for ((i_range, j_range), delta2) in single_excit
                        if length(i_range) == 0 || length(j_range)==0
                            continue
                        end
                        block3 = RasBlock(block1.focka, block1.fockb.+delta1.+delta2)
                        haskey(v_perm,block3) || continue
                        #when block3 == block2 can do double excitations and contract with same F array
                        if block3 == block2
                            for j in j_range, i in i_range
                                comb_ij = (j-1)*no + i
                                comb_ij >= comb_kl || continue
                                Kb = lu_Jb[i,j]
                                #Kb = lu[block1.fockb.+delta1][i,j,Jb]   # Ka is local to block3
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

                @views sig_Ib = sig1[block1][Ib,:,:]
                #@views sig_Ib = sig1[block1][:,Ib,:]
                @views C = v_perm[block2]
                @tensor begin
                    sig_Ib[Ia, r] += F[Jb]*C[Jb, Ia, r]
                    #sig_Ib[Ia, r] += F[Jb]*C[Ia, Jb, r]
                end
            end
            
            #collecting double excitations where block2 != block3
            #don't need to compute single excitations bec all were found in above block
            for ((i_range, j_range), delta2) in single_excit
                block3 = RasBlock(block1.focka, block1.fockb.+delta1.+delta2)
                haskey(v_perm,block3) || continue
                block3 != block2 || continue
                F_ij = zeros(size(v_perm[block3], 1))
                #F_ij = zeros(size(v.data[block3], 2))
                for Ib in 1:size(vec,1)
                #for Ib in 1:size(vec,2)
                    fill!(F_ij, 0.0)
                    for l in l_range, k in k_range
                        Jb = lu[block1.fockb][k,l,Ib]
                        Jb != 0 || continue
                        sign_kl = sign(Jb)
                        Jb = abs(Jb)
                        comb_kl = (l-1)*no + k
                        @views lu_Jb = lu[block1.fockb.+delta1][:,:,Jb]
                        for j in j_range, i in i_range
                            comb_ij = (j-1)*no + i
                            comb_ij >= comb_kl || continue

                            Kb = lu_Jb[i,j]
                            #Kb = lu[block1.fockb.+delta1][i,j,Jb]   # Kb is local to block3
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
                    
                    @views sig_Ib = sig1[block1][Ib,:,:]
                    #@views sig_Ib = sig1[block1][:,Ib,:]
                    @views C = v_perm[block3]
                    @tensor begin
                        sig_Ib[Ia, r] += F_ij[Jb]*C[Jb, Ia, r]
                        #sig_Ib[Ia, r] += F_ij[Jb]*C[Ia, Jb, r]
                    end
                end
            end
        end
    end

    starti = 1
    dim = get_dim(v)
    sig = zeros(Float64, dim, nroots)
    for (block, vec) in sig1
        tmp = reshape(sgnK.*permutedims(vec, (2,1,3)), (size(vec,1)*size(vec,2), nroots))
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
    nroots = size(first(v.data)[2],3)

    for (block1, vec) in v.data
        for ((k_range, l_range), delta1) in single_excit
            if length(k_range) == 0 || length(l_range)==0
                continue
            end
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
                    @views lu_Ja = lu[block1.focka.+delta1][:,:,Ja]
                    for ((i_range, j_range), delta2) in single_excit
                        if length(i_range) == 0 || length(j_range)==0
                            continue
                        end
                        block3 = RasBlock(block1.focka.+delta1.+delta2, block1.fockb)
                        haskey(v.data,block3) || continue
                        #when block3 == block2 can do double excitations and contract with same F array
                        if block3 == block2
                            for j in j_range, i in i_range
                                comb_ij = (j-1)*no + i
                                comb_ij >= comb_kl || continue
                                Ka = lu_Ja[i,j]
                                #Ka = lu[block1.focka.+delta1][i,j,Ja]   # Ka is local to block3
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
                        @views lu_Ja = lu[block1.focka.+delta1][:,:,Ja]
                        for j in j_range, i in i_range
                            comb_ij = (j-1)*no + i
                            comb_ij >= comb_kl || continue

                            Ka = lu_Ja[i,j]
                            #Ka = lu[block1.focka.+delta1][i,j,Ja]   # Ka is local to block3
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
    nroots = size(first(v.data)[2],3)
    
    for (block1, vec) in v.data
        Ckl = Array{Float64,3}
        for ((k_range, l_range), delta_a) in single_excit
            if length(k_range) == 0 || length(l_range)==0
                continue
            end

            for ((i_range, j_range), delta_b) in single_excit
                if length(j_range) == 0 || length(i_range)==0
                    continue
                end
                block2 = RasBlock(block1.focka.+delta_a, block1.fockb.+delta_b)
                haskey(v.data, block2) || continue
                dim1,dim2 = get_ckl_dim(ras_spaces, block1.focka, k_range, l_range)
                Ckl_scr1 = zeros(Float64, dim1, size(v.data[block2],2), size(vec,3))
                Ckl_scr2 = zeros(Float64, dim2, size(v.data[block2],2), size(vec,3))
                F = zeros(Float64, size(v.data[block2],2))
                for l in l_range, k in k_range
                    if l == k
                        R = zeros(Int, dim1)
                        L = zeros(Int, dim1)
                        Ckl = deepcopy(Ckl_scr1)
                        VI = zeros(Float64, dim1 ,nroots)
                    else
                        R = zeros(Int, dim2)
                        L = zeros(Int, dim2)
                        Ckl = deepcopy(Ckl_scr2)
                        VI = zeros(Float64, dim2 ,nroots)
                    end
                    count = 0
                    for (Iidx, I) in enumerate(lu[block1.focka][k,l,:])
                        if I != 0
                            count += 1
                            R[count] = Iidx
                            L[count] = I
                        end
                    end
                    
                    hkl .= ints.h2[:,:,l,k]
                    get_Ckl!(Ckl, v.data[block2], L, count, nroots)
                    
                    #for Ib in 1:size(vec,2)
                    #    fill!(F, 0.0)
                    #    @views lu_Ib = lu[block1.fockb][:,:,Ib]
                    #    for j in j_range, i in i_range
                    #        Jb = lu_Ib[i,j]
                    #        #Jb = lu[block1.fockb][i, j, Ib]
                    #        Jb != 0 || continue
                    #        sign_b = sign(Jb)
                    #        Jb = abs(Jb)
                    #        F[Jb] += hkl[j,i]*sign_b
                    #    end
                    for Ib in 1:size(vec,2)
                        fill!(F, 0.0)
                        for i in i_range
                            @views lu_Ib = lu[block1.fockb][i,:,Ib]
                            for j in j_range 
                                Jb = lu_Ib[j]
                                Jb != 0 || continue
                                sign_b = sign(Jb)
                                Jb = abs(Jb)
                                F[Jb] += hkl[j,i]*sign_b
                            end
                        end
                        
                        #contract with Ckl and add into VI
                        fill!(VI, 0.0)
                        _mult!(Ckl, F, VI)

                        @views sigIB = sig3[block1][R,Ib,:]
                        scatter!(sigIB, VI, count, nroots)
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
Sigma three is the mixed spin block (both alpha and beta single excitations)
"""
function sigma_three_nodiag(v::RASVector, ints::InCoreInts, ras_spaces::SVector{3, Int}, lu::Dict{Tuple{Int,Int,Int}, Array{Int,3}})
    sig3 = initalize_sig(v)#={{{=#
    single_excit = make_single_excit(ras_spaces)
    no = sum(ras_spaces) 
    hkl = zeros(Float64, no, no)
    nroots = size(first(v.data)[2],3)
    
    for (block1, vec) in v.data
        Ckl = Array{Float64,3}
        for ((k_range, l_range), delta_a) in single_excit
            if length(k_range) == 0 || length(l_range)==0
                continue
            end

            for ((i_range, j_range), delta_b) in single_excit
                if length(j_range) == 0 || length(i_range)==0
                    continue
                end
                block2 = RasBlock(block1.focka.+delta_a, block1.fockb.+delta_b)
                haskey(v.data, block2) || continue
                dim1,dim2 = get_ckl_dim(ras_spaces, block1.focka, k_range, l_range)
                Ckl_scr2 = zeros(Float64, dim2, size(v.data[block2],2), size(vec,3))
                F = zeros(Float64, size(v.data[block2],2))
                for l in l_range, k in k_range
                    l != k || continue
                    R = zeros(Int, dim2)
                    L = zeros(Int, dim2)
                    Ckl = deepcopy(Ckl_scr2)
                    VI = zeros(Float64, dim2 ,nroots)
                    count = 0
                    for (Iidx, I) in enumerate(lu[block1.focka][k,l,:])
                        if I != 0
                            count += 1
                            R[count] = Iidx
                            L[count] = I
                        end
                    end
                    
                    hkl .= ints.h2[:,:,l,k]
                    get_Ckl!(Ckl, v.data[block2], L, count, nroots)
                    
                    for Ib in 1:size(vec,2)
                        fill!(F, 0.0)
                        @views lu_Ib = lu[block1.fockb][:,:,Ib]
                        for j in j_range, i in i_range
                            j != i || continue
                            Jb = lu_Ib[i,j]
                            Jb != 0 || continue
                            sign_b = sign(Jb)
                            Jb = abs(Jb)
                            F[Jb] += hkl[j,i]*sign_b
                        end
                        
                        #contract with Ckl and add into VI
                        fill!(VI, 0.0)
                        _mult!(Ckl, F, VI)

                        @views sigIB = sig3[block1][R,Ib,:]
                        scatter!(sigIB, VI, count, nroots)
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
Sigma three is the mixed spin block (both alpha and beta single excitations)
"""
function sigma_three_diag(v::RASVector, ints::InCoreInts, ras_spaces::SVector{3, Int}, lu::Dict{Tuple{Int,Int,Int}, Array{Int,3}})
    sig3 = initalize_sig(v)#={{{=#
    no = sum(ras_spaces) 
    hkl = zeros(Float64, no, no)
    nroots = size(first(v.data)[2],3)
    
    i_orbs, ii_orbs, iii_orbs = make_ras_spaces(ras_spaces)
    ranges = [i_orbs, ii_orbs, iii_orbs]
    
    for (block1, vec) in v.data
        Ckl = Array{Float64,3}
        F = zeros(Float64, size(vec,2))
        for orbs in ranges
            dim1, dim2 = get_ckl_dim(ras_spaces, block1.focka, orbs, orbs)
            Ckl_scr1 = zeros(Float64, dim1, size(vec,2), size(vec,3))
            for k in orbs
                Ckl = deepcopy(Ckl_scr1)
                R = zeros(Int, dim1)
                L = zeros(Int, dim1)
                VI = zeros(Float64, dim1 ,nroots)
                count = 0
                for (Iidx, I) in enumerate(lu[block1.focka][k,k,:])
                    if I != 0
                        count += 1
                        R[count] = Iidx
                        L[count] = I
                    end
                end

                hkl .= ints.h2[:,:,k,k]
                get_Ckl!(Ckl, vec, L, count, nroots)

                for Ib in 1:size(vec,2)
                    fill!(F, 0.0)
                    @views lu_Ib = lu[block1.fockb][:,:,Ib]
                    for orbs_b in ranges
                        for i in orbs_b
                            Jb = lu_Ib[i,i]
                            Jb != 0 || continue
                            sign_b = sign(Jb)
                            Jb = abs(Jb)
                            F[Jb] += hkl[i,i]*sign_b
                        end

                        #contract with Ckl and add into VI
                        fill!(VI, 0.0)
                        _mult!(Ckl, F, VI)

                        @views sigIB = sig3[block1][R,Ib,:]
                        scatter!(sigIB, VI, count, nroots)
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
Sigma three is the mixed spin block (both alpha and beta single excitations)
"""
function sigma_three_old_old(v::RASVector, ints::InCoreInts, ras_spaces::SVector{3, Int}, lu::Dict{Tuple{Int,Int,Int}, Array{Int,3}})
    sig3 = initalize_sig(v)#={{{=#
    single_excit = make_single_excit(ras_spaces)
    no = sum(ras_spaces) 
    hkl = zeros(Float64, no, no)
    nroots = size(first(v.data)[2],3)

    for (block1, vec) in v.data
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
Sigma three is the mixed spin block (both alpha and beta single excitations)
"""
function sigma_three_old(v::RASVector, ints::InCoreInts, ras_spaces::SVector{3, Int}, lu::Dict{Tuple{Int,Int,Int}, Array{Int,3}})
    sig3 = initalize_sig(v)#={{{=#
    single_excit = make_single_excit(ras_spaces)
    no = sum(ras_spaces) 
    hkl = zeros(Float64, no, no)
    nroots = size(first(v.data)[2],3)
    
    for (block1, vec) in v.data
        for ((k_range, l_range), delta_a) in single_excit
            for ((i_range, j_range), delta_b) in single_excit
                block2 = RasBlock(block1.focka.+delta_a, block1.fockb.+delta_b)
                haskey(v.data, block2) || continue
                for aconfig in 1:size(vec, 1)
                    @views sigIa = sig3[block1][aconfig, :, :]
                    for l in l_range, k in k_range
                        Ja = lu[block1.focka][k, l, aconfig]
                        Ja != 0 || continue     
                        sign_a = sign(Ja)
                        Ja = abs(Ja)
                        hkl .= ints.h2[:,:,l,k]
                        @views v2 = v.data[block2][Ja, :, :]
                        #gather
                        get_beta!(i_range, j_range, lu[block1.fockb], hkl, nroots, sign_a, sigIa, v2)
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
    
    for (block1, vec) in v.data
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
                                delta_a = get_fock_delta(ai, bj, P.ras_spaces)
                                delta_b = get_fock_delta(bj, ai, P.ras_spaces)

                                block2 = RasBlock(block1.focka.+delta_a, block1.fockb.+delta_b)
                                haskey(v.data, block2) || continue

                                #Sp.Sm + Sm.Sp
                                La = lu[block1.focka][ai,bj,Ia]
                                La != 0 || continue
                                sign_a = sign(La)
                                La = abs(La)
                                Lb = lu[block1.fockb][bj,ai,Ib]
                                Lb != 0 || continue
                                sign_b = sign(Lb)
                                Lb = abs(Lb)
                                for r in 1:nr
                                    s2[r] -= sign_a*sign_b*vec[Ia, Ib,r]*v.data[block2][La, Lb, r]
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

function apply_S2_matrix(P::RASCIAnsatz_2, C::AbstractArray{T}) where T
    ###{{{
    #S2 = (S+S- + S-S+)1/2 + Sz.Sz
    #   = 1/2 sum_ij(ai'bi bj'ai + bj'aj ai'bi) + Sz.Sz
    #   do swaps and you can end up adding the two together to get rid
    #   of the 1/2 factor so 
    #   = (-1) sum_ij(ai'aj|alpha>bj'bi|beta> + Sz.Sz
    ###

    nr = size(C,2)
    v = RASVector(C, P)
    s2v = initalize_sig(v)
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
                    s2v[block1][Ia,Ib,:] .+= 0.25.*vec[Ia, Ib, :]
                end

                #Sz.Sz (β)
                count_b = (P.nb-1)*P.nb
                for i in 1:count_b
                    s2v[block1][Ia,Ib,:] .+= 0.25.*vec[Ia, Ib, :]
                end

                #Sz.Sz (α,β)
                for ai in config_a
                    for bj in config_b
                        if ai != bj
                            s2v[block1][Ia,Ib,:] .-= 0.5.*vec[Ia, Ib, :]
                        end
                    end
                end

                ##Sp.Sm + Sm.Sp Diagonal Part
                for ai in config_a
                    if ai in config_b
                    else
                        s2v[block1][Ia,Ib,:] .+= 0.75.*vec[Ia, Ib, :]
                    end
                end

                for bi in config_b
                    if bi in config_a
                    else
                        s2v[block1][Ia,Ib,:] .+= 0.75.*vec[Ia, Ib, :]
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
                                s2v[block1][Ia,Ib,:] .-= sign_a*sign_b*v[block2][La, Lb, :]
                            end
                        end
                    end
                end
            end
        end
    end

    starti = 1
    S2 = zeros(Float64, P.dim, nr)
    for (block, vec) in s2v
        tmp = reshape(vec, (size(vec,1)*size(vec,2), nr))
        S2[starti:starti+(size(vec,1)*size(vec,2))-1, :] .= tmp
        starti += (size(vec,1)*size(vec,2))
    end
    return S2#=}}}=#
end

function compute_1rdm(prob::RASCIAnsatz_2, C::Vector)
    v = RASVector(C, prob)#={{{=#
    lu = fill_lu(v, prob.ras_spaces)
    single_excit = make_single_excit(prob.ras_spaces)
    rdm1a = zeros(prob.no, prob.no)
    rdm1b = zeros(prob.no, prob.no)

    for (block1, vec) in v.data
        for ((k_range, l_range), delta1) in single_excit
            block2 = RasBlock(block1.focka.+delta1, block1.fockb)
            haskey(v.data,block2) || continue
            for Ia in 1:size(vec,1)
                for l in l_range, k in k_range  
                    Ja = lu[block1.focka][k,l,Ia]
                    Ja != 0 || continue
                    sign_kl = sign(Ja)
                    Ja = abs(Ja)
                    rdm1a[k,l] += sign_kl*dot(v.data[block2][Ja,:], v.data[block1][Ia,:])
                end
            end
        end
    end
    
    for (block1, vec) in v.data
        for ((k_range, l_range), delta1) in single_excit
            block2 = RasBlock(block1.focka, block1.fockb.+delta1)
            haskey(v.data,block2) || continue
            for Ib in 1:size(vec,2)
                for l in l_range, k in k_range  
                    Jb = lu[block1.fockb][k,l,Ib]
                    Jb != 0 || continue
                    sign_kl = sign(Jb)
                    Jb = abs(Jb)
                    rdm1b[k,l] += sign_kl*dot(v.data[block2][:,Jb], v.data[block1][:,Ib])
                end
            end
        end
    end

    return rdm1a, rdm1b#=}}}=#
end

function compute_1rdm_2rdm_old(prob::RASCIAnsatz_2, C::Vector)
    v = RASVector(C, prob)#={{{=#
    lu = fill_lu(v, prob.ras_spaces)
    single_excit = make_single_excit(prob.ras_spaces)
    rdm1a, rdm1b = compute_1rdm(prob, C)
    rdm2aa = zeros(prob.no, prob.no, prob.no, prob.no)
    rdm2bb = zeros(prob.no, prob.no, prob.no, prob.no)
    rdm2ab = zeros(prob.no, prob.no, prob.no, prob.no)
    
    ras1 = range(start=1, stop=prob.ras_spaces[1])
    ras2 = range(start=prob.ras_spaces[1]+1,stop=prob.ras_spaces[1]+prob.ras_spaces[2])
    ras3 = range(start=prob.ras_spaces[1]+prob.ras_spaces[2]+1, stop=prob.ras_spaces[1]+prob.ras_spaces[2]+prob.ras_spaces[3])
    
    aconfig = zeros(Int, prob.na)
    aconfig_a = zeros(Int, prob.na-1)
    #alpha alpha p'q'rs
    for (block1, vec) in v.data
        idx = 0
        det3 = SubspaceDeterminantString(prob.ras_spaces[3], block1.focka[3])
        for n in 1:det3.max
            det2 = SubspaceDeterminantString(prob.ras_spaces[2], block1.focka[2])
            for j in 1:det2.max
                det1 = SubspaceDeterminantString(prob.ras_spaces[1], block1.focka[1])
                for i in 1:det1.max
                    #idx = calc_full_ras_index(det1, det2, det3)
                    idx += 1
                    aconfig = [det1.config;det2.config.+det1.no;det3.config.+det1.no.+det2.no]
                    
                    for s in aconfig
                        tmp = deepcopy(aconfig)
                        sgn_a, aconfig_a = apply_annihilation(tmp, s)
                        #if length(aconfig_a) != prob.na -1
                        #    error("a")
                        #end

                        for r in aconfig_a
                            tmp2 = deepcopy(aconfig_a)
                            sgn_aa, aconfig_aa = apply_annihilation(tmp2, r)
                            #if length(aconfig_aa) != prob.na -2
                            #    error("aa")
                            #end
                            #sgn_aa != 0 || continue
                            for q in 1:prob.no
                                tmp3 = deepcopy(aconfig_aa)
                                sgn_c, config_c = apply_creation(tmp3, q)
                                sgn_c != 0 || continue
                                #if length(config_c) != prob.na -1
                                #    error("c")
                                #end
                                for p in 1:prob.no
                                    p != q || continue
                                    tmp4 = deepcopy(config_c)
                                    d1_c, d2_c, d3_c = breakup_config(tmp4, ras1, ras2, ras3)
                                    sgn_cc, idx_new = apply_creation!(d1_c, d2_c, d3_c, ras1, ras2, ras3, p)
                                    sgn_cc != 0 || continue
                                    delta = find_fock_delta_ccaa(s,r,q,p,ras1, ras2, ras3)
                                    new_block = RasBlock(block1.focka.+delta, block1.fockb)
                                    haskey(v.data, new_block) || continue
                                    rdm2aa[p,s,q,r] += sgn_a*sgn_aa*sgn_c*sgn_cc*dot(v.data[new_block][idx_new,:], v.data[block1][idx,:])
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
    
    #beta beta p'q'rs
    bconfig = zeros(Int, prob.nb)
    bconfig_a = zeros(Int, prob.nb-1)
    for (block1, vec) in v.data
        idx = 0
        det3 = SubspaceDeterminantString(prob.ras_spaces[3], block1.fockb[3])
        for n in 1:det3.max
            det2 = SubspaceDeterminantString(prob.ras_spaces[2], block1.fockb[2])
            for j in 1:det2.max
                det1 = SubspaceDeterminantString(prob.ras_spaces[1], block1.fockb[1])
                for i in 1:det1.max
                    idx += 1
                    #idx = calc_full_ras_index(det1, det2, det3)
                    bconfig = [det1.config;det2.config.+det1.no;det3.config.+det1.no.+det2.no]
                    
                    for s in bconfig
                        tmp = deepcopy(bconfig)
                        sgn_a, bconfig_a = apply_annihilation(tmp, s)
                        #if length(aconfig_a) != prob.na -1
                        #    error("a")
                        #end

                        for r in bconfig_a
                            tmp2 = deepcopy(bconfig_a)
                            sgn_aa, bconfig_aa = apply_annihilation(tmp2, r)
                            #if length(aconfig_aa) != prob.na -2
                            #    error("aa")
                            #end
                            #sgn_aa != 0 || continue
                            for q in 1:prob.no
                                tmp3 = deepcopy(bconfig_aa)
                                sgn_c, bconfig_c = apply_creation(tmp3, q)
                                sgn_c != 0 || continue
                                #if length(config_c) != prob.na -1
                                #    error("c")
                                #end
                                for p in 1:prob.no
                                    p != q || continue
                                    tmp4 = deepcopy(bconfig_c)
                                    d1_c, d2_c, d3_c = breakup_config(tmp4, ras1, ras2, ras3)
                                    sgn_cc, idx_new = apply_creation!(d1_c, d2_c, d3_c, ras1, ras2, ras3, p)
                                    sgn_cc != 0 || continue
                                    delta = find_fock_delta_ccaa(s,r,q,p,ras1, ras2, ras3)
                                    new_block = RasBlock(block1.focka, block1.fockb.+delta)
                                    haskey(v.data, new_block) || continue
                                    rdm2bb[p,s,q,r] += sgn_a*sgn_aa*sgn_c*sgn_cc*dot(v.data[new_block][:,idx_new], v.data[block1][:,idx])
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

    #alpha beta
    for (block1, vec) in v.data
        for ((k_range, l_range), delta_a) in single_excit
            for ((i_range, j_range), delta_b) in single_excit
                block2 = RasBlock(block1.focka.+delta_, block1.fockb.+delta_b)
                haskey(v.data, block2) || continue
                for aconfig in 1:size(vec, 1)
                    for l in l_range, k in k_range
                        Ja = lu[block1.focka][k, l, aconfig]
                        Ja != 0 || continue     
                        sign_a = sign(Ja)
                        Ja = abs(Ja)
                        for bconfig in 1:size(vec, 2)
                            for j in j_range, i in i_range
                                Jb = lu[block1.fockb][i, j, bconfig]
                                Jb != 0 || continue     
                                sign_b = sign(Jb)
                                Jb = abs(Jb)
                                sgn = sign_a*sign_b
                                rdm2ab[l,k,j,i] += sgn*v.data[block2][Ja,Jb]*v.data[block1][aconfig, bconfig]
                            end
                        end
                    end
                end
            end
        end
    end
    
    return rdm1a, rdm1b, rdm2aa, rdm2bb, rdm2ab#=}}}=#
end

function compute_1rdm_2rdm(prob::RASCIAnsatz_2, C::Vector)
    v = RASVector(C, prob)#={{{=#
    lu = fill_lu(v, prob.ras_spaces)
    rdm1a, rdm1b = compute_1rdm(prob, C)
    rdm2aa = zeros(prob.no, prob.no, prob.no, prob.no)
    rdm2bb = zeros(prob.no, prob.no, prob.no, prob.no)
    rdm2ab = zeros(prob.no, prob.no, prob.no, prob.no)
    
    single_excit = make_single_excit(prob.ras_spaces)
    double_excit = make_excitation_classes_ccaa(prob.ras_spaces)

    ras1, ras2, ras3 = make_ras_spaces(prob.ras_spaces)
    
    #ras1 = range(start=1, stop=prob.ras_spaces[1])
    #ras2 = range(start=prob.ras_spaces[1]+1,stop=prob.ras_spaces[1]+prob.ras_spaces[2])
    #ras3 = range(start=prob.ras_spaces[1]+prob.ras_spaces[2]+1, stop=prob.ras_spaces[1]+prob.ras_spaces[2]+prob.ras_spaces[3])
    
    aconfig = zeros(Int, prob.na)
    aconfig_a = zeros(Int, prob.na-1)
    #alpha alpha p'q'rs
    for (block1, vec) in v.data
        idx = 0
        det3 = SubspaceDeterminantString(prob.ras_spaces[3], block1.focka[3])
        for n in 1:det3.max
            det2 = SubspaceDeterminantString(prob.ras_spaces[2], block1.focka[2])
            for j in 1:det2.max
                det1 = SubspaceDeterminantString(prob.ras_spaces[1], block1.focka[1])
                for i in 1:det1.max
                    #idx = calc_full_ras_index(det1, det2, det3)
                    idx += 1
                    aconfig = [det1.config;det2.config.+det1.no;det3.config.+det1.no.+det2.no]

                    for ((p_range, q_range, r_range, s_range), delta_e) in double_excit
                        new_block = RasBlock(block1.focka.+ delta_e, block1.fockb)
                        haskey(v.data, new_block) || continue
                        for s in s_range
                            tmp = deepcopy(aconfig)
                            sgn_a, aconfig_a = apply_annihilation(tmp, s)
                            sgn_a != 0 || continue
                            if length(aconfig_a) != prob.na -1
                                error("a")
                            end
                            for r in r_range
                                r != s || continue
                                tmp2 = deepcopy(aconfig_a)
                                sgn_aa, aconfig_aa = apply_annihilation(tmp2, r)
                                sgn_aa != 0 || continue
                                if length(aconfig_aa) != prob.na -2
                                    println(aconfig_aa)
                                    error("aa")
                                end
                                for q in q_range
                                    tmp3 = deepcopy(aconfig_aa)
                                    sgn_c, config_c = apply_creation(tmp3, q)
                                    sgn_c != 0 || continue
                                    if length(config_c) != prob.na -1
                                        error("c")
                                    end
                                    for p in p_range
                                        p != q || continue
                                        tmp4 = deepcopy(config_c)
                                        d1_c, d2_c, d3_c = breakup_config(tmp4, ras1, ras2, ras3)
                                        sgn_cc, idx_new = apply_creation!(d1_c, d2_c, d3_c, ras1, ras2, ras3, p)
                                        sgn_cc != 0 || continue
                                        rdm2aa[p,s,q,r] += sgn_a*sgn_aa*sgn_c*sgn_cc*dot(v.data[new_block][idx_new,:], v.data[block1][idx,:])
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
    
    #beta beta p'q'rs
    bconfig = zeros(Int, prob.nb)
    bconfig_a = zeros(Int, prob.nb-1)
    for (block1, vec) in v.data
        idx = 0
        det3 = SubspaceDeterminantString(prob.ras_spaces[3], block1.fockb[3])
        for n in 1:det3.max
            det2 = SubspaceDeterminantString(prob.ras_spaces[2], block1.fockb[2])
            for j in 1:det2.max
                det1 = SubspaceDeterminantString(prob.ras_spaces[1], block1.fockb[1])
                for i in 1:det1.max
                    idx += 1
                    #idx = calc_full_ras_index(det1, det2, det3)
                    bconfig = [det1.config;det2.config.+det1.no;det3.config.+det1.no.+det2.no]
                    for ((p_range, q_range, r_range, s_range), delta_e) in double_excit
                        new_block = RasBlock(block1.focka, block1.fockb.+delta_e)
                        haskey(v.data, new_block) || continue
                        for s in s_range
                            tmp = deepcopy(bconfig)
                            sgn_a, bconfig_a = apply_annihilation(tmp, s)
                            sgn_a != 0 || continue
                            #if length(aconfig_a) != prob.na -1
                            #    error("a")
                            #end
                            for r in r_range
                                r != s || continue
                                tmp2 = deepcopy(bconfig_a)
                                sgn_aa, bconfig_aa = apply_annihilation(tmp2, r)
                                #if length(aconfig_aa) != prob.na -2
                                #    error("aa")
                                #end
                                sgn_aa != 0 || continue
                                for q in q_range
                                    tmp3 = deepcopy(bconfig_aa)
                                    sgn_c, config_c = apply_creation(tmp3, q)
                                    sgn_c != 0 || continue
                                    #if length(config_c) != prob.na -1
                                    #    error("c")
                                    #end
                                    for p in p_range
                                        p != q || continue
                                        tmp4 = deepcopy(config_c)
                                        d1_c, d2_c, d3_c = breakup_config(tmp4, ras1, ras2, ras3)
                                        sgn_cc, idx_new = apply_creation!(d1_c, d2_c, d3_c, ras1, ras2, ras3, p)
                                        sgn_cc != 0 || continue
                                        rdm2bb[p,s,q,r] += sgn_a*sgn_aa*sgn_c*sgn_cc*dot(v.data[new_block][:,idx_new], v.data[block1][:,idx])
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

    #alpha beta
    for (block1, vec) in v.data
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
                        for bconfig in 1:size(vec, 2)
                            for j in j_range, i in i_range
                                Jb = lu[block1.fockb][i, j, bconfig]
                                Jb != 0 || continue     
                                sign_b = sign(Jb)
                                Jb = abs(Jb)
                                sgn = sign_a*sign_b
                                rdm2ab[l,k,j,i] += sgn*v.data[block2][Ja,Jb]*v.data[block1][aconfig, bconfig]
                            end
                        end
                    end
                end
            end
        end
    end
    
    return rdm1a, rdm1b, rdm2aa, rdm2bb, rdm2ab#=}}}=#
end

function make_ras_spaces(ras_spaces::SVector{3,Int})
    if ras_spaces[1] == 0
        ras1 = []
        if ras_spaces[2] == 0
            ras2 = []
            ras3 = range(start=1, stop=ras_spaces[3])
        else
            ras2 = range(start=1, stop=ras_spaces[2])
            if ras_spaces[3] == 0
                ras3 = []
            else
                ras3 = range(start=ras_spaces[2]+1, stop=ras_spaces[2]+ras_spaces[3])
            end
        end
    else
        ras1 = range(start=1, stop=ras_spaces[1])
        if ras_spaces[2] == 0
            ras2 = []
            if ras_spaces[3]==0
                ras3=[]
            else
                ras3 = range(start=ras_spaces[1]+1, stop=ras_spaces[1]+ras_spaces[3])
            end
        else
            ras2 = range(start=ras_spaces[1]+1, stop=ras_spaces[1]+ras_spaces[2])
            if ras_spaces[3] == 0
                ras3 = []
            else
                ras3 = range(start=ras_spaces[1]+ras_spaces[2]+1, stop=ras_spaces[1]+ras_spaces[2]+ras_spaces[3])
            end
        end
    end
    return ras1, ras2, ras3
end

    
    



