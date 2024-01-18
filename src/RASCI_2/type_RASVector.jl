struct RASVector{T}
    data::OrderedDict{Tuple{Int, Int, Int, Int}, Array{3, T}}
    #                 Tuple{nhα, nhβ, npα, npβ}, Array{alpha, beta, roots}
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
function RASVector(v::Array{T}, ras_spaces::Tuple{Int, Int, Int}, na, nb, no; max_h=0, max_p=0, max_h2=nothing, max_p2=nothing) where T
    a_blocks, fock_as = make_blocks(ras_spaces, na, max_h, max_p)
    b_blocks, fock_bs = make_blocks(ras_spaces, nb, max_h, max_p)
    rasvec = OrderedDict{Tuple{Int, Int, Int, Int}, Array{3, T}}()
    nroots = size(v, 2)
    
    start = 1
    for i in 1:length(a_blocks)
        dima = binomial(ras_spaces[1], fock_as[i][1])*binomial(ras_spaces[2], fock_as[i][2])*binomial(ras_spaces[3], fock_as[i][3])
        for j in 1:length(b_blocks)
            dimb = binomial(ras_spaces[1], fock_bs[j][1])*binomial(ras_spaces[2], fock_bs[j][2])*binomial(ras_spaces[3], fock_bs[j][3])
            if a_blocks[i][1]+b_blocks[j][1]<= max_h
                if a_blocks[i][2]+b_blocks[j][2] <= max_p
                    rasvec[(a_blocks[i][1], b_blocks[j][1], a_blocks[i][2], b_blocks[j][2])] = reshape(v[start:start+dima*dimb-1, :], dima, dimb, nroots)
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
                        rasvec[(a_blocks2[i][1], b_blocks2[j][1], a_blocks2[i][2], b_blocks2[j][2])] = reshape(v[start:start+dima*dimb-1, :], dima, dimb, nroots)
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
function shift(v::RASVector, current_hp, spin="alpha")
    if spin == "alpha"
        shift = 1
        for (hp, vec) in v
            if (hp[1], hp[3]) == current_hp
                return shift
            else
                shift += size(vec, 1)
            end
        end
        return shift
    else
        shift = 1
        for (hp, vec) in v
            if (hp[2], hp[4]) == current_hp
                return shift
            else
                shift += size(vec, 2)
            end
        end
        return shift
    end
end

function get_dima(v::RASVector)
    dima = 0
    for (hp, vec) in v
        dima += size(vec, 1)
    end
    return dima
end

function get_dimb(v::RASVector)
    dimb = 0
    for (hp, vec) in v
        dimb += size(vec, 2)
    end
    return dimb
end

function fill_lu(v::RASVector, ne::Int, ras_spaces::Tuple{Int, Int, Int})
    i_orbs = [1:1:ras_spaces[1];]
    start2 = ras_spaces[1]+1
    end2 = start2+ras_spaces[2]-1
    ii_orbs = [start2:1:end2;]
    start = norbs-ras_spaces[3]+1
    iii_orbs = [start:1:norbs;]

    ras1 = generate_configs(i_orbs, 

    for (hp, vec) in v
    end
    return lu
end

function generate_configs(orbs::Vector{T}, ne::T) where T
    configs = Vector[]
    result = Vector{T}[[]]
    for elem in x, j in eachindex(result)
        push!(result, [result[j] ; elem])
        if length(last(result)) == ne
            push!(configs, last(result))
        end
    end
    return configs
end

function _sum_spin_pairs!(sig::RASVector, v::RASVector, F::Array{Float}, I::Int, hp_s::Tuple{Int, Int}, hp_d::Tuple{Int, Int})
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
    i_orbs = [1:1:ras_spaces[1];]
    start2 = ras_spaces[1]+1
    end2 = start2+ras_spaces[2]-1
    ii_orbs = [start2:1:end2;]
    start = norbs-ras_spaces[3]+1
    iii_orbs = [start:1:norbs;]

    single_exc = OrderedDict{Tuple{Vector{Int}, Vector{Int}}, Tuple{Int, Int}}()
    single_exc[(i_orbs, i_orbs)] = (0, 0) #ras1->ras1
    single_exc[(ii_orbs, ii_orbs)] = (0, 0) #ras2->ras2
    single_exc[(iii_orbs, iii_orbs)] = (0, 0) #ras3->ras3
    single_exc[(i_orbs, ii_orbs)] = (1, 0) #ras1->ras2
    single_exc[(i_orbs, iii_orbs)] = (1, 1) #ras1->ras3
    single_exc[(ii_orbs, i_orbs)] = (-1, 0) #ras2->ras1
    single_exc[(ii_orbs, iii_orbs)] = (0, 1) #ras2->ras3
    single_exc[(iii_orbs, i_orbs)] = (-1, -1) #ras3->ras1
    single_exc[(iii_orbs, ii_orbs)] = (0, -1) #ras3->ras2
    return single_exc
end

"""
Not sure i actually use this function yet
"""
function make_alpha_blocks(C::RASVector)
    alpha_blocks = Vector{Tuple{Int, Int}}()
    shift = 0
    for block in keys(C)
        alpha_block = (block[1], block[3])
        push!(alpha_blocks, alpha_block)
        shift += dima
    end
    return alpha_blocks
end

"""
Sigma two is all alpha
"""
function sigma_two(C::RASVector, ints::InCoreInts, ras_spaces::Tuple{Int, Int, Int}, no::Int, nroots::Int)
    sig2 = RASVector(RASVector.data, zeros{3, T}())
    single_excit = make_single_excit(ras_spaces)
    dima = get_dima(C)
    gkl = get_gkl(ints, prob) 
    
    F = zeros(Float64, dima)

    for Ia in 1:dima
        comb_kl = 0
        comb_ij = 0
        fill!(F,0.0)
        for (se_a, hpa) in single_excit
            for k in se_a[1], l in se_a[2]
                Ja = lu[k, l, Ia]
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
        end
    end
    return sig2
end

"""
Sigma three is the mixed spin block (both alpha and beta single excitations)
"""
function sigma_three(C::RASVector, ints::InCoreInts, ras_spaces::Tuple{Int, Int, Int}, no::Int, nroots::Int)
    sig3 = RASVector(RASVector.data, zeros{3, T}())
    single_excit = make_single_excit(ras_spaces)
    
    hkl = zeros(Float64, no, no)

    for block in keys(C.data)
        for (se_a, hpa) in single_excit
            for (se_b, hpb) in single_excit
                new_block = (block[1]+hpa[1], block[2]+hpb[1], block[3]+hpa[2], block[4]+hpb[2])
                if haskey(new_block, C.data)
                    for aconfig in 1:size(C[block], 1)
                        for k in se_a[1], l in se_a[2]
                            Ja = lu[k, l, aconfig]
                            sign_a = sign(Ja)
                            hkl .= ints.h2[:,:,k,l]
                            for bconfig in 1:size(C[block], 2)
                                @views sig = sig3[block][aconfig, b_config, :]
                                for i in se_b[1], j in se_b[2]
                                    Jb = lu[i, j, bconfig]
                                    sign_b = sign(Jb)
                                    h = hkl[i,j]
                                    @views v = C[new_block][Ja, Jb, :]
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


