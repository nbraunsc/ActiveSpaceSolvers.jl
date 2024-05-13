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

Constructor to create RASCI Vector that allowes problems like DDCIAnsatz where you want multiple cases of holes/particles
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
    return RASVector(rasvec)
end#=}}}=#

function RASVector(v, prob::DDCIAnsatz)
    h=Int8(0)#={{{=#
    p=Int8(0)
    h2=Int8(0)
    p2=Int8(0)
    h3 = Int8(0)
    p3 = Int8(0)
    if prob.ex_level==1
        h=Int8(1)
        p=Int8(0)
        h2=Int8(0)
        p2=Int8(1)
    elseif prob.ex_level==2
        h=Int8(2)
        p=Int8(0)
        h2=Int8(0)
        p2=Int8(2)
        h3 = Int8(1)
        p3 = Int8(1)
    elseif prob.ex_level==3
        h=Int8(2)
        p=Int8(1)
        h2=Int8(1)
        p2=Int8(2)
    else
        error("No DDCIAnsatz Excitation Level Defined")
    end


    a_blocks, fock_as = make_blocks(prob.ras_spaces, prob.na, h, p)
    b_blocks, fock_bs = make_blocks(prob.ras_spaces, prob.nb, h, p)
    rasvec = OrderedDict{ActiveSpaceSolvers.RASCI_2.RasBlock, Array{Float64,3}}()
    nroots = size(v, 2)
    
    start = 1

    for i in 1:length(a_blocks)
        dima = binomial(prob.ras_spaces[1], fock_as[i][1])*binomial(prob.ras_spaces[2], fock_as[i][2])*binomial(prob.ras_spaces[3], fock_as[i][3])
        for j in 1:length(b_blocks)
            dimb = binomial(prob.ras_spaces[1], fock_bs[j][1])*binomial(prob.ras_spaces[2], fock_bs[j][2])*binomial(prob.ras_spaces[3], fock_bs[j][3])
            if a_blocks[i][1]+b_blocks[j][1]<= h
                if a_blocks[i][2]+b_blocks[j][2] <= p
                    block1 = RasBlock(fock_as[i], fock_bs[j])
                    rasvec[block1] = reshape(v[start:start+dima*dimb-1, :], dima, dimb, nroots)
                    start += dima*dimb
                end
            end
        end
    end
    
    a_blocks2, fock_as2 = make_blocks(prob.ras_spaces, prob.na, h2, p2)
    b_blocks2, fock_bs2 = make_blocks(prob.ras_spaces, prob.nb, h2, p2)
    for i in 1:length(a_blocks2)
        dima = binomial(prob.ras_spaces[1], fock_as2[i][1])*binomial(prob.ras_spaces[2], fock_as2[i][2])*binomial(prob.ras_spaces[3], fock_as2[i][3])
        for j in 1:length(b_blocks2)
            dimb = binomial(prob.ras_spaces[1], fock_bs2[j][1])*binomial(prob.ras_spaces[2], fock_bs2[j][2])*binomial(prob.ras_spaces[3], fock_bs2[j][3])
            if a_blocks2[i][1]+b_blocks2[j][1]<= h2
                if a_blocks2[i][2]+b_blocks2[j][2] <= p2
                    block1 = RasBlock(fock_as2[i], fock_bs2[j])
                    if haskey(rasvec, block1)
                        continue
                    else
                        rasvec[block1] = reshape(v[start:start+dima*dimb-1, :], dima, dimb, nroots)
                        start += dima*dimb
                    end
                end
            end
        end
    end

    if h3 != 0
        ##DDCI 2x
        a_blocks2, fock_as2 = make_blocks(prob.ras_spaces, prob.na, h3, p3)
        b_blocks2, fock_bs2 = make_blocks(prob.ras_spaces, prob.nb, h3, p3)
        for i in 1:length(a_blocks2)
            dima = binomial(prob.ras_spaces[1], fock_as2[i][1])*binomial(prob.ras_spaces[2], fock_as2[i][2])*binomial(prob.ras_spaces[3], fock_as2[i][3])
            for j in 1:length(b_blocks2)
                dimb = binomial(prob.ras_spaces[1], fock_bs2[j][1])*binomial(prob.ras_spaces[2], fock_bs2[j][2])*binomial(prob.ras_spaces[3], fock_bs2[j][3])
                if a_blocks2[i][1]+b_blocks2[j][1]<= h3
                    if a_blocks2[i][2]+b_blocks2[j][2] <= p3
                        block1 = RasBlock(fock_as2[i], fock_bs2[j])
                        if haskey(rasvec, block1)
                            continue
                        else
                            rasvec[block1] = reshape(v[start:start+dima*dimb-1, :], dima, dimb, nroots)
                            start += dima*dimb
                        end
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
