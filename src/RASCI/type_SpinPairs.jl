using JLD2
using InCoreIntegrals
using StaticArrays


struct Spin_Pair
    pair::Tuple{Int, Int} #(alpha HP category, beta HP category)
    dim::Int
end

"""
    make_spin_pairs(prob::RASCIAnsatz)

"""
function make_spin_pairs(prob::RASCIAnsatz)
    spin_pairs = Vector{Spin_Pair}()
    a_categories = ActiveSpaceSolvers.RASCI.make_categories(prob, spin="alpha")
    b_categories = ActiveSpaceSolvers.RASCI.make_categories(prob, spin="beta")
    for i in 1:length(a_categories)
        dima = length(a_categories[i].idxs)
        for j in a_categories[i].connected 
            dimb = length(b_categories[j].idxs)
            x = Spin_Pair((i, j), dima*dimb)
            push!(spin_pairs, x)
        end
    end
    return spin_pairs, a_categories, b_categories
end

"""
    make_spin_pairs(prob::RASCIAnsatz)

"""
function make_new_spin_pairs(prob::RASCIAnsatz)
    spin_pairs = Vector{Spin_Pair}()
    a_categories = ActiveSpaceSolvers.RASCI.make_new_categories(prob, spin="alpha")
    b_categories = ActiveSpaceSolvers.RASCI.make_new_categories(prob, spin="beta")
    for i in 1:length(a_categories)
        dima = length(a_categories[i].idxs)
        for j in a_categories[i].connected 
            dimb = length(b_categories[j].idxs)
            x = Spin_Pair((i, j), dima*dimb)
            push!(spin_pairs, x)
        end
    end
    return spin_pairs, a_categories, b_categories
end


"""
    make_spin_pairs(prob::RASCIAnsatz, a_categories::Vector{<:HP_Category}, b_categories::Vector{<:HP_Category})

"""
function make_spin_pairs(prob::RASCIAnsatz, a_categories::Vector{<:HP_Category}, b_categories::Vector{<:HP_Category})
    spin_pairs = Vector{Spin_Pair}()
    for i in 1:length(a_categories)
        dima = length(a_categories[i].idxs)
        for j in a_categories[i].connected 
            dimb = length(b_categories[j].idxs)
            x = Spin_Pair((i, j), dima*dimb)
            push!(spin_pairs, x)
        end
    end
    return spin_pairs
end

"""
    find_spin_pair(spin_pairs::Vector{Spin_Pair}, current::Tuple{Int,Int})

"""
function find_spin_pair(spin_pairs::Vector{Spin_Pair}, current::Tuple{Int,Int})
    for i in 1:length(spin_pairs)
        if spin_pairs[i].pair == current 
            return i
        else
            continue
        end
    end
    return 0
end

"""
    find_spin_pair(spin_pairs::Vector{Spin_Pair}, current::Int, spin="beta")

"""
function find_spin_pair(spin_pairs::Vector{Spin_Pair}, current::Int, spin="beta")
    if spin == "beta"
        for i in 1:length(spin_pairs)
            if spin_pairs[i].pair[2] == current
                return i
            else
                continue
            end
        end
    else
        for j in 1:length(spin_pairs)
            if spin_pairs[j].pair[1] == current
                return j
            else
                continue
            end
        end
    end
end

function possible_pairs(spin_pairs::Vector{Spin_Pair}, alpha::Int)
    pairs = zeros(Int, length(spin_pairs))
    for i in 1:length(spin_pairs)
        if spin_pairs[i].pair[1] == alpha
            pairs[i] = spin_pairs[i].pair[2]
        else
            continue
        end
    end
    return pairs
end

function possible_spin_pairs(spin_pairs::Vector{Spin_Pair}, alpha::Int)
    pairs = Vector{Tuple{Int, Int}}()
    for i in 1:length(spin_pairs)
        if spin_pairs[i].pair[1] == alpha
            push!(pairs, spin_pairs[i].pair)
        else
            continue
        end
    end
    return pairs
end

function pair(pairs::Vector{Int}, val::Int)
    for (pos, b) in enumerate(pairs)
        if b == val
            return pos
        else
            continue
        end
    end
    return 0
end

    





