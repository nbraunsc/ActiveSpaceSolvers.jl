using ActiveSpaceSolvers
using QCBase
import LinearMaps
using OrderedCollections
using BlockDavidson
using StaticArrays
using LinearAlgebra
using Printf
using TimerOutputs

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
    HP_cats_a::Vector{HP_Category_CA}
    HP_cats_b::Vector{HP_Category_CA}
end

function RASCIAnsatz(no::Int, na, nb, ras_spaces::Any; max_h=0, max_p=ras_spaces[3])
    na <= no || throw(DimensionMismatch)
    nb <= no || throw(DimensionMismatch)
    sum(ras_spaces) == no || throw(DimensionMismatch)
    ras_spaces = convert(SVector{3,Int},collect(ras_spaces))
    na = convert(Int, na)
    nb = convert(Int, nb)
    tmp = RASCIAnsatz(no, na, nb, ras_spaces, max_h, max_p)
    HP_cats_a, HP_cats_b = make_categories(tmp)

    dim = 0
    for i in 1:length(HP_cats_a)
        dima = length(HP_cats_a[i][2])
        for j in HP_cats_a[i][1]
            dimb = length(HP_cats_b[j][2])
            dim += dima*dimb
        end
    end

    return RASCIAnsatz(no, na, nb, dima, dimb, ras_dim, ras_spaces, max_h, max_p, HP_cats_a, HP_cats_b);
end

function RASCIAnsatz(no::Int, na::Int, nb::Int, ras_spaces::SVector{3,Int}, max_h, max_p)
    return RASCIAnsatz(no, na, nb, 0, 0, 0, ras_spaces, max_h, max_p);
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
        @printf(" Iter: %4i", iters)
        #print("Iter: ", iters, " ")
        #@printf(" %-50s", "Compute sigma 1: ")
        flush(stdout)
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


struct HP_Category_CA{N} <: HP_Category
    idx::Int
    hp::Tuple{Int, Int} #(holes, particles)
    connected::Vector{Int} #list of allowed pairings of other spin categories
    idxs::N #number of configs in this HP category
    shift::Int #shift from local to global indexes
    lookup::Array{N, 3} #single spin lookup table for single excitations
    cat_lookup::Array{N, 3} #single spin lookup table for single excitations
end

struct RASCI_OlsenGraph
    no::Int
    ne::Int
    spaces::SVector{3,Int}
    max::Int
    vert::Array{Int32}
    connect::Dict{Int32, Vector{Int32}}
    weights::Dict{Tuple{Int32, Int32}, Int32}
end

function make_categories(prob::RASCIAnsatz)
    categories = ActiveSpaceSolvers.RASCI.generate_spin_categories(prob)
    all_cats_a = Vector{HP_Category_CA}()
    
    cats_a = deepcopy(categories)
    cats_b = deepcopy(categories)
    fock_list_a, del_at_a = make_fock_from_categories(categories, prob, "alpha")
    deleteat!(cats_a, del_at_a)
    len_cat_a = length(cats_a)
        
    fock_list_b, del_at_b = make_fock_from_categories(categories, prob, "beta")
    deleteat!(cats_b, del_at_b)
    len_cat_b = length(cats_b)
    
    #ALPHA
    connected = make_spincategory_connections(cats_a, cats_b, prob)

    shift = 0
    for j in 1:len_cat_a
        idxas = Vector{Int}()
        graph_a = make_cat_graphs(fock_list_a[j], prob)
        idxas = ActiveSpaceSolvers.RASCI.dfs_idxs(graph_a, 1, graph_a.max, 0) 
        #sort!(idxas)
        lu = zeros(Int, graph_a.no, graph_a.no, idxas)
        cat_lu = zeros(Int, graph_a.no, graph_a.no, idxas)
        push!(all_cats_a, HP_Category_CA(j, cats_a[j], connected[j], idxas, shift, lu, cat_lu))
        shift += idxas
    end

    #have to do same loop as before bec all categories need initalized for the dfs search for lookup tables
    for k in 1:len_cat_a
        graph_a = make_cat_graphs(fock_list_a[k], prob)
        lu, cat_lu = ActiveSpaceSolvers.RASCI.dfs(graph_a, 1, graph_a.max, all_cats_a[k].shift, all_cats_a[k].lookup, all_cats[k].cat_lookup, all_cats)
        all_cats_a[k].lookup .= lu
        all_cats_a[k].cat_lookup .= cat_lu
    end

    #BETA 
    connected = make_spincategory_connections(cats_b, cats_a, prob)
    return all_cats_a, all_cats_b
end

"""
    dfs_idxs(ket::RASCI_OlsenGraph, start, max, lu, cat_lu, categories, config_dict, visited=Vector(zeros(max)), path=[])

Does a depth-first search to find string configurations
"""
function dfs_idxs(ket::RASCI_OlsenGraph, start, max, count, visited=Vector(zeros(max)), path=[])
    visited[start] = true#={{{=#
    push!(path, start)
    if start == max
        #get config,index, add to nodes dictonary
        idx_loc, config = get_index(ket.ne, path, ket.weights)
        count += 1
        
    else
        for i in ket.connect[start]
            if visited[i]==false
                dfs_idxs(ket, i,max, count, visited, path)
            end
        end
    end

    #remove current vertex from path and mark as unvisited
    pop!(path)
    visited[start]=false
    return count#=}}}=#
end

"""
    dfs(ket::RASCI_OlsenGraph, start, max, lu, cat_lu, categories, config_dict, visited=Vector(zeros(max)), path=[])

Does a depth-first search to find string configurations then applys a single excitation to fill the lookup table
"""
function dfs(ket::RASCI_OlsenGraph, start, max, shift, lu, cat_lu, categories, visited=Vector(zeros(max)), path=[])
    visited[start] = true#={{{=#
    push!(path, start)
    if start == max
        #get config,index, add to nodes dictonary
        idx_loc, config = get_index(ket.ne, path, ket.weights)
        
        for orb in config
            for orb_c in 1:ket.no
                #if orb_c in config
                #    continue
                #end
                sgn, conf, idx = ActiveSpaceSolvers.RASCI.apply_single_excitation!(config, orb, orb_c)
                if conf == 0
                    continue
                end
                lu[orb, orb_c, idx_loc] = sgn*idx
                new_cat = find_cat(idx, categories) #need to make this function work differently 
                cat_lu[orb, orb_c, idx_loc] = new_cat.idx
            end
        end
    else
        for i in ket.connect[start]
            if visited[i]==false
                dfs(ket, i,max,shift, lu, cat_lu, categories, visited, path)
            end
        end
    end

    #remove current vertex from path and mark as unvisited
    pop!(path)
    visited[start]=false
    return lu, cat_lu#=}}}=#
end

"""
    find_cat(idx::Int, categories::Vector{<:HP_Category})

Doesn't get used often but will find the HP_cateogry from a given index
"""
function find_cat(idx::Int, categories::Vector{<:HP_Category})
    #this function will find the category that idx belongs to{{{
    for cat in categories
        tmp = collect(1:cat.idxs)
        tmp .+= cat.shift
        if idx in tmp
            return cat.idx
        else
            continue
        end
    end
    return 0#=}}}=#
end

"""
    apply_single_excitation!(config, a_orb, c_orb, config_dict, categories::Vector{HP_Category_CA})

"""
function apply_single_excitation!(config, a_orb, c_orb)
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
        
        if haskey(config_dict, new2);
            idx = config_dict[new2]
        else
            return 1, 0, 0,0
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

        if haskey(config_dict, new2);
            idx = config_dict[new2]
        else
            return 1, 0, 0,0
        end

        sign_c = 1
        if insert_here % 2 != 1
            sign_c = -1
        end
    end

    return sign_c*sign_a, new2, idx#=}}}=#
end

"""
    generate_spin_categories(prob::RASCIAnsatz)

"""
function generate_spin_categories(prob::RASCIAnsatz)
    categories = []#={{{=#

    for h in 1:prob.max_h+1
        holes = h-1
        for p in 1:prob.max_p+1
            particles = p-1
            cat = (holes, particles)
            push!(categories, cat)
        end
    end
    return categories#=}}}=#
end

"""
    make_fock_from_categories(categories, prob::RASCIAnsatz, spin="alpha")

Generates a list of fock sectors that are possible
"""
function make_fock_from_categories(categories, prob::RASCIAnsatz, spin="alpha")
    fock_list = []#={{{=#
    cat_delete = []
    if spin == "alpha"
        if prob.na < prob.ras_spaces[1]
            start = (prob.na, 0, 0)
        elseif prob.na > prob.ras_spaces[1]+prob.ras_spaces[2]
            start = (prob.ras_spaces[1], prob.ras_spaces[2], prob.na-(prob.ras_spaces[1]+prob.ras_spaces[2]))
        else
            start = (prob.ras_spaces[1], prob.na-prob.ras_spaces[1], 0)
        end

        for i in 1:length(categories)
            fock = (start[1]-categories[i][1],prob.na-((start[3]+categories[i][2])+(start[1]-categories[i][1])) ,start[3]+categories[i][2])
            push!(fock_list, fock)

            if any(fock.<0)
                push!(cat_delete, i)
                continue
            end

            if fock[1]>prob.ras_spaces[1] || fock[2]>prob.ras_spaces[2] || fock[3]>prob.ras_spaces[3]
                push!(cat_delete, i)
            end
        end
    
    else

        if prob.nb < prob.ras_spaces[1]
            start = (prob.nb, 0, 0)

        elseif prob.nb > prob.ras_spaces[1]+prob.ras_spaces[2]
            start = (prob.ras_spaces[1], prob.ras_spaces[2], prob.nb-(prob.ras_spaces[1]+prob.ras_spaces[2]))
        else
            start = (prob.ras_spaces[1], prob.nb-prob.ras_spaces[1], 0)
        end

        for i in 1:length(categories)
            fock = (start[1]-categories[i][1],prob.nb-((start[3]+categories[i][2])+(start[1]-categories[i][1])) ,start[3]+categories[i][2])
            push!(fock_list, fock)
            if any(fock.<0)
                push!(cat_delete, i)
                continue
            end
            
            if fock[1]>prob.ras_spaces[1] || fock[2]>prob.ras_spaces[2] || fock[3]>prob.ras_spaces[3]
                push!(cat_delete, i)
            end
        end
    end
    deleteat!(fock_list, cat_delete)
    return fock_list, cat_delete#=}}}=#
end

"""
    make_spincategory_connections(cats1, cats2, prob::RASCIAnsatz)

"""
function make_spincategory_connections(cats1, cats2, prob::RASCIAnsatz)
    connected = Vector{Vector{Int}}()#={{{=#
    for i in 1:length(cats1)
        tmp = Vector{Int}()
        for j in 1:length(cats2)
            if cats1[i][1]+cats2[j][1] <= prob.max_h
                if cats1[i][2]+cats2[j][2] <= prob.max_p
                    append!(tmp, j)
                end
            end
        end
        append!(connected, [tmp])
    end
    return connected#=}}}=#
end

"""
    make_cat_graphs(fock_list, prob::RASCIAnsatz)

Makes GRMS graphs for a specific fock section (i.e. a specific number of electrons in ras1, ras2, and ras3)
"""
function make_cat_graphs(fock_list, prob::RASCIAnsatz)
    #this function will make RASCI Olsen graphs from given fock sector lists{{{
    ras1 = ActiveSpaceSolvers.RASCI.make_ras_x(prob.ras_spaces[1], fock_list[1], SVector(prob.ras_spaces[1], 0, 0), 0, 0)
    ras2 = ActiveSpaceSolvers.RASCI.make_ras_x(prob.ras_spaces[2], fock_list[2], SVector(prob.ras_spaces[2], 0, 0), 0, 0)
    ras3 = ActiveSpaceSolvers.RASCI.make_ras_x(prob.ras_spaces[3], fock_list[3], SVector(prob.ras_spaces[3], 0, 0), 0, 0)
    
    n_unocc_ras2 = (prob.ras_spaces[2]-fock_list[2])+1
    n_unocc_ras3 = (prob.ras_spaces[3]-fock_list[3])+1
    
    update_x_subgraphs!(ras2, n_unocc_ras2, fock_list[2], maximum(ras1))
    update_x_subgraphs!(ras3, n_unocc_ras3, fock_list[3], maximum(ras2))
    
    rows = size(ras1,1)+size(ras2,1)+ size(ras3, 1)-2
    columns = size(ras1, 2) + size(ras2, 2) + size(ras3,2)-2
    full = zeros(Int, rows, columns)
    loc = [size(ras1,1),size(ras1,2)]
    full[1:size(ras1,1), 1:size(ras1,2)] .= ras1
    loc2 = [size(ras2,1)+loc[1]-1, loc[2]+size(ras2,2)-1]
    full[loc[1]:loc2[1], loc[2]:loc2[2]] .= ras2
    loc3 = [size(ras3,1)+loc2[1]-1, size(ras3,2)+loc2[2]-1]
    full[loc2[1]:loc3[1], loc2[2]:loc3[2]] .= ras3
    y = ActiveSpaceSolvers.RASCI.make_ras_y(full)
    vert, max_val = ActiveSpaceSolvers.RASCI.make_vert_graph_ras(full)
    connect, weights = ActiveSpaceSolvers.RASCI.make_graph_dict(y, vert)
    graph = ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph(prob.no, sum(fock_list), prob.ras_spaces, max_val, vert, connect, weights)
    return graph#=}}}=#
end

"""
    update_x_subgraphs!(x, n_unocc, nelec, shift)

Helper function for making the GRMS fock sector graphs
"""
function update_x_subgraphs!(x, n_unocc, nelec, shift)
    if size(x,2) != 0#={{{=#
        x[:,1] .= shift
    end

    if size(x,1) != 0
        x[1,:] .= shift
    end
    for i in 2:nelec+1
        for j in 2:n_unocc
            x[j, i] = x[j-1, i] + x[j, i-1]
        end
    end#=}}}=#
end

"""
    make_ras_x(norbs, nelec, ras_spaces::SVector{3, Int}, ras1_min=0, ras3_max=ras_spaces[3])

Makes x matrix in the GRMS method to help with indexing and finding configurations
"""
function make_ras_x(norbs, nelec, ras_spaces::SVector{3, Int}, ras1_min=0, ras3_max=ras_spaces[3])
    n_unocc = (norbs-nelec)+1#={{{=#
    x = zeros(Int, n_unocc, nelec+1)
    if ras1_min == 0 && ras3_max==ras_spaces[3]
        #do fci graph
        #fill first row and columns
        if size(x,2) != 0
            x[:,1] .= 1
        end

        if size(x,1) != 0
            x[1,:] .= 1
        end


        for i in 2:nelec+1
            for j in 2:n_unocc
                x[j, i] = x[j-1, i] + x[j, i-1]
            end
        end
        return x
    else
        if ras1_min == 0
            x[:,1] .= 1
        end


        if n_unocc == norbs+1
            x[:,1] .=1
            return x
        end


        #meaning if dim of prob  = 1, only one possible config
        if n_unocc == 1
            x[1,:].=1
            return x
        end

        x[1,:].=1
        loc = [1,1]
        #ras_spaces = (3,3,3)

        #RAS1
        if ras1_min == 0
            h = 1
        else
            h = ras_spaces[1]-ras1_min
        end
        for spot in 1:h
            loc[1] += 1
            update_x!(x, loc)
        end
        p = ras_spaces[1]-h
        loc[2] += p

        #RAS2
        p2 = nelec-ras1_min-ras3_max
        h2 = ras_spaces[2] - p2
        for spot in 1:h2
            loc[1] += 1
            #check
            if loc[1] > size(x)[1]
                return x
            else
                update_x!(x, loc) #updates everything at loc and to the right
            end
            #update_x!(x, loc) #updates everything at loc and to the right
        end
        loc[2] += p2


        #RAS3
        h3 = ras_spaces[3] - ras3_max
        if h3 == 0
            h3 = 1
        end

        for spot in 1:h3
            loc[1] += 1
            #check
            if loc[1] > size(x)[1]
                return x
            else
                update_x!(x, loc) #updates everything at loc and to the right
            end
        end#=}}}=#
    end
    return x
end

"""
    make_ras_y(x)

Makes y matrix from x matrix in GRMS method
"""
function make_ras_y(x)
    y = x#={{{=#
    y = vcat(zeros(Int, (1, size(x)[2])), x)
    y = y[1:size(x)[1], :]
    for i in 1:size(y)[1]
        for j in 1:size(y)[2]
            if x[i,j] == 0
                y[i,j] = 0
            end
        end
    end
    return y#=}}}=#
end

"""
    make_graph_dict(y,vert)

Used in the older version of the depth first search algorithm
"""
function make_graph_dict(y,vert)
    connect = Dict{Int32, Vector{Int32}}() #key: node, value: ones its connected to{{{
    weights = Dict{Tuple{Int32, Int32}, Int32}()     #key: Tuple(node1, node2), value: arc weight between nodes 1 and 2
    for row in 1:size(y)[1]
        for column in 1:size(y)[2]
            #at last row and column
            if row==size(y)[1] && column==size(y)[2]
                return connect, weights
            
            #at non existent node (RAS graphs)
            elseif vert[row,column] == 0
                continue
            
            #at last row or no node present (RAS graphs)
            elseif row == size(y)[1] || vert[row+1,column]==0
                connect[vert[row,column]]=[vert[row,column+1]]
                #check if moving right is a node (RAS graphs)
                if vert[row,column+1] != 0
                    weights[vert[row,column], vert[row,column+1]] = y[row, column+1]
                end

            #at last column or no node present (RAS graphs)
            elseif column == size(y)[2] || vert[row,column+1]==0
                connect[vert[row,column]]=[vert[row+1, column]]
            

            else
                connect[vert[row,column]]=[vert[row,column+1],vert[row+1,column]]
                #check if moving right is a node (RAS graphs)
                if vert[row,column+1] != 0
                    weights[vert[row,column],vert[row,column+1]] = y[row,column+1]
                end
            end
        end
    end
    return connect, weights#=}}}=#
end

"""
    make_vert_graph_ras(x)

Used in the older version of the depth-first search algorithm
"""
function make_vert_graph_ras(x)
    vert = Array{Int16}(zeros(size(x)))#={{{=#
    count = 1
    for row in 1:size(x)[1]
        for column in 1:size(x)[2]
            if x[row,column] != 0
                vert[row,column] = count
                count += 1
            end
        end
    end
    max_val = findmax(vert)[1]
    return vert, max_val#=}}}=#
end

"""
    get_index(nelecs, path, weights)

"""
function get_index(nelecs, path, weights)
    index = 1 #={{{=#
    config = Vector{Int32}(zeros(nelecs))
    count = 1

    for i in 1:length(path)-1
        if (path[i],path[i+1]) in keys(weights)
            index += weights[(path[i],path[i+1])]
            config[count]=i
            count += 1
        end
    end
    return index, config#=}}}=#
end


    






