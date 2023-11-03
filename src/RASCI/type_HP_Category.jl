using JLD2
using InCoreIntegrals
using StaticArrays

abstract type HP_Category end

#the hp probably isnt need i dont use it for anything 

struct HP_Category_CA <: HP_Category
    idx::Int #its index
    hp::Tuple{Int, Int} #(holes, particles)
    connected::Vector{Int} #list of allowed pairings of other spin categories
    idxs::Vector{Int}
    shift::Int #shift from local to global indexes
    lookup::Array{Int, 3} #single spin lookup table for single excitations
    cat_lookup::Array{Int, 3} #single spin lookup table for single excitations
end

struct HP_Category_C <: HP_Category
    idx::Int #its index
    hp::Tuple{Int, Int} #(holes, particles)
    connected::Vector{Int} #list of allowed pairings of other spin categories
    idxs::Vector{Int}
    shift::Int #shift from local to global indexes
    lookup::Array{Int, 2} #single spin lookup table for single excitations
end

struct HP_Category_A <: HP_Category
    idx::Int #its index
    hp::Tuple{Int, Int} #(holes, particles)
    connected::Vector{Int} #list of allowed pairings of other spin categories
    idxs::Vector{Int}
    shift::Int #shift from local to global indexes
    lookup::Array{Int, 2} #single spin lookup table for single excitations
end

struct HP_Category_CC <: HP_Category
    idx::Int #its index
    hp::Tuple{Int, Int} #(holes, particles)
    connected::Vector{Int} #list of allowed pairings of other spin categories
    idxs::Vector{Int}
    shift::Int #shift from local to global indexes
    lookup::Array{Int, 3} #single spin lookup table for single excitations
end

struct HP_Category_CCA <: HP_Category
    idx::Int #its index
    hp::Tuple{Int, Int} #(holes, particles)
    connected::Vector{Int} #list of allowed pairings of other spin categories
    idxs::Vector{Int}
    shift::Int #shift from local to global indexes
    lookup::Array{Int, 4} #single spin lookup table for single excitations
end

struct HP_Category_CCAA <: HP_Category
    idx::Int #its index
    hp::Tuple{Int, Int} #(holes, particles)
    connected::Vector{Int} #list of allowed pairings of other spin categories
    idxs::Vector{Int}
    shift::Int #shift from local to global indexes
    lookup::Array{Int, 5} #single spin lookup table for single excitations
end

struct HP_Category_Bra <: HP_Category
    idx::Int
    connected::Vector{Int}
    idxs::Vector{Int}
    shift::Int
end


"""
    HP_Category_CA()

"""
function HP_Category_CA()
    return HP_Category_CA(1,(0,0),Vector{Int}(), Vector{Int}(), Vector{Int}(), Array{Int, 3}())
end

"""
    HP_Category_CA(idx::Int, hp::Tuple{Int,Int}, connected::Vector{Int})

"""
function HP_Category_CA(idx::Int, hp::Tuple{Int,Int}, connected::Vector{Int})
    return HP_Category_CA(idx, hp, Vector{Int}(), Vector{Int}(), connected, Array{Int, 3}())
end

"""
    make_categories(prob::RASCIAnsatz; spin="alpha")

Generates the list of HP_Categories
"""
function make_categories(prob::RASCIAnsatz; spin="alpha")
    categories = ActiveSpaceSolvers.RASCI.generate_spin_categories(prob)#={{{=#
    all_cats = Vector{HP_Category_CA}()
    
    cats_a = deepcopy(categories)
    cats_b = deepcopy(categories)
    fock_list_a, del_at_a = make_fock_from_categories(categories, prob, "alpha")
    deleteat!(cats_a, del_at_a)
    len_cat_a = length(cats_a)
        
    fock_list_b, del_at_b = make_fock_from_categories(categories, prob, "beta")
    deleteat!(cats_b, del_at_b)
    len_cat_b = length(cats_b)

    if spin == "alpha"
        connected = make_spincategory_connections(cats_a, cats_b, prob)

        #compute configs
        as = compute_config_dict(fock_list_a, prob, "alpha")
        #this reverses the config dictionary to get the index as the key 
        rev_as = Dict(value => key for (key, value) in as)
        #as_old =  ActiveSpaceSolvers.RASCI.compute_configs(prob)[1]
        #for i in keys(rev_as)
        #    idx = as_old[i]
        #    rev_as[i] = idx
        #end
        max_a = length(as)
        shift = 0
        for j in 1:len_cat_a
            idxas = Vector{Int}()
            graph_a = make_cat_graphs(fock_list_a[j], prob)
            idxas = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_a, 1, graph_a.max, idxas, rev_as) 
            sort!(idxas)
            lu = zeros(Int, graph_a.no, graph_a.no, length(idxas))
            cat_lu = zeros(Int, graph_a.no, graph_a.no, length(idxas))
            push!(all_cats, HP_Category_CA(j, cats_a[j], connected[j], idxas, shift, lu, cat_lu))
            shift += length(idxas)
        end
        
        #have to do same loop as before bec all categories need initalized for the dfs search for lookup tables
        for k in 1:len_cat_a
            graph_a = make_cat_graphs(fock_list_a[k], prob)
            lu, cat_lu = ActiveSpaceSolvers.RASCI.dfs_single_excitation(graph_a, 1, graph_a.max, all_cats[k].lookup, all_cats[k].cat_lookup, all_cats, rev_as)
            all_cats[k].lookup .= lu
            all_cats[k].cat_lookup .= cat_lu
        end
        return all_cats

    else
        connected = make_spincategory_connections(cats_b, cats_a, prob)
        #compute configs
        bs = compute_config_dict(fock_list_b, prob, "beta")
        rev_bs = Dict(value => key for (key, value) in bs)
        max_b = length(bs)
        shift = 0
        for j in 1:len_cat_b
            idxbs = Vector{Int}()
            graph_b = make_cat_graphs(fock_list_b[j], prob)
            idxbs = ActiveSpaceSolvers.RASCI.dfs_fill_idxs(graph_b, 1, graph_b.max, idxbs, rev_bs) 
            sort!(idxbs)
            lu = zeros(Int, graph_b.no, graph_b.no, length(idxbs))
            cat_lu = zeros(Int, graph_b.no, graph_b.no, length(idxbs))
            push!(all_cats, HP_Category_CA(j, cats_b[j], connected[j], idxbs, shift, lu, cat_lu))
            shift += length(idxbs)
        end

        for k in 1:len_cat_b
            graph_b = make_cat_graphs(fock_list_b[k], prob)
            lu, cat_lu = ActiveSpaceSolvers.RASCI.dfs_single_excitation(graph_b, 1, graph_b.max, all_cats[k].lookup,all_cats[k].cat_lookup, all_cats, rev_bs)
            all_cats[k].lookup .= lu
            all_cats[k].cat_lookup .= cat_lu
        end
        return all_cats
    end#=}}}=#
end

function make_new_categories(prob::RASCIAnsatz; spin="alpha")
    all_cats = Vector{HP_Category_CA}()
    x = [(0, 0), (0, 1), (1, 0), (1, 1)] 
    if spin == "alpha"
        #this reverses the config dictionary to get the index as the key 
        for j in 1:length(prob.cats_a)
            idxas = prob.cats_a[j][3]
            lu, cat_lu = ActiveSpaceSolvers.RASCI.fill_single_excitation_lu(prob.a_configs, idxas, prob.no, prob.cats_a)
            push!(all_cats, HP_Category_CA(j, (0,0), prob.cats_a[j][1], idxas, prob.cats_a[j][2], lu, cat_lu))
        end
        return all_cats

    else
        for j in 1:length(prob.cats_b)
            idxbs = prob.cats_b[j][3]
            lu, cat_lu = ActiveSpaceSolvers.RASCI.fill_single_excitation_lu(prob.b_configs, idxbs, prob.no, prob.cats_b)
            push!(all_cats, HP_Category_CA(j, (0,0), prob.cats_b[j][1], idxbs, prob.cats_b[j][2], lu, cat_lu))
        end
        return all_cats
    end#=}}}=#
end

function fill_single_excitation_lu(config_dict, idxs, no, cats) #cats is either prob.cats_a or prob.cats_b
    lu = zeros(Int, no, no, length(idxs))
    cat_lu = zeros(Int, no, no, length(idxs))
    rev_configs = Dict(value => key for (key, value) in config_dict)
    cat_org = find_cat(idxs[1], cats)
    for i in idxs
        config = rev_configs[i]
        idx_loc = i-cats[cat_org][2]
        for orb in config
            for orb_c in 1:no
                sign_a, new = ActiveSpaceSolvers.RASCI.apply_a_dumb(config, orb)
                sign_c, final_c = ActiveSpaceSolvers.RASCI.apply_c_dumb(new, orb_c)
                if haskey(config_dict, final_c)
                    idx = config_dict[final_c]
                    cat = find_cat(idx, cats)
                    sgn = sign_a*sign_c
                    lu[orb, orb_c, idx_loc] = sgn*idx
                    cat_lu[orb, orb_c, idx_loc] = cat
                end
            end
        end
    end
    return lu, cat_lu
end


function find_cat(idx::Int, cats::Vector{Tuple{Vector{Int}, Int, Vector{Int}}})
    for cat in 1:length(cats)
        if idx in cats[cat][3]
            return cat
        else
            continue
        end
    end
end


"""
    compute_config_dict(fock_list, prob::RASCIAnsatz, spin="alpha")

"""
function compute_config_dict(fock_list, prob::RASCIAnsatz, spin="alpha")
    configs_dict = Dict{Int, Vector{Int32}}()#={{{=#
    configs = []
    for j in 1:length(fock_list)
        graph_a = ActiveSpaceSolvers.RASCI.make_cat_graphs(fock_list[j], prob)
        if spin == "alpha"
            config_dict = ActiveSpaceSolvers.RASCI.old_dfs(prob.na, graph_a.connect, graph_a.weights, 1, graph_a.max)
        else
            config_dict = ActiveSpaceSolvers.RASCI.old_dfs(prob.nb, graph_a.connect, graph_a.weights, 1, graph_a.max)
        end
        for x in keys(config_dict)
            push!(configs, x)
        end
        
    end
    
    for i in 1:length(configs)
        configs_dict[i] = configs[i]
    end
    return configs_dict#=}}}=#
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
    make_category_connections(categories, prob::RASCIAnsatz)

"""
function make_category_connections(categories, prob::RASCIAnsatz)
    #={{{=#
    connected = Vector{Vector{Int}}()
    for i in 1:length(categories)
        tmp = Vector{Int}()
        for j in 1:length(categories)
            if categories[i][1]+categories[j][1] <= prob.max_h
                if categories[i][2]+categories[j][2] <= prob.max_p
                    append!(tmp, j)
                end
            end
        end
        append!(connected, [tmp])
    end
    return connected#=}}}=#
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
    #graph = ActiveSpaceSolvers.RASCI.RASCI_OlsenGraph(prob.no, sum(fock_list), prob.ras_spaces, prob.ras1_min, prob.ras3_max, max_val, vert, connect, weights)
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
    find_cat(idx::Int, categories::Vector{<:HP_Category})

Doesn't get used often but will find the HP_cateogry from a given index
"""
function find_cat(idx::Int, categories::Vector{<:HP_Category})
    #this function will find the category that idx belongs to{{{
    for cat in categories
        if idx in cat.idxs
            return cat
        else
            continue
        end
    end
    return 0#=}}}=#
end

"""
    find_cat(idx::Int, categories::Vector{<:HP_Category})

Doesn't get used often but will find the HP_cateogry from a given index
"""
function find_cat(tmp::Tuple{Int, Int}, categories::Vector{<:HP_Category})
    #this function will find the category that idx belongs to{{{
    for cat in categories
        if tmp == cat.hp
            return cat
        else
            continue
        end
    end
    return 0#=}}}=#
end



