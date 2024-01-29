"""
Type to organize ras subspace configurations in the form of DeterminantString
"""
mutable struct SubspaceDeterminantString
    no::Int
    ne::Int
    config::Vector{Int}
    lin_index::Int
    max::Int
end

"""#={{{=#
    calc_nchk(n::Integer,k::Integer)

Calculates binomial coefficient: n choose k
"""
function calc_nchk(n::Integer,k::Integer)
    accum::Int = 1
    for i in 1:k
        accum = accum * (n-k+i) ÷ i
    end
    return accum
end

binom_coeff = Array{Int,2}(undef,100,100)
fill!(binom_coeff, -1)
for i in 0:size(binom_coeff,2)-1
    for j in i:size(binom_coeff,1)-1
        binom_coeff[j+1,i+1] = calc_nchk(j,i)
    end
end

"""
    get_nchk(n::Integer,k::Integer)

Looks up binomial coefficient from a precomputed table: n choose k
"""
@inline function get_nchk(n,k)
    return binom_coeff[n+1,k+1]
end#=}}}=#

function SubspaceDeterminantString(no::Int, ne::Int)
    return SubspaceDeterminantString(no, ne, Vector(1:ne), 1, get_nchk(no,ne))
end

function SubspaceDeterminantString(no::Int, ne::Int, config::Vector{Int})
    return SubspaceDeterminantString(no, ne, config, calc_linear_index!(no, ne, config), get_nchk(no,ne))
end

function incr_comb!(comb::Array{Int,1}, Mend::Int)
    #=
    For a given combination, form the next combination
    =#
    #={{{=#
    N = length(comb)
    for i in N:-1:1
        if comb[i] < Mend - N + i
            comb[i] += 1
            for j in i+1:N
                comb[j]=comb[j-1]+1
            end
            return
        end
    end
    return
end
#=}}}=#

function incr!(c::SubspaceDeterminantString)
    #=
    Increment determinant DeterminantString
    =#
    if c.max == nothing
        calc_max!(c)
    end
    if c.lin_index == c.max
        return
    end
    c.lin_index += 1
    incr_comb!(c.config, c.no)
end

function calc_max!(c::SubspaceDeterminantString)
    #=
    Calculate dimension of space accessible to a DeterminantString
    =#
    c.max = get_nchk(c.no,c.ne)
end

"""
    calc_linear_index!(no::Int, ne::Int, config::Vector{Int})

Calculate the linear index
"""
function calc_linear_index!(no::Int, ne::Int, config::Vector{Int})
    #={{{=#
    lin_index = 1
    v_prev::Int = 0

    for i::Int in 1:ne
        v = config[i]
        for j::Int in v_prev+1:v-1
            lin_index += binom_coeff[no-j+1,ne-i+1]
        end
        v_prev = v
    end
    return lin_index
end
#=}}}=#


"""
    calc_linear_index!(c::DeterminantString)

Calculate the linear index
"""
function calc_linear_index!(c::SubspaceDeterminantString)
    #={{{=#
    c.lin_index = 1
    v_prev::Int = 0

    for i::Int in 1:c.ne
        v = c.config[i]
        for j::Int in v_prev+1:v-1
            c.lin_index += binom_coeff[c.no-j+1,c.ne-i+1]
            #@btime $c.lin_index += $binom_coeff[$c.no-$j+1,$c.ne-$i+1]
            #c.lin_index += get_nchk(c.no-j,c.ne-i)
        end
        v_prev = v
    end
end
#=}}}=#

"""
    calc_linear_index!(c::SubspaceDeterminantString, binomcoeff::Array{Int,2})

Calculate the linear index, passing in binomial coefficient matrix makes it much faster
"""
function calc_linear_index!(c::SubspaceDeterminantString, binomcoeff::Array{Int,2})
    c.lin_index = 1
    v_prev::Int = 0

    for i::Int in 1:c.ne
        v = c.config[i]
        for j::Int in v_prev+1:v-1
            c.lin_index += binomcoeff[c.no-j+1,c.ne-i+1]
        end
        v_prev = v
    end
end

function calc_full_ras_index(c1::SubspaceDeterminantString, c2::SubspaceDeterminantString, c3::SubspaceDeterminantString)
    calc_linear_index!(c1)
    calc_linear_index!(c2)
    calc_linear_index!(c3)

    idx = c1.lin_index + (c2.lin_index-1)*c1.max + (c3.lin_index-1)*c1.max*c2.max
    return idx
end



