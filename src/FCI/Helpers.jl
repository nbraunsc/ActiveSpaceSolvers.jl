"""
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


# binom_coeff = Array{Int,2}(undef,31,31)
# for i in 0:size(binom_coeff,2)-1
#     for j in i:size(binom_coeff,1)-1
#         binom_coeff[j+1,i+1] = calc_nchk(j,i)
#     end
# end
binom_coeff = Array{Int,2}(undef,1000,1000)
for i in 0:size(binom_coeff,2)-1
    for j in i:size(binom_coeff,1)-1
        binom_coeff[j+1,i+1] = calc_nchk(j,i)
        if calc_nchk(j,i)>1E12
            break
        end
        
    end
end
"""
    get_nchk(n::Integer,k::Integer)

Looks up binomial coefficient from a precomputed table: n choose k
"""
@inline function get_nchk(n,k)
    return binom_coeff[n+1,k+1]
end


### 
function string_to_index(str::String)
    return parse(Int, reverse(str); base=2)
end

function index_to_string(index::Int)
    return [parse(Int, ss) for ss in reverse(bitstring(index))]
end





