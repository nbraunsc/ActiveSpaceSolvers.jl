#using InCoreIntegrals
using ActiveSpaceSolvers
using QCBase
import LinearMaps
using OrderedCollections
using BlockDavidson
using StaticArrays
using LinearAlgebra
using Printf
using TimerOutputs

#Optional{T} = Union{T, Nothing}

struct DDCIAnsatz <: Ansatz
    no::Int
    na::Int  # number of alpha
    nb::Int  # number of beta
    dim::Int
    ras_spaces::SVector{3, Int}   # Number of orbitals in each ras space (RAS1, RAS2, RAS3)
    ex_level::Int
end

"""
    DDCI(no, na, nb, ras_spaces::Any, ex_level)
Constructor
# Arguments
- `no`: Number of spatial orbitals
- `na`: Number of α electrons
- `nb`: Number of β electrons
- `ras_spaces`: Number of orbitals in each (RAS1, RAS2, RAS3)
- `ex_level`: DDCI excitation level (1,2,or 3)
"""
function Base.display(p::DDCIAnsatz)
    @printf(" DDCI:: #Orbs = %-3i #α = %-2i #β = %-2i Fock Spaces: (%i, %i, %i) RASCI Dimension: %-3i DDCI_EX_Level: %-2i \n",p.no,p.na,p.nb,p.ras_spaces[1], p.ras_spaces[2], p.ras_spaces[3],p.dim, p.ex_level)
end

function DDCIAnsatz(no::Int, na, nb, ras_spaces::Any; ex_level=1)
    na <= no || throw(DimensionMismatch)
    nb <= no || throw(DimensionMismatch)
    sum(ras_spaces) == no || throw(DimensionMismatch)
    ras_spaces = convert(SVector{3,Int},collect(ras_spaces))
    na = convert(Int, na)
    nb = convert(Int, nb)
    if ex_level==1
        h = Int8(1)
        p = Int8(0)
        h2 = Int8(0)
        p2 = Int8(1)
        rdim = calc_rdim_ddci(ras_spaces, na, nb, h, p, h2, p2)
        return DDCIAnsatz(no, na, nb, rdim, ras_spaces, ex_level);
    
    elseif ex_level==2
        h = Int8(2)
        p = Int8(0)
        h2 = Int8(0)
        p2 = Int8(2)
        h3 = Int8(1)
        p3 = Int8(1)
        rdim = calc_rdim_ddci(ras_spaces, na, nb, h, p, h2, p2, h3, p3)
        return DDCIAnsatz(no, na, nb, rdim, ras_spaces, ex_level);

    elseif ex_level==3
        h = Int8(2)
        p = Int8(1)
        h2 = Int8(1)
        p2 = Int8(2)
        rdim = calc_rdim_ddci(ras_spaces, na, nb, h, p, h2, p2)
        return DDCIAnsatz(no, na, nb, rdim, ras_spaces, ex_level);
    end
end

"""
    LinearMap(ints, prb::RASCIAnsatz_2)

Get LinearMap with takes a vector and returns action of H on that vector

# Arguments
- ints: `InCoreInts` object
- prb:  `RASCIAnsatz` object
"""
function LinearMaps.LinearMap(ints::InCoreInts{T}, prob::DDCIAnsatz) where {T}
    next_ddci,h,p = find_full_ddci(prob)
    rasvec2 = ActiveSpaceSolvers.RASCI_2.fill_lu_helper(next_ddci, h, p)
    @time lu2 = ActiveSpaceSolvers.RASCI_2.fill_lu(rasvec2, prob.ras_spaces)

    ras_help = fill_lu_helper_ddci(prob)
    @time lu = ActiveSpaceSolvers.RASCI_2.fill_lu(ras_help, prob.ras_spaces)

    iters = 0
    function mymatvec(v)
        rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(v, prob)
        #lu = ActiveSpaceSolvers.RASCI_2.fill_lu(rasvec, prob.ras_spaces)

        iters += 1
        #@printf(" Iter: %4i", iters)
        print("Iter: ", iters, " ")
        #@printf(" %-50s", "Compute sigma 1: ")
        #flush(stdout)
        #display(size(v))
       
        nr = 0
        if length(size(v)) == 1
            nr = 1
        else 
            nr = size(v)[2]
        end
        
        sigma1 = ActiveSpaceSolvers.RASCI_2.sigma_one(rasvec, ints, prob.ras_spaces, lu2)
        sigma2 = ActiveSpaceSolvers.RASCI_2.sigma_two(rasvec, ints, prob.ras_spaces, lu2)
        sigma3 = ActiveSpaceSolvers.RASCI_2.sigma_three(rasvec, ints, prob.ras_spaces, lu)
        
        #@time sigma1 = ActiveSpaceSolvers.RASCI_2.sigma_one(rasvec, ints, prob.ras_spaces, lu2)
        #@time sigma2 = ActiveSpaceSolvers.RASCI_2.sigma_two(rasvec, ints, prob.ras_spaces, lu2)
        #@time sigma3 = ActiveSpaceSolvers.RASCI_2.sigma_three(rasvec, ints, prob.ras_spaces, lu)
        
        sig = sigma1 + sigma2 + sigma3
        sig .+= ints.h0*v
        return sig
    end
    return LinearMap(mymatvec, prob.dim, prob.dim, issymmetric=true, ismutating=false, ishermitian=true)
end

"""
    LinOpMat(ints, prb::A)

Get LinearMap with takes a vector and returns action of H on that vector

# Arguments
- ints: `InCoreInts` object
- prb:  `A` object
"""
function BlockDavidson.LinOpMat(ints::InCoreInts{T}, prob::DDCIAnsatz) where {T}
    next_ddci,h,p = find_full_ddci(prob)
    rasvec2 = ActiveSpaceSolvers.RASCI_2.fill_lu_helper(next_ddci, h, p)
    @time lu2 = ActiveSpaceSolvers.RASCI_2.fill_lu(rasvec2, prob.ras_spaces)

    ras_help = fill_lu_helper_ddci(prob)
    @time lu = ActiveSpaceSolvers.RASCI_2.fill_lu(ras_help, prob.ras_spaces)

    iters = 0
    function mymatvec(v)
        rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(v, prob)
        #@time lu = ActiveSpaceSolvers.RASCI_2.fill_lu(rasvec, prob.ras_spaces)

        iters += 1
        #@printf(" Iter: %4i", iters)
        #print("Iter: ", iters, " ")
        #flush(stdout)
        #display(size(v))
       
        nr = 0
        if length(size(v)) == 1
            nr = 1
        else 
            nr = size(v)[2]
        end
        
        @time sigma1 = ActiveSpaceSolvers.RASCI_2.sigma_one(rasvec, ints, prob.ras_spaces, lu2)
        @time sigma2 = ActiveSpaceSolvers.RASCI_2.sigma_two(rasvec, ints, prob.ras_spaces, lu2)
        @time sigma3 = ActiveSpaceSolvers.RASCI_2.sigma_three(rasvec, ints, prob.ras_spaces, lu)
        
        sig = sigma1 + sigma2 + sigma3
        
        sig .+= ints.h0*v
        return sig
    end
    return LinOpMat{T}(mymatvec, prob.dim, true)
end

function calc_rdim_ddci(ras_spaces::SVector{3, Int}, na::Int, nb::Int, h::Int8, p::Int8, h2::Int8, p2::Int8)
    a_blocks, fock_as = ActiveSpaceSolvers.RASCI_2.make_blocks(ras_spaces, na, h, p)#={{{=#
    b_blocks, fock_bs = ActiveSpaceSolvers.RASCI_2.make_blocks(ras_spaces, nb, h, p)
    tmp = []
    
    start = 0
    for i in 1:length(a_blocks)
        dima = binomial(ras_spaces[1], fock_as[i][1])*binomial(ras_spaces[2], fock_as[i][2])*binomial(ras_spaces[3], fock_as[i][3])
        for j in 1:length(b_blocks)
            dimb = binomial(ras_spaces[1], fock_bs[j][1])*binomial(ras_spaces[2], fock_bs[j][2])*binomial(ras_spaces[3], fock_bs[j][3])
            if a_blocks[i][1]+b_blocks[j][1]<= h
                if a_blocks[i][2]+b_blocks[j][2] <= p
                    block1 = ActiveSpaceSolvers.RASCI_2.RasBlock(fock_as[i], fock_bs[j])
                    push!(tmp, block1)
                    start += dima*dimb
                end
            end
        end
    end
    
    a_blocks2, fock_as2 = ActiveSpaceSolvers.RASCI_2.make_blocks(ras_spaces, na, h2, p2)
    b_blocks2, fock_bs2 = ActiveSpaceSolvers.RASCI_2.make_blocks(ras_spaces, nb, h2, p2)
    for i in 1:length(a_blocks2)
        dima = binomial(ras_spaces[1], fock_as2[i][1])*binomial(ras_spaces[2], fock_as2[i][2])*binomial(ras_spaces[3], fock_as2[i][3])
        for j in 1:length(b_blocks2)
            dimb = binomial(ras_spaces[1], fock_bs2[j][1])*binomial(ras_spaces[2], fock_bs2[j][2])*binomial(ras_spaces[3], fock_bs2[j][3])
            if a_blocks2[i][1]+b_blocks2[j][1]<= h2
                if a_blocks2[i][2]+b_blocks2[j][2] <= p2
                    block1 = ActiveSpaceSolvers.RASCI_2.RasBlock(fock_as2[i], fock_bs2[j])
                    if block1 in tmp
                        continue
                    else
                        push!(tmp, block1)
                        start += dima*dimb
                    end
                end
            end
        end
    end#=}}}=#
    return start
end

function calc_rdim_ddci(ras_spaces::SVector{3, Int}, na::Int, nb::Int, h::Int8, p::Int8, h2::Int8, p2::Int8, h3::Int8, p3::Int8)
    a_blocks, fock_as = ActiveSpaceSolvers.RASCI_2.make_blocks(ras_spaces, na, h, p)#={{{=#
    b_blocks, fock_bs = ActiveSpaceSolvers.RASCI_2.make_blocks(ras_spaces, nb, h, p)
    tmp = []
    
    start = 0
    for i in 1:length(a_blocks)
        dima = binomial(ras_spaces[1], fock_as[i][1])*binomial(ras_spaces[2], fock_as[i][2])*binomial(ras_spaces[3], fock_as[i][3])
        for j in 1:length(b_blocks)
            dimb = binomial(ras_spaces[1], fock_bs[j][1])*binomial(ras_spaces[2], fock_bs[j][2])*binomial(ras_spaces[3], fock_bs[j][3])
            if a_blocks[i][1]+b_blocks[j][1]<= h
                if a_blocks[i][2]+b_blocks[j][2] <= p
                    block1 = ActiveSpaceSolvers.RASCI_2.RasBlock(fock_as[i], fock_bs[j])
                    push!(tmp, block1)
                    start += dima*dimb
                end
            end
        end
    end
    
    a_blocks2, fock_as2 = ActiveSpaceSolvers.RASCI_2.make_blocks(ras_spaces, na, h2, p2)
    b_blocks2, fock_bs2 = ActiveSpaceSolvers.RASCI_2.make_blocks(ras_spaces, nb, h2, p2)
    for i in 1:length(a_blocks2)
        dima = binomial(ras_spaces[1], fock_as2[i][1])*binomial(ras_spaces[2], fock_as2[i][2])*binomial(ras_spaces[3], fock_as2[i][3])
        for j in 1:length(b_blocks2)
            dimb = binomial(ras_spaces[1], fock_bs2[j][1])*binomial(ras_spaces[2], fock_bs2[j][2])*binomial(ras_spaces[3], fock_bs2[j][3])
            if a_blocks2[i][1]+b_blocks2[j][1]<= h2
                if a_blocks2[i][2]+b_blocks2[j][2] <= p2
                    block1 = ActiveSpaceSolvers.RASCI_2.RasBlock(fock_as2[i], fock_bs2[j])
                    if block1 in tmp
                        continue
                    else
                        push!(tmp, block1)
                        start += dima*dimb
                    end
                end
            end
        end
    end
    
    a_blocks2, fock_as2 = ActiveSpaceSolvers.RASCI_2.make_blocks(ras_spaces, na, h3, p3)
    b_blocks2, fock_bs2 = ActiveSpaceSolvers.RASCI_2.make_blocks(ras_spaces, nb, h3, p3)
    for i in 1:length(a_blocks2)
        dima = binomial(ras_spaces[1], fock_as2[i][1])*binomial(ras_spaces[2], fock_as2[i][2])*binomial(ras_spaces[3], fock_as2[i][3])
        for j in 1:length(b_blocks2)
            dimb = binomial(ras_spaces[1], fock_bs2[j][1])*binomial(ras_spaces[2], fock_bs2[j][2])*binomial(ras_spaces[3], fock_bs2[j][3])
            if a_blocks2[i][1]+b_blocks2[j][1]<= h3
                if a_blocks2[i][2]+b_blocks2[j][2] <= p3
                    block1 = ActiveSpaceSolvers.RASCI_2.RasBlock(fock_as2[i], fock_bs2[j])
                    if block1 in tmp
                        continue
                    else
                        push!(tmp, block1)
                        start += dima*dimb
                    end
                end
            end
        end
    end#=}}}=#
    return start
end


function find_full_ddci(prob::DDCIAnsatz)
    if prob.ex_level == 1
        h = Int8(1)
        p = Int8(1)
        h2 = Int8(0)
        p2 = Int8(0)
        rdim = calc_rdim_ddci(prob.ras_spaces, prob.na, prob.nb, h, p, h2, p2)
        return DDCIAnsatz(prob.no, prob.na, prob.nb, rdim, prob.ras_spaces, 10), h, p
    elseif prob.ex_level ==2 || prob.ex_level==3
        h = Int8(2)
        p = Int8(2)
        h2 = Int8(0)
        p2 = Int8(0)
        rdim = calc_rdim_ddci(prob.ras_spaces, prob.na, prob.nb, h, p, h2, p2)
        return DDCIAnsatz(prob.no, prob.na, prob.nb, rdim, prob.ras_spaces, 20), h, p
    end
end
    

function ActiveSpaceSolvers.apply_sminus(v::Matrix, ansatz::DDCIAnsatz)
    if ansatz.nb + 1 > ansatz.no#={{{=#
        error(" Can't decrease Ms further")
    end
    # Sm = b'a
    # = c(IJ,s) <IJ|b'a|KL> c(KL,t)
    # = c(IJ,s)c(KL,t) <J|<I|b'a|K>|L>
    # = c(IJ,s)c(KL,t) <J|<I|ab'|K>|L> (-1)
    # = c(IJ,s)c(KL,t) <J|<I|a|K>b'|L> (-1) (-1)^ket_a.ne
    # = c(IJ,s)c(KL,t) <I|a|K><J|b'|L> (-1) (-1)^ket_a.ne
    
    nroots = size(v,2)
    v2 = ActiveSpaceSolvers.RASCI_2.RASVector(v, ansatz)
    
    sgnK = -1
    if ansatz.na % 2 != 0 
        sgnK = -sgnK
    end

    bra_ansatz = DDCIAnsatz(ansatz.no, ansatz.na-1, ansatz.nb+1, ansatz.ras_spaces, ex_level=ansatz.ex_level)
    wtmp = ActiveSpaceSolvers.RASCI_2.RASVector(zeros(bra_ansatz.dim, nroots), bra_ansatz)
    w = initalize_sig(wtmp)
    
    create_list = make_excitation_classes_c(ansatz.ras_spaces)
    ann_list = make_excitation_classes_a(ansatz.ras_spaces)
    ras1, ras2, ras3 = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(ansatz.ras_spaces)
    #note ras1 is same as ras1_bra so do not need to compute them
    
    for (block1, vec) in v2.data
        #loop over alpha strings
        idxa = 0
        det3a = SubspaceDeterminantString(ansatz.ras_spaces[3], block1.focka[3])
        for na in 1:det3a.max
            det2a = SubspaceDeterminantString(ansatz.ras_spaces[2], block1.focka[2])
            for ja in 1:det2a.max
                det1a = SubspaceDeterminantString(ansatz.ras_spaces[1], block1.focka[1])
                for ia in 1:det1a.max
                    idxa += 1
                    aconfig = [det1a.config;det2a.config.+det1a.no;det3a.config.+det1a.no.+det2a.no]

                    #now beta strings
                    idxb=0
                    det3b = SubspaceDeterminantString(ansatz.ras_spaces[3], block1.fockb[3])
                    for n in 1:det3b.max
                        det2b = SubspaceDeterminantString(ansatz.ras_spaces[2], block1.fockb[2])
                        for j in 1:det2b.max
                            det1b = SubspaceDeterminantString(ansatz.ras_spaces[1], block1.fockb[1])
                            for i in 1:det1b.max
                                idxb += 1
                                bconfig = [det1b.config;det2b.config.+det1b.no;det3b.config.+det1b.no.+det2b.no]
                                for (p_range, delta_c) in create_list
                                    for (q_range, delta_a) in ann_list
                                        block2 = RasBlock(block1.focka.+delta_a, block1.fockb.+delta_c)
                                        haskey(w, block2) || continue
                                        for q in q_range
                                            tmp = deepcopy(aconfig)
                                            if q in tmp
                                                sgn_q, det_a = apply_annihilation(tmp, q)
                                                sgn_q != 0 || continue
                                                d1_a, d2_a, d3_a = breakup_config(det_a, ras1, ras2, ras3)
                                                det1_q = SubspaceDeterminantString(length(ras1), length(d1_a), d1_a)
                                                det2_q = SubspaceDeterminantString(length(ras2), length(d2_a), d2_a.-length(ras1))
                                                det3_q = SubspaceDeterminantString(length(ras3), length(d3_a), d3_a.-length(ras1).-length(ras2))

                                                idxa_new = calc_full_ras_index(det1_q, det2_q, det3_q)
                                                for p in p_range
                                                    p == q || continue
                                                    tmp2 = deepcopy(bconfig)
                                                    if p in tmp2
                                                        continue
                                                    else
                                                        d1_c, d2_c, d3_c = breakup_config(tmp2, ras1, ras2, ras3)
                                                        sgn_p, idxb_new = apply_creation!(d1_c, d2_c, d3_c, ras1, ras2, ras3, p)
                                                        sgn_p != 0 || continue

                                                        w[block2][idxa_new, idxb_new,:] .+= sgnK*sgn_q*sgn_p*vec[idxa,idxb,:]
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                                incr!(det1b)
                            end
                            incr!(det2b)
                        end
                        incr!(det3b)
                    end
                    incr!(det1a)
                end
                incr!(det2a)
            end
            incr!(det3a)
        end
    end

    starti = 1
    w2 = zeros(Float64, bra_ansatz.dim, nroots)
    for (block, vec) in w
        tmp = reshape(vec, (size(vec,1)*size(vec,2), nroots))
        w2[starti:starti+(size(vec,1)*size(vec,2))-1, :] .= tmp
        starti += (size(vec,1)*size(vec,2))
    end
    
    #only keep the states that aren't zero (that weren't killed by S-)
    wout = zeros(size(w2,1),0)
    for i in 1:nroots
        ni = norm(w2[:,i])
        if isapprox(ni, 0, atol=1e-4) == false
            wout = hcat(wout, w2[:,i]./ni)
        end
    end

    return wout, bra_ansatz#=}}}=#
end

"""
"""
function ActiveSpaceSolvers.apply_splus(v::Matrix, ansatz::DDCIAnsatz)

    # Sp = a'b{{{
    # = c(IJ,s) <IJ|a'b|KL> c(KL,t)
    # = c(IJ,s)c(KL,t) <J|<I|a'b|K>|L>
    # = c(IJ,s)c(KL,t) <J|<I|a'|K>b|L> (-1)^ket_a.ne
    # = c(IJ,s)c(KL,t) <I|a'|K><J|b|L> (-1)^ket_a.ne

    nroots = size(v,2)
    v2 = ActiveSpaceSolvers.RASCI_2.RASVector(v, ansatz)
    
    if ansatz.na + 1 > ansatz.no
        error(" Can't increase Ms further")
    end

    sgnK = 1
    if ansatz.na % 2 != 0 
        sgnK = -sgnK
    end
    
    bra_ansatz = DDCIAnsatz(ansatz.no, ansatz.na+1, ansatz.nb-1, ansatz.ras_spaces, ex_level=ansatz.ex_level)
    wtmp = ActiveSpaceSolvers.RASCI_2.RASVector(zeros(bra_ansatz.dim, nroots), bra_ansatz)
    w = initalize_sig(wtmp)
    
    create_list = make_excitation_classes_c(ansatz.ras_spaces)
    ann_list = make_excitation_classes_a(ansatz.ras_spaces)
    ras1, ras2, ras3 = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(ansatz.ras_spaces)
    #note ras1 is same as ras1_bra so do not need to compute them
    
    for (block1, vec) in v2.data
        #loop over alpha strings
        idxa = 0
        det3a = SubspaceDeterminantString(ansatz.ras_spaces[3], block1.focka[3])
        for na in 1:det3a.max
            det2a = SubspaceDeterminantString(ansatz.ras_spaces[2], block1.focka[2])
            for ja in 1:det2a.max
                det1a = SubspaceDeterminantString(ansatz.ras_spaces[1], block1.focka[1])
                for ia in 1:det1a.max
                    idxa += 1
                    aconfig = [det1a.config;det2a.config.+det1a.no;det3a.config.+det1a.no.+det2a.no]

                    #now beta strings
                    idxb=0
                    det3b = SubspaceDeterminantString(ansatz.ras_spaces[3], block1.fockb[3])
                    for n in 1:det3b.max
                        det2b = SubspaceDeterminantString(ansatz.ras_spaces[2], block1.fockb[2])
                        for j in 1:det2b.max
                            det1b = SubspaceDeterminantString(ansatz.ras_spaces[1], block1.fockb[1])
                            for i in 1:det1b.max
                                idxb += 1
                                bconfig = [det1b.config;det2b.config.+det1b.no;det3b.config.+det1b.no.+det2b.no]
                                for (p_range, delta_c) in create_list
                                    for (q_range, delta_a) in ann_list
                                        block2 = RasBlock(block1.focka.+delta_c, block1.fockb.+delta_a)
                                        haskey(w, block2) || continue
                                        for q in q_range
                                            tmp = deepcopy(bconfig)
                                            if q in tmp
                                                sgn_q, det_a = apply_annihilation(tmp, q)
                                                sgn_q != 0 || continue
                                                d1_a, d2_a, d3_a = breakup_config(det_a, ras1, ras2, ras3)
                                                det1_q = SubspaceDeterminantString(length(ras1), length(d1_a), d1_a)
                                                det2_q = SubspaceDeterminantString(length(ras2), length(d2_a), d2_a.-length(ras1))
                                                det3_q = SubspaceDeterminantString(length(ras3), length(d3_a), d3_a.-length(ras1).-length(ras2))

                                                idxb_new = calc_full_ras_index(det1_q, det2_q, det3_q)
                                                for p in p_range
                                                    p == q || continue
                                                    tmp2 = deepcopy(aconfig)
                                                    if p in tmp2
                                                        continue
                                                    else
                                                        d1_c, d2_c, d3_c = breakup_config(tmp2, ras1, ras2, ras3)
                                                        sgn_p, idxa_new = apply_creation!(d1_c, d2_c, d3_c, ras1, ras2, ras3, p)
                                                        sgn_p != 0 || continue

                                                        w[block2][idxa_new, idxb_new,:] .+= sgnK*sgn_q*sgn_p*vec[idxa,idxb,:]
                                                    end
                                                end
                                            end
                                        end
                                    end
                                end
                                incr!(det1b)
                            end
                            incr!(det2b)
                        end
                        incr!(det3b)
                    end
                    incr!(det1a)
                end
                incr!(det2a)
            end
            incr!(det3a)
        end
    end

    starti = 1
    w2 = zeros(Float64, bra_ansatz.dim, nroots)
    for (block, vec) in w
        tmp = reshape(vec, (size(vec,1)*size(vec,2), nroots))
        w2[starti:starti+(size(vec,1)*size(vec,2))-1, :] .= tmp
        starti += (size(vec,1)*size(vec,2))
    end
    
    #only keep the states that aren't zero (that weren't killed by S-)
    wout = zeros(size(w2,1),0)
    for i in 1:nroots
        ni = norm(w2[:,i])
        display(ni)
        if isapprox(ni, 0, atol=1e-4) == false
            wout = hcat(wout, w2[:,i]./ni)
        end
    end

    return wout, bra_ansatz#=}}}=#
end

"""
    build_H_matrix(ints, P::A)

Build the Hamiltonian defined by `ints` in the Slater Determinant Basis  specified by `P`
"""
function ActiveSpaceSolvers.build_H_matrix(ints::InCoreInts, prob::DDCIAnsatz) 
    nr = prob.dim
    v = Matrix(1.0I, nr, nr)
    rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(v, prob)
    lu = ActiveSpaceSolvers.RASCI_2.fill_lu(rasvec, prob.ras_spaces)
    next_ddci,h,p = find_full_ddci(prob)
    lu2 = fill_lu_ddci(next_ddci, h, p, nr, prob.ras_spaces)
    
    sigma1 = ActiveSpaceSolvers.RASCI_2.sigma_one(rasvec, ints, prob.ras_spaces, lu2)
    sigma2 = ActiveSpaceSolvers.RASCI_2.sigma_two(rasvec, ints, prob.ras_spaces, lu2)
    sigma3 = ActiveSpaceSolvers.RASCI_2.sigma_three(rasvec, ints, prob.ras_spaces, lu)

    sig = sigma1 + sigma2 + sigma3

    Hmat = .5*(sig+sig')
    Hmat += 1.0I*ints.h0
    return Hmat
end

function fill_lu_helper_ddci(prob::DDCIAnsatz) 
    h=Int8(0)
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
    rasvec =Vector{Tuple{ActiveSpaceSolvers.RASCI_2.RasBlock, Int, Int}}()

    a_blocks, fock_as = ActiveSpaceSolvers.RASCI_2.make_blocks(prob.ras_spaces, prob.na, h, p)
    b_blocks, fock_bs = ActiveSpaceSolvers.RASCI_2.make_blocks(prob.ras_spaces, prob.nb, h, p)

    for i in 1:length(a_blocks)
        dima = binomial(prob.ras_spaces[1], fock_as[i][1])*binomial(prob.ras_spaces[2], fock_as[i][2])*binomial(prob.ras_spaces[3], fock_as[i][3])
        for j in 1:length(b_blocks)
            dimb = binomial(prob.ras_spaces[1], fock_bs[j][1])*binomial(prob.ras_spaces[2], fock_bs[j][2])*binomial(prob.ras_spaces[3], fock_bs[j][3])
            if a_blocks[i][1]+b_blocks[j][1]<= h
                if a_blocks[i][2]+b_blocks[j][2] <= p
                    block1 = ActiveSpaceSolvers.RASCI_2.RasBlock(fock_as[i], fock_bs[j])
                    push!(rasvec, (block1, dima, dimb))
                end
            end
        end
    end
    
    a_blocks2, fock_as2 = ActiveSpaceSolvers.RASCI_2.make_blocks(prob.ras_spaces, prob.na, h2, p2)
    b_blocks2, fock_bs2 = ActiveSpaceSolvers.RASCI_2.make_blocks(prob.ras_spaces, prob.nb, h2, p2)
    for i in 1:length(a_blocks2)
        dima = binomial(prob.ras_spaces[1], fock_as2[i][1])*binomial(prob.ras_spaces[2], fock_as2[i][2])*binomial(prob.ras_spaces[3], fock_as2[i][3])
        for j in 1:length(b_blocks2)
            dimb = binomial(prob.ras_spaces[1], fock_bs2[j][1])*binomial(prob.ras_spaces[2], fock_bs2[j][2])*binomial(prob.ras_spaces[3], fock_bs2[j][3])
            if a_blocks2[i][1]+b_blocks2[j][1]<= h2
                if a_blocks2[i][2]+b_blocks2[j][2] <= p2
                    block1 = ActiveSpaceSolvers.RASCI_2.RasBlock(fock_as2[i], fock_bs2[j])
                    if (block1, dima, dimb) in rasvec
                        continue
                    else
                        push!(rasvec, (block1, dima, dimb))
                    end
                end
            end
        end
    end

    if h3 != 0
        ##DDCI 2x
        a_blocks2, fock_as2 = ActiveSpaceSolvers.RASCI_2.make_blocks(prob.ras_spaces, prob.na, h3, p3)
        b_blocks2, fock_bs2 = ActiveSpaceSolvers.RASCI_2.make_blocks(prob.ras_spaces, prob.nb, h3, p3)
        for i in 1:length(a_blocks2)
            dima = binomial(prob.ras_spaces[1], fock_as2[i][1])*binomial(prob.ras_spaces[2], fock_as2[i][2])*binomial(prob.ras_spaces[3], fock_as2[i][3])
            for j in 1:length(b_blocks2)
                dimb = binomial(prob.ras_spaces[1], fock_bs2[j][1])*binomial(prob.ras_spaces[2], fock_bs2[j][2])*binomial(prob.ras_spaces[3], fock_bs2[j][3])
                if a_blocks2[i][1]+b_blocks2[j][1]<= h3
                    if a_blocks2[i][2]+b_blocks2[j][2] <= p3
                        block1 = ActiveSpaceSolvers.RASCI_2.RasBlock(fock_as2[i], fock_bs2[j])
                        if (block1, dima, dimb) in rasvec
                            continue
                        else
                            push!(rasvec, (block1, dima, dimb))
                        end
                    end
                end
            end
        end
    end
    return rasvec
end


function fill_lu_ddci(rasvec, ras_spaces::SVector{3, Int})
    ras1, ras2, ras3 = ActiveSpaceSolvers.RASCI_2.make_ras_spaces(ras_spaces)
    norbs = sum(ras_spaces)
    lookup = ActiveSpaceSolvers.RASCI_2.initalize_lu(rasvec, norbs)
   
    #general lu table
    for (fock1, lu_data) in lookup
        idx = 0
        det3 = ActiveSpaceSolvers.RASCI_2.SubspaceDeterminantString(ras_spaces[3], fock1[3])
        for n in 1:det3.max
            det2 = ActiveSpaceSolvers.RASCI_2.SubspaceDeterminantString(ras_spaces[2], fock1[2])
            for j in 1:det2.max
                det1 = ActiveSpaceSolvers.RASCI_2.SubspaceDeterminantString(ras_spaces[1], fock1[1])
                for i in 1:det1.max
                    idx += 1
                    config = [det1.config;det2.config.+det1.no;det3.config.+det1.no.+det2.no]

                    for k in config
                        tmp = deepcopy(config)
                        sgn_a, deta = ActiveSpaceSolvers.RASCI_2.apply_annihilation(tmp, k)

                        for l in 1:sum(ras_spaces)
                            tmp2 = deepcopy(deta)
                            if l in tmp2
                                continue
                            end

                            sgn_c, detc = ActiveSpaceSolvers.RASCI_2.apply_creation(tmp2, l)
                            #sgn_c != 0 || continue
                            delta_kl = ActiveSpaceSolvers.RASCI_2.get_fock_delta(k, l, ras_spaces)
                            haskey(lookup, fock1.+delta_kl) || continue
                            
                            d1, d2, d3 = ActiveSpaceSolvers.RASCI_2.breakup_config(detc, ras1, ras2, ras3)

                            det1_c = ActiveSpaceSolvers.RASCI_2.SubspaceDeterminantString(length(ras1), length(d1), d1)
                            det2_c = ActiveSpaceSolvers.RASCI_2.SubspaceDeterminantString(length(ras2), length(d2), d2.-length(ras1))
                            det3_c = ActiveSpaceSolvers.RASCI_2.SubspaceDeterminantString(length(ras3), length(d3), d3.-length(ras1).-length(ras2))
                            idx_new = ActiveSpaceSolvers.RASCI_2.calc_full_ras_index(det1_c, det2_c, det3_c)
                            
                            lookup[fock1][k, l, idx] = sgn_a*sgn_c*idx_new
                        end
                    end
                    ActiveSpaceSolvers.RASCI_2.incr!(det1)
                end
                ActiveSpaceSolvers.RASCI_2.incr!(det2)
            end
            ActiveSpaceSolvers.RASCI_2.incr!(det3)
        end
    end
    return lookup
end#=}}}=#
