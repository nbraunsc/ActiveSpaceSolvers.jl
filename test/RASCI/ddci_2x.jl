using ActiveSpaceSolvers
using LinearAlgebra
using OrderedCollections
using JLD2      
using Test

@load "RASCI/ras_h6/_integrals.jld2"

### DDCI manual test through RAS framework for 2x
#

@testset "DDCI(2x) using RAS framework" begin
    ras = RASCIAnsatz_2(6, 3, 3, (2, 2, 2), max_h=2, max_p=2)
    guess = Matrix(1.0I, ras.dim, ras.dim);
    ras_help = ActiveSpaceSolvers.RASCI_2.fill_lu_helper(ras, ras.max_h, ras.max_p)
    lookup = ActiveSpaceSolvers.RASCI_2.fill_lu(ras_help, ras.ras_spaces)
    rasvec = ActiveSpaceSolvers.RASCI_2.RASVector(guess, ras)

    ddci = DDCIAnsatz(6,3,3,(2,2,2),ex_level=2)

    guess_ddci = Matrix(1.0I, ddci.dim, ddci.dim);
    next_ddci,h,p = ActiveSpaceSolvers.DDCI.find_full_ddci(ddci)
    rasvec2 = ActiveSpaceSolvers.RASCI_2.fill_lu_helper(next_ddci, h, p)
    lu2 = ActiveSpaceSolvers.RASCI_2.fill_lu(rasvec2, ddci.ras_spaces)

    ras_help = ActiveSpaceSolvers.DDCI.fill_lu_helper_ddci(ddci)
    lookup_ddci = ActiveSpaceSolvers.RASCI_2.fill_lu(ras_help, ddci.ras_spaces)
    rasvec_ddci = ActiveSpaceSolvers.RASCI_2.RASVector(guess_ddci, ddci)


    ddci_keys = keys(rasvec_ddci.data)
    tmp = []
    for i in ddci_keys
        push!(tmp, i)
    end


    rasvec_tmp = OrderedDict{ActiveSpaceSolvers.RASCI_2.RasBlock, Array{Float64, 3}}()
    for i in tmp
        rasvec_tmp[i] = rasvec_ddci.data[i]
    end

    #now add in other RasBlocks that are zero but are in RAS model (and not in DDCI)
    ras_keys = keys(rasvec.data)
    tmp_ras = []
    for i in ras_keys
        push!(tmp_ras, i)
    end


    for i in tmp_ras
        if i in tmp
            continue
        else
            rasvec_tmp[i] = zeros(size(rasvec.data[i]))
        end
    end

    for i in tmp
        full_arr = zeros(size(rasvec_tmp[i],1), size(rasvec_tmp[i],2), ras.dim)
        full_arr[:,:,1:ddci.dim] .= rasvec_tmp[i]
        rasvec_tmp[i] = full_arr
    end

    rasvec_rasddci = ActiveSpaceSolvers.RASCI_2.RASVector(rasvec_tmp)

    sig2_rd = ActiveSpaceSolvers.RASCI_2.sigma_two(rasvec_rasddci, ints, ras.ras_spaces, lookup);

    #have to manually zero out blocks i think
    sig2_rd[ddci.dim+1:ras.dim,:].=0;
    sig2_rd[:,ddci.dim+1:ras.dim].=0;

    sig2_rd_tmp = sig2_rd[1:ddci.dim, 1:ddci.dim];

    sig2_ddci = ActiveSpaceSolvers.RASCI_2.sigma_two(rasvec_ddci, ints, ddci.ras_spaces, lu2);


    #test

    #build H for testing
    sig1_rd = ActiveSpaceSolvers.RASCI_2.sigma_one(rasvec_rasddci, ints, ras.ras_spaces, lookup);
    sig2_rd = ActiveSpaceSolvers.RASCI_2.sigma_two(rasvec_rasddci, ints, ras.ras_spaces, lookup);
    sig3_rd = ActiveSpaceSolvers.RASCI_2.sigma_three(rasvec_rasddci, ints, ras.ras_spaces, lookup);

    sig1_rd[ddci.dim+1:ras.dim,:].=0;
    sig1_rd[:,ddci.dim+1:ras.dim].=0;
    sig1_rd_tmp = sig1_rd[1:ddci.dim, 1:ddci.dim];

    sig2_rd[ddci.dim+1:ras.dim,:].=0;
    sig2_rd[:,ddci.dim+1:ras.dim].=0;
    sig2_rd_tmp = sig2_rd[1:ddci.dim, 1:ddci.dim];

    sig3_rd[ddci.dim+1:ras.dim,:].=0;
    sig3_rd[:,ddci.dim+1:ras.dim].=0;
    sig3_rd_tmp = sig3_rd[1:ddci.dim, 1:ddci.dim];

    sig_rasddci = sig1_rd_tmp + sig2_rd_tmp + sig3_rd_tmp

    H_ras_ddci = .5*(sig_rasddci+sig_rasddci')
    H_ras_ddci += 1.0I*ints.h0

    sig1_ddci = ActiveSpaceSolvers.RASCI_2.sigma_one(rasvec_ddci, ints, ddci.ras_spaces, lu2);
    sig2_ddci = ActiveSpaceSolvers.RASCI_2.sigma_two(rasvec_ddci, ints, ddci.ras_spaces, lu2);
    sig3_ddci = ActiveSpaceSolvers.RASCI_2.sigma_three(rasvec_ddci, ints, ddci.ras_spaces, lookup_ddci);

    sig_ddci = sig1_ddci + sig2_ddci + sig3_ddci

    H_ddci = .5*(sig_ddci+sig_ddci')
    H_ddci += 1.0I*ints.h0

    e, v = eigen(H_ddci)
    eras, vras = eigen(H_ras_ddci)
    println("Eigenvalues of DDCI Hamiltonian")
    display(e[1:10])
    println("Eigenvalues of RASCI (running DDCI) Hamiltonian")
    display(eras[1:10])

#    println(H_ddci==H_ras_ddci)
    @test isapprox(e, eras, atol=10e-14)
end





