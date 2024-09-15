using QCBase
using ActiveSpaceSolvers
using InCoreIntegrals
using LinearAlgebra
using Printf
using Test
using Arpack
using NPZ
using JLD2

@load "RASCI/ras_h6/_ras_solution.jld2"

#v = abs.(v)

@testset "RASCI (H6, 3α, 3β)" begin
    display(ras)
    println(solver)
    solution = ActiveSpaceSolvers.solve(ints, ras, solver)
    display(solution)
    eval = solution.energies
    @test isapprox(eval, ras_sol.energies, atol=10e-13)

    #davidson
    solver2 = ActiveSpaceSolvers.SolverSettings(nroots=4, tol=1e-10, maxiter=200, package="davidson")
    display(solver2)
    sol2 = ActiveSpaceSolvers.solve(ints, ras, solver2)
    display(sol2)
    @test isapprox(sol2.energies, ras_sol.energies, atol=10e-13)
end

@testset "RASCI expval of S^2" begin
    display(ras)

    s2_new = ActiveSpaceSolvers.RASCI.compute_S2_expval(ras_sol.vectors, ras)
    for i in 1:4
        @printf(" %4i S^2 = %12.8f\n", i, s2_new[i])
    end
    @test isapprox(s2_new, s2, atol=10e-14)
end

@load "RASCI/ras_h12/_integrals.jld2"
ecore = ints.h0
@load "RASCI/ras_h12/FermiCG_test_data.jld2"

@testset "Testing RASCI H12 agaisnt TPSCI Manual Test of RAS(S)" begin
    ras = RASCIAnsatz_2(12,6,6,(4,4,4),max_h=1, max_p=1)
    solver = SolverSettings(nroots=10, tol=1e-8, maxiter=300, verbose=1, package="davidson")
    sol = solve(ints, ras, solver)
    e_tpsci = e0.+ecore
    @test isapprox(sol.energies, e_tpsci, atol=10e-10)
end



