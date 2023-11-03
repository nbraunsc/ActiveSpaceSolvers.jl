using ActiveSpaceSolvers
using InCoreIntegrals
using Test
using JLD2
using QCBase
#using RDM

#@load "RASCI/ras_h6/_integrals.jld2"
@load "RASCI/ras_h6/_ras_solution.jld2"

function test_rdms(problem, ints::InCoreInts, solver)
    ras = RASCIAnsatz(6, 3, 3, (2, 2, 2), max_h=1, max_p=1)
    solution = ActiveSpaceSolvers.solve(ints, problem, solver)
    e = solution.energies
    v = solution.vectors
    #e, v = solve(ints, problem, rand(problem.dim), 30, 1, 1e-6, true, false)
    e = e[1]
    #e = e[1]+ints.h0
    v = v[:,1]
    #a, b, aa, bb, ab = ActiveSpaceSolvers.RASCI.compute_1rdm_2rdm_new(problem, solution.vectors[:,1])
    a, b, aa, bb, ab = ActiveSpaceSolvers.compute_1rdm_2rdm(solution, root=1)
    #spin summ the rdms
    rdm1 = a + b
    rdm2 = aa + bb + 2*ab
    e_testing = InCoreIntegrals.compute_energy(ints, rdm1, rdm2)
    return e, e_testing
end

@testset "RASCI Contract RDMS with ints to get E" begin
    ras = RASCIAnsatz(6, 3, 3, (2, 2, 2), max_h=1, max_p=1)
    solver = SolverSettings(nroots=1, tol=1e-6, maxiter=12)
    e, e_testing = test_rdms(ras, ints, solver)
    @test isapprox(e_testing, e, atol=10e-13)
end




    


