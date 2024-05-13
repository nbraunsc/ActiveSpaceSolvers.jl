using ActiveSpaceSolvers
using Arpack
using LinearAlgebra
using KrylovKit 
using LinearMaps
        
LinearAlgebra.ishermitian(lop::LinOpMat) = return lop.sym
"""
This type contains the solver settings information needed to solve the problem. 
    
    - nroots::Int
    - tol::Float64
    - maxiter::Int
    - verbose::Int
    - package::String ["arpack", "davidson"]

"""
struct  SolverSettings 
    nroots::Int
    tol::Float64
    maxiter::Int
    verbose::Int
    package::String
    max_ss_vecs::Int
    lindep_thresh::Float64
end


"""
    SolverSettings(;nroots=1, tol=1e-8, maxiter=100, verbose=0, package="arpack")

Default value constructor
"""
function SolverSettings(;   
                            nroots=1, 
                            tol     =1e-8, 
                            maxiter =2000, 
                            verbose =0, 
                            package ="davidson",
                            #package ="arpack",
                            max_ss_vecs = 8,
                            lindep_thresh = 1e-10
    )
    return SolverSettings(nroots, tol, maxiter, verbose, package, max_ss_vecs, lindep_thresh)
end


"""
    solve(ints::InCoreInts{T}, ansatz::A, S::SolverSettings; v0=nothing) where {T, A<:Ansatz}

Get the energies and eigenstates (stored as a `Solution{A,T}` type), for the Hamiltonian (defined 
by `ints`) with the wavefunction approximated by the ansatz (defined by `ansatz`), and passing the 
solver settings (defined by `S`) to the solver.

# Arguments
- `ints`: Integrals
- `ansatz`: Subtype of the abstract type, `Ansatz`
- `S`: SolverSettings
- `v0`: initial guess (optional).
    If provided with "davidson", it needs to have number of columns equal to number of roots sought.
    If provided with "arpack", it can only be one vector.
"""
function solve(ints::InCoreInts{T}, ansatz::A, S::SolverSettings; v0=nothing, diag_vec=nothing) where {T, A<:Ansatz}

    #e = Vector{T}([])
    #v = Matrix{T}([])
    if dim(ansatz) <= 20 
        H = build_H_matrix(ints, ansatz)
        F = eigen(H)
        return Solution{A,T}(ansatz, F.values, F.vectors)
    end
    e = Vector{T}([])
    v = zeros(T, 1,1) 

    if lowercase(S.package) == "arpack"

        
        Hmap = LinearMap(ints, ansatz)

        if v0 == nothing
            e,v = Arpack.eigs(Hmap, nev = S.nroots, which=:SR, tol=S.tol, maxiter=S.maxiter)
            #try
            #    e,v = Arpack.eigs(Hmap, nev = S.nroots, which=:SR, tol=S.tol, maxiter=S.maxiter)
            #catch err
            #    println("Warning: ", err)
            #end

        else
            e,v = Arpack.eigs(Hmap, v0=v0[:,1], nev = S.nroots, which=:SR, tol=S.tol, maxiter=S.maxiter)
        end
        return Solution{A,T}(ansatz, e, v)

    elseif lowercase(S.package) == "davidson"
        #Hmap = LinearMap(ints, ansatz)
        Hmap = LinOpMat(ints, ansatz)
        dav = Davidson(Hmap; 
                       max_iter=S.maxiter, 
                       max_ss_vecs=S.max_ss_vecs, 
                       tol=S.tol, 
                       nroots=S.nroots, 
                       v0=v0, 
                       lindep_thresh=S.lindep_thresh)
        e,v = BlockDavidson.eigs(dav, Adiag=diag_vec)
        return Solution{A,T}(ansatz, e, v)

#    elseif lowercase(S.package) == "krylovkit"
#        
#
#        Hmap = LinOpMat(ints, ansatz)
#        
#        display(ansatz)
#        display(norm(Hmap*rand(3920,1)))
#
#        e, v, info = KrylovKit.eigsolve(Hmap, S.nroots, :SR, 
#                                        #verbosity   = S.verbose, 
#                                        maxiter     = S.maxiter, 
#                                        #krylovdim  = 20, 
#                                        issymmetric = issymmetric(Hmap), 
#                                        ishermitian = true, 
#                                        eager       = true,
#                                        tol         = S.tol)
#        v = hcat(v[1:R]...)
#        if S.verbose > 0
#            @printf(" Number of matvecs performed: %5i\n", info.numops)
#            @printf(" Number of subspace restarts: %5i\n", info.numiter)
#        end
#        return Solution{P,T}(ansatz, e, v)
    end
end

