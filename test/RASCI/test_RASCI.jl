using FermiCG
using ActiveSpaceSolvers
using NPZ
using InCoreIntegrals
using RDM
using JLD2 

#h0 = npzread("../ActiveSpaceSolvers.jl/test/RASCI/ras_h12/h0.npy")
#h1 = npzread("../ActiveSpaceSolvers.jl/test/RASCI/ras_h12/h1.npy")
#h2 = npzread("../ActiveSpaceSolvers.jl/test/RASCI/ras_h12/h2.npy")
#ints = InCoreInts(h0, h1, h2)
#@load "../ActiveSpaceSolvers.jl/test/RASCI/ras_h6/_integrals.jld2"
#@load "../ActiveSpaceSolvers.jl/test/RASCI/ras_h12/_integrals.jld2"
@load "/Users/nicole/My Drive/code/ActiveSpaceSolvers.jl/test/RASCI/ras_h12/_integrals.jld2"
ecore = ints.h0

clusters_in    = [(1:4),(5:8),(9:12)]
n_clusters = 3

na = 6
nb = 6

d1 = RDM1(12)

clusters = [MOCluster(i,collect(clusters_in[i])) for i = 1:length(clusters_in)]
init_fspace = [ (4,4), (2,2), (0,0)]
display(clusters)

ref_fock = FockConfig(init_fspace)

M=200

nroots = 10
ci_vector = FermiCG.TPSCIstate(clusters, FermiCG.FockConfig(init_fspace), R=nroots);

# Build Cluster basis
cluster_bases = FermiCG.compute_cluster_eigenbasis_spin(ints, clusters, d1, [20,20,20], ref_fock, max_roots=M, verbose=1);
#
# Build ClusteredOperator
clustered_ham = FermiCG.extract_ClusteredTerms(ints, clusters);

#adding additional Fock Space Configs
#h
FermiCG.add_fockconfig!(ci_vector, [(3,4),(3,2),(0,0)])
FermiCG.add_fockconfig!(ci_vector, [(4,3),(2,3),(0,0)])
#p
FermiCG.add_fockconfig!(ci_vector, [(4,4),(1,2),(1,0)])
FermiCG.add_fockconfig!(ci_vector, [(4,4),(2,1),(0,1)])
#hp
FermiCG.add_fockconfig!(ci_vector, [(3,4),(2,2),(1,0)])
FermiCG.add_fockconfig!(ci_vector, [(4,3),(2,2),(0,1)])
FermiCG.add_fockconfig!(ci_vector, [(3,4),(3,1),(0,1)])
FermiCG.add_fockconfig!(ci_vector, [(4,3),(1,3),(1,0)])
#os-hh
#FermiCG.add_fockconfig!(ci_vector, [(3,3),(3,3),(0,0)])
#os-pp
#FermiCG.add_fockconfig!(ci_vector, [(4,4),(1,1),(1,1)])
#os-hpp
#FermiCG.add_fockconfig!(ci_vector, [(3,4),(2,1),(1,1)])
#FermiCG.add_fockconfig!(ci_vector, [(4,3),(1,2),(1,1)])
#os-hhp
#FermiCG.add_fockconfig!(ci_vector, [(3,3),(2,3),(1,0)])
#FermiCG.add_fockconfig!(ci_vector, [(3,3),(3,2),(0,1)])
#os-hhpp
#FermiCG.add_fockconfig!(ci_vector, [(3,3),(2,2),(1,1)])

FermiCG.expand_each_fock_space!(ci_vector, cluster_bases)

FermiCG.eye!(ci_vector)
#
# Build Cluster Operators
cluster_ops = FermiCG.compute_cluster_ops(cluster_bases, ints);
#
# Add cmf hamiltonians for doing MP-style PT2 
FermiCG.add_cmf_operators!(cluster_ops, cluster_bases, ints, d1.a, d1.b, verbose=0);

e0, v = FermiCG.tps_ci_direct(ci_vector, cluster_ops, clustered_ham)
#@time e2 = FermiCG.compute_pt2_energy(v, cluster_ops, clustered_ham, thresh_foi=1e-8);
clustered_S2 = FermiCG.extract_S2(ci_vector.clusters)
@time s2 = FermiCG.compute_expectation_value_parallel(v, cluster_ops, clustered_S2)

@save "/Users/nicole/My Drive/code/ActiveSpaceSolvers.jl/test/RASCI/ras_h12/FermiCG_test_data.jld2" clusters e0 v s2 ecore cluster_bases

#@save "../ActiveSpaceSolvers.jl/test/RASCI/ras_h12/FermiCG_test_data_no_os.jld2" clusters cluster_bases e0 v s2 ecore








