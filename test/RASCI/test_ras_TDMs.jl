using ActiveSpaceSolvers
using Test
using JLD2

@load "RASCI/ras_h6/tdm_c_a.jld2"
@load "RASCI/ras_h6/tdm_c_b.jld2"

@load "RASCI/ras_h6/tdm_ca_aa.jld2"
@load "RASCI/ras_h6/tdm_ca_bb.jld2"
@load "RASCI/ras_h6/tdm_ca_ab.jld2"

@load "RASCI/ras_h6/tdm_cc_bb.jld2"
@load "RASCI/ras_h6/tdm_cc_aa.jld2"
@load "RASCI/ras_h6/tdm_cc_ab.jld2"

@load "RASCI/ras_h6/tdm_cca_abb.jld2"
@load "RASCI/ras_h6/tdm_cca_aba.jld2"
@load "RASCI/ras_h6/tdm_cca_bbb.jld2"
@load "RASCI/ras_h6/tdm_cca_aaa.jld2"

@testset "Original RASCI Ansatz test TDMs against FCI TDMs" begin
    @testset "c_a" begin
        tdm_c_a_2 = compute_operator_c_a(ras_bra_c_a, ras_ket_c_a);
        @test isapprox(tdm_c_a_2, tdm_c_a_ras, atol=1e-12)
    end
    
    @testset "c_b" begin
        tdm_c_b_2 = compute_operator_c_b(ras_bra_c_b, ras_ket_c_b);
        @test isapprox(tdm_c_b_2, tdm_c_b_ras, atol=1e-12)
    end
    
    @testset "ca_aa" begin
        tdm_ca_aa_2 = compute_operator_ca_aa(ras_bra_ca_aa, ras_ket_ca_aa);
        @test isapprox(tdm_ca_aa_2, tdm_ca_aa_ras, atol=1e-12)
    end
    
    @testset "ca_bb" begin
        tdm_ca_bb_2 = compute_operator_ca_bb(ras_bra_ca_bb, ras_ket_ca_bb);
        @test isapprox(tdm_ca_bb_2, tdm_ca_bb_ras, atol=1e-12)
    end
    
    @testset "ca_ab" begin
        tdm_ca_ab_2 = compute_operator_ca_ab(ras_bra_ca_ab, ras_ket_ca_ab);
        @test isapprox(tdm_ca_ab_2, tdm_ca_ab_ras, atol=1e-12)
    end
    
    @testset "cc_bb" begin
        tdm_cc_bb_2 = compute_operator_cc_bb(ras_bra_cc_bb, ras_ket_cc_bb);
        @test isapprox(tdm_cc_bb_2, tdm_cc_bb_ras, atol=1e-12)
    end
    
    @testset "cc_aa" begin
        tdm_cc_aa_2 = compute_operator_cc_aa(ras_bra_cc_aa, ras_ket_cc_aa);
        @test isapprox(tdm_cc_aa_2, tdm_cc_aa_ras, atol=1e-12)
    end
    
    @testset "cc_ab" begin
        tdm_cc_ab_2 = compute_operator_cc_ab(ras_bra_cc_ab, ras_ket_cc_ab);
        @test isapprox(tdm_cc_ab_2, tdm_cc_ab_ras, atol=1e-12)
    end

    @testset "cca_abb" begin
        tdm_cca_abb_2 = compute_operator_cca_abb(ras_bra_cca_abb, ras_ket_cca_abb);
        @test isapprox(tdm_cca_abb_2, tdm_cca_abb_ras, atol=1e-12)
    end

    @testset "cca_aba" begin
        tdm_cca_aba_2 = compute_operator_cca_aba(ras_bra_cca_aba, ras_ket_cca_aba);
        @test isapprox(tdm_cca_aba_2, tdm_cca_aba_ras, atol=1e-12)
    end

    @testset "cca_bbb" begin
        tdm_cca_bbb_2 = compute_operator_cca_bbb(ras_bra_cca_bbb, ras_ket_cca_bbb);
        @test isapprox(tdm_cca_bbb_2, tdm_cca_bbb_ras, atol=1e-12)
    end

    @testset "cca_aaa" begin
        tdm_cca_aaa_2 = compute_operator_cca_aaa(ras_bra_cca_aaa, ras_ket_cca_aaa);
        @test isapprox(tdm_cca_aaa_2, tdm_cca_aaa_ras, atol=1e-12)
    end

end

@testset "RASCI_Ansatz_2 TDMs against FCI TDMs" begin
    @testset "c_a" begin
        tdm_c_a_2 = compute_operator_c_a(ras2_bra_c_a, ras2_ket_c_a);
        @test isapprox(tdm_c_a_2, tdm_c_a_ras2, atol=1e-12)
    end
    
    @testset "c_b" begin
        tdm_c_b_2 = compute_operator_c_b(ras2_bra_c_b, ras2_ket_c_b);
        @test isapprox(tdm_c_b_2, tdm_c_b_ras2, atol=1e-12)
    end
    
    @testset "ca_aa" begin
        tdm_ca_aa_2 = compute_operator_ca_aa(ras2_bra_ca_aa, ras2_ket_ca_aa);
        @test isapprox(tdm_ca_aa_2, tdm_ca_aa_ras2, atol=1e-12)
    end
    
    @testset "ca_bb" begin
        tdm_ca_bb_2 = compute_operator_ca_bb(ras2_bra_ca_bb, ras2_ket_ca_bb);
        @test isapprox(tdm_ca_bb_2, tdm_ca_bb_ras2, atol=1e-12)
    end
    
    @testset "ca_ab" begin
        tdm_ca_ab_2 = compute_operator_ca_ab(ras2_bra_ca_ab, ras2_ket_ca_ab);
        @test isapprox(tdm_ca_ab_2, tdm_ca_ab_ras2, atol=1e-12)
    end
    
    @testset "cc_bb" begin
        tdm_cc_bb_2 = compute_operator_cc_bb(ras2_bra_cc_bb, ras2_ket_cc_bb);
        @test isapprox(tdm_cc_bb_2, tdm_cc_bb_ras2, atol=1e-12)
    end
    
    @testset "cc_aa" begin
        tdm_cc_aa_2 = compute_operator_cc_aa(ras2_bra_cc_aa, ras2_ket_cc_aa);
        @test isapprox(tdm_cc_aa_2, tdm_cc_aa_ras2, atol=1e-12)
    end
    
    @testset "cc_ab" begin
        tdm_cc_ab_2 = compute_operator_cc_ab(ras2_bra_cc_ab, ras2_ket_cc_ab);
        @test isapprox(tdm_cc_ab_2, tdm_cc_ab_ras2, atol=1e-12)
    end

    @testset "cca_abb" begin
        tdm_cca_abb_2 = compute_operator_cca_abb(ras2_bra_cca_abb, ras2_ket_cca_abb);
        @test isapprox(tdm_cca_abb_2, tdm_cca_abb_ras2, atol=1e-12)
    end

    @testset "cca_aba" begin
        tdm_cca_aba_2 = compute_operator_cca_aba(ras2_bra_cca_aba, ras2_ket_cca_aba);
        @test isapprox(tdm_cca_aba_2, tdm_cca_aba_ras2, atol=1e-12)
    end

    @testset "cca_bbb" begin
        tdm_cca_bbb_2 = compute_operator_cca_bbb(ras2_bra_cca_bbb, ras2_ket_cca_bbb);
        @test isapprox(tdm_cca_bbb_2, tdm_cca_bbb_ras2, atol=1e-12)
    end

    @testset "cca_aaa" begin
        tdm_cca_aaa_2 = compute_operator_cca_aaa(ras2_bra_cca_aaa, ras2_ket_cca_aaa);
        @test isapprox(tdm_cca_aaa_2, tdm_cca_aaa_ras2, atol=1e-12)
    end

end


