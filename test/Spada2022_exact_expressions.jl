# Exadt expressions from Spada et al. 2022 — Sec. III "Non-interacting Bose gas" & elsewhere 
# Eqs. (ideal1)–(ideal2): Z_N recursion; internal energy E = −∂ln Z_N/∂β.
# Fig. 2: G₁/Z₁ ratio (Eq. 35, open/close detailed balance).

using Test
using Halcyon

@testset "Spada2022 exact expressions self-consistency" begin
    L, λ = 1.0, 0.5
    @testset "A1 λ_T/L round-trip (same control parameter as Fig. 1 / Fig. 3)" begin
        for r in (0.795, 1.257, 1.646)
            β = β_from_λT_ratio(r, L, λ)
            @test λT_over_L(β, L, λ) ≈ r rtol = 1e-12
        end
    end
    @testset "A2 E1_exact decreases with β (cools toward ground)" begin
        β1, β2 = 1.0, 2.0
        e1, e2 = E1_exact(β1, L, λ), E1_exact(β2, L, λ)
        @test isfinite(e1) && isfinite(e2)
        @test e1 > e2
    end
    @testset "A3 E_N_exact(1) ≈ E1_exact" begin
        β = β_from_λT_ratio(0.974, L, λ)
        @test E_N_exact(1, β, L, λ) ≈ E1_exact(β, L, λ) rtol = 1e-5
    end
    @testset "A4 E_N_exact(2) finite" begin
        for β in (1.2, 2.5)
            e = E_N_exact(2, β, L, λ)
            @test isfinite(e)
        end
    end
    @testset "A5 G1_Z1_ratio Fig. 2 / Eq. 35" begin
        for λT_L in (0.795, 1.257)
            g = G1_Z1_ratio(λT_L)
            @test g > 0 && isfinite(g)
        end
    end
end
