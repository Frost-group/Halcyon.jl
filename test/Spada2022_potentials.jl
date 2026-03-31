# Potentials used by worm moves (Spada2022 Sec. II). Gradients feed virial estimator (Appendix).

using Test
using Halcyon

@testset "Spada2022 potentials" begin
    @testset "HarmonicPotential" begin
        # V(r) = ½ k |r|²
        V = HarmonicPotential()
        @test V([0.0]) ≈ 0.0
        @test V([1.0]) ≈ 0.5              # ½ × 1 × 1²
        @test V([2.0, 3.0]) ≈ 6.5         # ½ × 1 × (4+9)
        @test HarmonicPotential(k=2.0)([-1.0, 2.0]) ≈ 5.0  # ½ × 2 × (1+4)
    end
    @testset "DoubleWellPotential" begin
        V = DoubleWellPotential()
        @test V([0.0]) ≈ 0.0
    end
    @testset "LennardJonesPotential" begin
        U = LennardJonesPotential()
        @test U([1.0]) ≈ 0.0
        @test U([1000.0]) ≈ 0.0 atol=1e-10
        @test LennardJonesPotential(ϵ=2.0, σ=0.5)([0.5]) ≈ 0.0
    end
    @testset "CoulombPotential" begin
        U = CoulombPotential()
        @test U([1.0]) ≈ 1.0
        @test CoulombPotential(g=2.0)([2.0]) ≈ 1.0
    end
    @testset "YakubRonchiPotential" begin
        L = 10.0
        r_cut = yakub_ronchi_r_cut(L)
        @test r_cut ≈ L * cbrt(3 / (4π))
        U = YakubRonchiPotential(L=L, g=1.0)
        r = 0.5
        inv_c = 1 / r_cut^3
        ref = 1 / r + 0.5 * r^2 * inv_c - 1.5 / r_cut
        @test U(r) ≈ ref
    end
    @testset "YukawaPotential" begin
        U = YukawaPotential(g=1.0, m=1.0)
        @test U([1.0]) ≈ -exp(-1) / 1
        @test YukawaPotential(g=2.0, m=0.5)([2.0]) ≈ -2 * exp(-1) / 2
        r = [0.5, 1.0]; rn = sqrt(1.25)
        @test YukawaPotential(g=1.0, m=2.0)(r) ≈ -exp(-rn * 2) / rn
    end
end

@testset "potential_derivative (for virial estimator)" begin
    @testset "HarmonicPotential Vector" begin
        # V = ½ k |r|², ∇V = k r
        V = HarmonicPotential(k=2.0)
        r = [1.0, 2.0, 3.0]
        @test potential_derivative(V, r) ≈ 2.0 * r
    end
    @testset "Harmonic 3D" begin
        V = HarmonicPotential(k=2.0)
        r = Float64[1, 2, 3]
        g = potential_derivative(V, r)
        @test g ≈ 2.0 * r
        @test length(g) == 3
    end
    @testset "DoubleWellPotential 3D" begin
        V = DoubleWellPotential(A=10.0, B=1.0)
        r = Float64[0.5, 0.5, 0.5]
        g = potential_derivative(V, r)
        @test length(g) == 3
        @test all(isfinite, g)
    end
    @testset "AzizPotential colinear" begin
        U = AzizPotential()
        r = Float64[3.5, 0.0, 0.0]
        g = potential_derivative(U, r)
        @test length(g) == 3
        @test all(isfinite, g)
        @test abs(g[2]) < 1e-10
        @test abs(g[3]) < 1e-10
        @test isfinite(potential_derivative(U, 5.0))
    end
    @testset "LennardJones ForwardDiff" begin
        U = LennardJonesPotential(ϵ=1.0, σ=1.0)
        r = Float64[1.5, 0.0, 0.0]
        g = potential_derivative(U, r)
        @test g isa AbstractVector
        @test all(isfinite, g)
    end
end

@testset "centroid_virial_term (pair potentials)" begin
    using LinearAlgebra: norm, dot
    @testset "Coulomb: hand-computed" begin
        U = CoulombPotential(g=2.0)
        rij = [3.0, 4.0, 0.0]          # |r| = 5
        Δc  = [1.0, 0.0, 0.0]
        # ∇U = −g/r³ · r = −2/125 · [3,4,0]
        # (rij − Δc) · ∇U
        expected = dot(rij .- Δc, (-U.g / norm(rij)^3) .* rij)
        @test centroid_virial_term(U, rij, Δc) ≈ expected
    end
    @testset "Coulomb: reduces to primitive when Δc=0" begin
        U = CoulombPotential(g=1.0)
        rij = [1.0, 2.0, 2.0]
        Δc  = [0.0, 0.0, 0.0]
        @test centroid_virial_term(U, rij, Δc) ≈ virial_contribution(U, rij)
    end
end

@testset "energy_estimators (returning sensible numbers, not proof fo correctness)" begin
    sys = System(16, 1, D=3, β=1.0, V=HarmonicPotential(k=1.0))
    cfg = WormConfiguration(sys)
    Et, Ev = energy_estimators(cfg, sys)
    @test Et isa Float64 && Ev isa Float64
    @test all(isfinite, (Et, Ev))
end
