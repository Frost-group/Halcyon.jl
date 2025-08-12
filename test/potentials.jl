using Test
using Halcyon

@testset "Potential types" begin
    @testset "HarmonicPotential" begin
        V = HarmonicPotential()
        @test V([0.0]) ≈ 0.0
        @test V([1.0]) ≈ 1.0
        @test V([2.0, 3.0]) ≈ 13.0
        @test HarmonicPotential(k=2.0)([-1.0, 2.0]) ≈ 10.0
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
        @test U([1.0]) ≈ -1.0
        @test CoulombPotential(g=2.0)([2.0]) ≈ -1.0
    end

    @testset "YukawaPotential" begin
        U = YukawaPotential(g=1.0, m=1.0)
        @test U([1.0]) ≈ -exp(-1) / 1
        @test YukawaPotential(g=2.0, m=0.5)([2.0]) ≈ -2 * exp(-1) / 2
        r = [0.5, 1.0]; rn = sqrt(1.25)
        @test YukawaPotential(g=1.0, m=2.0)(r) ≈ -exp(-rn * 2) / rn
    end
end

