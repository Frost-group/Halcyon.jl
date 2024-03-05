using Test

@testset "Potential functions" begin
    @testset "Harmonic" begin
        @test Harmonic([0.0]) ≈ 0.0
        @test Harmonic([1.0]) ≈ 1.0
        @test Harmonic([2.0, 3.0]) ≈ 13.0
        @test Harmonic([-1.0, 2.0], k=2.0) ≈ 10.0
    end

    @testset "DoubleWell" begin
        @test DoubleWell([0.0]) ≈ 0.0
    end

    @testset "LennardJones" begin
        @test LennardJones([1.0]) ≈ 0.0
        @test LennardJones([1000.0]) ≈ 0.0 atol=1e-10
        @test LennardJones([0.5], ϵ=2.0, σ=0.5) ≈ 0.0
    end

    @testset "Coulomb" begin
        @test Coulomb([1.0]) ≈ -1.0
        @test Coulomb([2.0], g=2.0) ≈ -1.0
    end

    @testset "Yukawa" begin
        @test Yukawa([1.0], g=1.0, m=1.0) ≈ -exp(-1) / 1
        @test Yukawa([2.0], g=2.0, m=0.5) ≈ -2 * exp(-1) / 2
        @test Yukawa([0.5, 1.0], g=1.0, m=2.0) ≈ -exp(-sqrt(1.25) * 2) / sqrt(1.25)
    end
end

