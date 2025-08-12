using Halcyon
using BenchmarkTools
using Test

include("potentials.jl")
include("QHO.jl")

@testset "Integration tests - MC" begin
    sys = System(5, 5; V=HarmonicPotential(), U=NullPairPotential())
    path = Path(sys)

    # Warmup single move
    localMove!(sys, path, verbose=true)
    # Short run
    localMove!(sys, path)
    # Long run for performance
    @time localMove!(sys, path, moves=1_000_000)
    # A final diagnostic line
    localMove!(sys, path, verbose=true)
end

