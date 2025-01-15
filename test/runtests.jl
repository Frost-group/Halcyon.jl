println("Starting tests...")
println("Loading Halcyon...")
using Halcyon

using BenchmarkTools
using Test

println("Loading potentials.jl...")
include("potentials.jl")

@testset "Integration tests - MC" begin
    system=Halcyon.System(5,5)
    path=Halcyon.Path(system)
    
    Halcyon.localMove!(system,path,verbose=true)
    @time Halcyon.localMove!(system,path)
    println("Now 1_000_000 moves...")
    @time Halcyon.localMove!(system,path,moves=1_000_000)
    Halcyon.localMove!(system,path,verbose=true)
    println("Rattle rattle... should be lower in energy.")
end

