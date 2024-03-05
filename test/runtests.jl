using Halcyon

using Test

include("potentials.jl")

@testset "Integration tests - MC" begin
    system=Halcyon.System(5,5)
    path=Halcyon.Path(system)
    
    Halcyon.localMove!(system,path,verbose=true)
    @time Halcyon.localMove!(system,path)
    println("Now 1_000_000 moves...")
    @time for i in 1:1_000_000 Halcyon.localMove!(system,path) end
    Halcyon.localMove!(system,path,verbose=true)
    println("Rattle rattle... should be lower in energy.")
end

