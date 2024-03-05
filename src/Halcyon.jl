# Halcyon.jl
# Pigs might just fly.
# Path Integral Density Inference Theory

module Halcyon

# Namespace is currently very dirty and everything will need to get (re)sorted
# as it becomes clear where things should lie.
include("types.jl") 
#include("system.jl")
include("potentials.jl") 
include("PIMC.jl") # Path integrals for the win

export Harmonic, DoubleWell, LennardJones, Coulomb, Yukawa

end # module

