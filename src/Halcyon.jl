# Halcyon.jl
# Pigs might just fly.
# Path Integral Density Inference Theory

module Halcyon

# Namespace is currently very dirty and everything will need to get (re)sorted
# as it becomes clear where things should lie.
include("potentials.jl")
include("types.jl")
#include("system.jl")
include("PIMC.jl") # Path integrals for the win

export System, Path, localMove!, total_energy, energy_virial
export AbstractPotential, ExternalPotential, PairPotential,
       HarmonicPotential, DoubleWellPotential,
       LennardJonesPotential, CoulombPotential, YukawaPotential,
       NullPairPotential

end # module

