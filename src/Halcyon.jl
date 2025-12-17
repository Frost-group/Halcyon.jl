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

# Worm algorithm (Spada et al. 2022)
include("exact.jl")
include("worm.jl")

export System, Path, localMove!, total_energy, energy_virial
export AbstractPotential, ExternalPotential, PairPotential,
       HarmonicPotential, DoubleWellPotential,
       LennardJonesPotential, CoulombPotential, YukawaPotential,
       HardSpherePotential, NullPairPotential, cao_berne_ratio

# Worm algorithm exports
export Sector, Z_SECTOR, G_SECTOR, WormConfiguration, WormParams
export translate!, redraw!, open!, close!, swap!, move_head!, move_tail!, worm_step!
export energy_thermodynamic, get_cycle, recenter!, get_bead, get_endpoint
export z1, E1_exact, thermal_wavelength, critical_temperature
export β_from_λT_ratio, β_from_T_ratio, λT_over_L
export jacobi_theta3, G1_Z1_ratio, Z_N, E_N_exact, E_thermodynamic_limit

end # module

