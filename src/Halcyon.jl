# Halcyon.jl
# Pigs might just fly.
# Path Integral Density Inference Theory

module Halcyon

include("potentials.jl")
include("types.jl")       # System, WormConfiguration, WormParams
include("exact.jl")
include("worm.jl")        # Worm algorithm (Spada et al. 2022)
include("analysis.jl")

# Core types
export System, WormConfiguration, WormParams
export Sector, Z_SECTOR, G_SECTOR
export QuantumStatistics, Boltzmannons, Bosons, Fermions

# Potentials
export AbstractPotential, ExternalPotential, PairPotential
export HarmonicPotential, DoubleWellPotential
export LennardJonesPotential, CoulombPotential, YukawaPotential
export HardSpherePotential, AzizPotential, NullPairPotential
export cao_berne_ratio

# Worm algorithm moves
export translate!, redraw!, open!, close!, swap!, move_head!, move_tail!, worm_step!

# Estimators and analysis
export energy_estimators, energy_components
export potential_derivative
export get_cycle, extract_extended_path, recenter!
export get_bead, get_endpoint, total_winding, superfluid_fraction
export count_cycles, permutation_sign  # Fermion sign tracking
export radial_distribution, accumulate_density_matrix!
export momentum_distribution, cycle_length_distribution

# Exact solutions (for validation)
export z1, E1_exact, thermal_wavelength, critical_temperature
export β_from_λT_ratio, β_from_T_ratio, λT_over_L
export jacobi_theta3, G1_Z1_ratio, Z_N, E_N_exact, E_thermodynamic_limit

end # module
