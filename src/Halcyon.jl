# Halcyon.jl
# Pigs might just fly.
# Path Integral Density Inference Theory

module Halcyon

include("potentials.jl")
include("types.jl")       # System, WormConfiguration, WormParams
include("exact.jl")
include("worm.jl")        # Worm algorithm (Spada et al. 2022)
include("PermutationFamily.jl") # following DeBois 2014, or at leat what I think they did
include("LSTMPermutationModel.jl") # autoregressive LSTM for permutation sector probabilities
include("analysis.jl")
include("jacknife.jl")

# Core types
export System, WormConfiguration, WormParams
export Sector, Z_SECTOR, G_SECTOR
export QuantumStatistics, Boltzmannons, Bosons, Fermions

# Potentials
export AbstractPotential, ExternalPotential, PairPotential
export HarmonicPotential, DoubleWellPotential, HarmonicPairPotential
export LennardJonesPotential, CoulombPotential, KelbgCoulombPotential, YakubRonchiPotential, YakubRonchiKelbgPotential, YukawaPotential
export yakub_ronchi_r_cut, yakub_ronchi_phi
export yakub_ronchi_background_constant
export AzizPotential, NullPairPotential

# Worm algorithm moves
export translate!, redraw!, open!, close!, swap!, move_head!, move_tail!, worm_step!

# Estimators and analysis
export energy_estimators, energy_components
export potential_derivative, virial_contribution
export is_null, centroid_virial_term
export get_cycle, extract_extended_path, recenter!
export get_bead, get_endpoint, total_winding, superfluid_fraction
export count_cycles, permutation_sign  # Fermion sign tracking
export radial_distribution, accumulate_density_matrix!
export momentum_distribution, cycle_length_distribution

# Dornheim 2019 Permutations reproduction code, very messy currently
export particle_cycle_lengths, permutation_pl_weights, permutation_pcf_weights
export permutation_pcf_uncorrelated, permutation_pl_sum_rule
export accumulate_trap_radial_2d!, finalize_trap_radial_2d, blocking_mean_stderr
# Exact solutions (for validation)
export z1, E1_exact, thermal_wavelength, critical_temperature
export β_from_λT_ratio, β_from_T_ratio, λT_over_L
export jacobi_theta3, G1_Z1_ratio, Z_N, E_N_exact, E_thermodynamic_limit
export Z_N_Fermi, E_N_exact_Fermi
export ideal_fermion_permutation_P_l

# Permutation families (conjugacy classes / C-vector representation)
export integer_partition_count_table, permutation_family_count
export permutation_family_C, C_to_rank, C_from_rank
export DensePermutationFamilyStats, observe_permutation_family!, observe_permutation_family_reservoir!
# Accessors on DensePermutationFamilyStats
export log_multiplicities, cycle_count_matrix, empirical_probabilities
# Permutation sector models
export AbstractPermutationModel, MultiplicityModel, DuBoisModel, MaxEntModel, MAPHybridModel
export LSTMPermutationModel
export fit, probabilities, kl_divergence
# Importance sampling
export PermutationBias, make_permutation_bias

# Jacknife
export JackknifePermutationStats, jackknife_statistics, make_variance_optimised_bias

end # module
