using Halcyon
using Statistics
using LinearAlgebra
using Gnuplot
using StatsBase

# Simulation parameters
const β = 1.0        # Inverse temperature
const nbeads = 30    # Number of beads (M in Thijsesen - is this standard?)
const nparticles = 100  # Number of particles
const ω = 1.0         # Harmonic trap frequency
const ℏ = 1.0         # Planck constant
const m = 1.0         # Particle mass
const EQUILIBRATION_STEPS = 2_000    # ... Thijssen, First 2000 steps discarded
const MEASUREMENT_STEPS = 100        # Steps between measurements
const MEASUREMENTS = 280 #             # ... Thijssen, 28000 post-equilibration steps

# Analytical solutions for quantum harmonic oscillator
qho_ground_state_energy(ω::Float64) = 0.5 * ℏ * ω
qho_ground_state_density(x::Float64, ω::Float64) = sqrt(ω / (π * ℏ)) * exp(-ω * x^2 / ℏ)
qho_finite_temp_energy(ω::Float64, β::Float64) = (ℏ * ω / 2) * coth(β * ℏ * ω / 2)
qho_finite_temp_density(x::Float64, ω::Float64, β::Float64) = sqrt(m * ω / (2π * ℏ * sinh(β * ℏ * ω))) * exp(-(m * ω * x^2 / (2ℏ)) * tanh(β * ℏ * ω / 2))

# Initialize system with correct parameters
println("Initializing quantum particle in harmonic trap...")
system = Halcyon.System(
    nbeads, 
    nparticles, 
    mass=m,
    β=β,
    potential=Halcyon.Harmonic,
    λ=ℏ^2/(2m),
    τ=β/nbeads
)
path = Halcyon.Path(system)

# Equilibration
println("Equilibrating for $EQUILIBRATION_STEPS steps...")
@time Halcyon.localMove!(system, path, moves=EQUILIBRATION_STEPS, verbose=false)

# Data collection
println("Collecting measurements for $MEASUREMENT_STEPS steps...")
positions = zeros(MEASUREMENTS)
energies = zeros(MEASUREMENTS)

@time for i in 1:MEASUREMENTS
    Halcyon.localMove!(system, path, moves=MEASUREMENT_STEPS)
    positions[i] = mean(path.r)  # Center of mass
    energies[i] = Halcyon.total_energy(system, path)
end

# Analysis
mean_energy = mean(energies)
std_energy = std(energies)
sampled_qho_ground_state_energy = qho_ground_state_energy(ω)

println("\nResults:")
println("Numerical Energy (mean_energy ± std_energy): $mean_energy ± $std_energy")
println("qho ground state energy: $sampled_qho_ground_state_energy")
println("Relative Error: $(abs(mean_energy - sampled_qho_ground_state_energy)/sampled_qho_ground_state_energy * 100)%")

# Plotting
@gp "set terminal pngcairo enhanced size 800,400"
@gp "set output 'harmonic_trap_results.png'"
@gp "set multiplot layout 1,2"

# Energy convergence plot
@gp "set title 'Energy Convergence'"
@gp "set xlabel 'MC Steps'"
@gp "set ylabel 'Energy'"
steps = collect(1:MEASUREMENTS)  # Create proper x-axis array
@gp :- steps energies "w l title 'PIMC'" sampled_qho_ground_state_energy*ones(MEASUREMENTS) "w l title 'Exact'"

# PIMC density from sampled positions
edges = range(-3, 3, length=51)  # This gives 50 bins
histogram = fit(Histogram, positions, edges)
bin_width = step(edges)
# Normalize: divide by (total counts × bin width) to get proper density
density = normalize(histogram.weights, 1) ./ bin_width

# Plot!
x_points = collect(range(-3, 3, length=100))
sampled_qho_ground_state_density = qho_ground_state_density.(x_points, ω)
sampled_qho_finite_temp_density = qho_finite_temp_density.(x_points, ω, β)
centers = (edges[1:end-1] .+ edges[2:end]) ./ 2

@gp "set title 'Position Distribution'"
@gp "set xlabel 'Position'"
@gp "set ylabel 'Probability Density'"
@gp :- centers density "w histeps title 'PIMC'"
@gp :- x_points sampled_qho_ground_state_density "w l title 'Ground state'"
@gp :- x_points sampled_qho_finite_temp_density "w l title 'Finite temp'"

potential(x) = 0.5 * m * ω^2 * x^2
potential_values = potential.(x_points)

@gp :- x_points potential_values "w l title 'potential'"

println("\nPlots saved as 'harmonic_trap_results.png'") 
