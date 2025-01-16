using Halcyon
using Statistics
using LinearAlgebra
using Gnuplot
using StatsBase

# Simulation parameters
const β = 10.0        # Inverse temperature
const nbeads = 100    # Number of beads (M in Thijsesen - is this standard?)
const nparticles = 1  # Number of particles
const ω = 1.0         # Harmonic trap frequency
const ℏ = 1.0         # Planck constant
const m = 1.0         # Particle mass
const EQUILIBRATION_STEPS = 2_000    # First 2000 steps discarded
const MEASUREMENT_STEPS = 100        # Steps between measurements
const MEASUREMENTS = 280             # To get 28000 post-equilibration steps

# Analytical solutions for 1D quantum harmonic oscillator
qho_ground_state_energy(ω::Float64)=0.5 * ℏ * ω
# Ground state probability density
qho_ground_state_density(x::Float64, ω::Float64)=sqrt(ω / (π * ℏ)) * exp(-ω * x^2 / ℏ)

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
@time Halcyon.localMove!(system, path, moves=EQUILIBRATION_STEPS)

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
centers = (edges[1:end-1] .+ edges[2:end]) ./ 2

@gp "set title 'Position Distribution'"
@gp "set xlabel 'Position'"
@gp "set ylabel 'Probability Density'"
@gp :- centers density "w histeps title 'PIMC'"
@gp :- x_points sampled_qho_ground_state_density "w l title 'Ground state QHO'"

println("\nPlots saved as 'harmonic_trap_results.png'") 
