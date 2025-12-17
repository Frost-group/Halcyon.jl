using Halcyon
using Statistics
using LinearAlgebra
using Gnuplot
using StatsBase

"""
Example of Path Integral Molecular Dynamics for a quantum particle in a harmonic trap.
Compares numerical results with analytical solutions for the quantum harmonic oscillator.
"""

# Simulation parameters
const β = 10.0        # Inverse temperature
const M = 1     # Number of beads
const N = 1  # Number of particles
const ω = 1.0         # Harmonic trap frequency
const ℏ = 1.0         # Planck constant
const m = 1.0         # Particle m
const dt = 0.001       # MD timestep 
const EQUILIBRATION_STEPS = 5_000    
const MEASUREMENT_STEPS = 10_000     # Production steps
const MEASUREMENT_FREQUENCY = 10     # Steps between measurements

# Analytical solutions for 1D quantum harmonic oscillator
qho_ground_state_energy(ω::Float64) = 0.5 * ℏ * ω
qho_ground_state_density(x::Float64, ω::Float64) = 
    sqrt(ω / (π * ℏ)) * exp(-ω * x^2 / ℏ)

# Initialize system
println("Initializing quantum particle in harmonic trap...")
system = Halcyon.System(
    M, 
    N, 
    m=m,
    β=β,
    U=r -> Halcyon.Harmonic(r, ω=ω),  # Pass ω parameter
    λ=ℏ^2/(2m),
    τ=β/M
)
path = Halcyon.Path(system)

# Initialize velocities and remove COM motion
initialize_path!(system, path)

# Equilibration with thermostat
println("Equilibrating for $EQUILIBRATION_STEPS steps...")
@time begin
    for step in 1:EQUILIBRATION_STEPS
        run_pimd!(system, path, 1, dt=dt, thermostat=true)
        if step % 100 == 0
            current_energy = total_energy(system, path)
            positions_str = join(round.(path.r[:, :, 1], digits=3), ", ")
            println("Step $step, Energy: $current_energy, Positions: [$positions_str]")
        end
    end
end

# Data collection
println("Collecting measurements for $MEASUREMENT_STEPS steps...")
n_measurements = div(MEASUREMENT_STEPS, MEASUREMENT_FREQUENCY)
positions = zeros(n_measurements)
energies = zeros(n_measurements)
centroid_positions = zeros(n_measurements)

global measurement_idx = 1

@time begin
    for step in 1:MEASUREMENT_STEPS
        # Run PIMD with thermostat
        run_pimd!(system, path, 1, dt=dt, thermostat=true)
        
        # Collect data every MEASUREMENT_FREQUENCY steps
        if step % MEASUREMENT_FREQUENCY == 0
            # Store centroid position (average over beads)
            centroid_positions[measurement_idx] = 
                mean(view(path.r, 1, :, 1))  # For 1D, only need first spatial component
            
            # Calculate and store energy
            energies[measurement_idx] = 
                Halcyon.total_energy(system, path)
            
            # Store positions of all beads for density calculation
            positions[measurement_idx] = path.r[1, 1, 1]  # Store first bead position
            
            global measurement_idx += 1
        end
    end
end

# Analysis
mean_energy = mean(energies)
std_energy = std(energies)
exact_energy = qho_ground_state_energy(ω)

println("\nResults:")
println("PIMD Energy: $mean_energy ± $std_energy")
println("Exact Energy: $exact_energy")
println("Relative Error: $(abs(mean_energy - exact_energy)/exact_energy * 100)%")

# Plotting
@gp "set terminal pngcairo enhanced size 800,400"
@gp "set output 'harmonic_trap_md_results.png'"
@gp "set multiplot layout 1,2"

# Energy convergence plot
@gp "set title 'Energy Convergence'"
@gp "set xlabel 'MD Steps'"
@gp "set ylabel 'Energy'"
steps = collect(1:n_measurements) .* MEASUREMENT_FREQUENCY
@gp :- steps energies "w l title 'PIMD'" exact_energy*ones(n_measurements) "w l title 'Exact'"

# Position distribution
edges = range(-3, 3, length=51)
histogram = fit(Histogram, centroid_positions, edges)
bin_width = step(edges)
density = normalize(histogram.weights, 1) ./ bin_width

# Compare with analytical ground state density
x_points = collect(range(-3, 3, length=100))
exact_density = qho_ground_state_density.(x_points, ω)
centers = (edges[1:end-1] .+ edges[2:end]) ./ 2

@gp "set title 'Position Distribution'"
@gp "set xlabel 'Position'"
@gp "set ylabel 'Probability Density'"
@gp :- centers density "w histeps title 'PIMD'"
@gp :- x_points exact_density "w l title 'Ground state QHO'"

println("\nPlots saved as 'harmonic_trap_md_results.png'")

# Additional analysis: Autocorrelation time
function autocorrelation_time(x)
    μ = mean(x)
    σ² = var(x)
    N = length(x)
    max_lag = min(N-1, 100)  # Look at up to 100 lags
    
    ρ = zeros(max_lag)
    for k in 1:max_lag
        ρ[k] = sum((x[1:N-k] .- μ) .* (x[k+1:N] .- μ)) / ((N-k) * σ²)
    end
    
    # Integrated autocorrelation time
    τ_int = 1 + 2 * sum(ρ[ρ .> 0])  # Stop at first negative value
    return τ_int
end

τ_energy = autocorrelation_time(energies)
τ_position = autocorrelation_time(centroid_positions)

println("\nAutocorrelation Analysis:")
println("Energy autocorrelation time: $τ_energy steps")
println("Position autocorrelation time: $τ_position steps")

