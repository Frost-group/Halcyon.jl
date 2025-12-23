# examples/Ceperley1995-Fig4-gr.jl
# Replicate Ceperley 1995 Figure 4: Radial distribution function g(r)
# Comparing T=4.0K (normal liquid) and T=1.0K (superfluid)

using Halcyon, Printf, Gnuplot, Statistics

# ═══════════════════════════════════════════════════════════════════════════════
# Parameters
# ═══════════════════════════════════════════════════════════════════════════════

const N = 64
const M = 40
const D = 3
const λ = 6.0592 # Å² K
const ρ = 0.0218 # atoms/Å³ (SVP density)
const L_box = (N / ρ)^(1 / 3) # 14.3 Å
const r_max = L_box / 2
const nbins = 100

# MC parameters
const EQUILIBRATION_STEPS = 100_000
const MEASUREMENT_STEPS = 10_000_000
const MEASUREMENT_STRIDE = 1000

# ═══════════════════════════════════════════════════════════════════════════════
# Simulation
# ═══════════════════════════════════════════════════════════════════════════════

function run_gr_simulation(T::Float64)
    β = 1.0 / T
    @printf("\nSimulation: T = %.2f K, β = %.3f K⁻¹\n", T, β)

    sys = System(M, N; m=1.0, D=D, β=β, λ=λ, L=L_box, U=AzizPotential())
    cfg = WormConfiguration(sys)
    params = WormParams(C=1.0, j_max=20, r_max=L_box / 2) # Tuned for 3D

    println("Equilibrating...")
    for step in 1:EQUILIBRATION_STEPS
        worm_step!(cfg, sys, params)
    end

    println("Sampling...")
    gr_accum = zeros(nbins)
    count = 0

    # We need r_centers for plotting, computed once
    r_centers = []

    for step in 1:MEASUREMENT_STEPS
        worm_step!(cfg, sys, params)

        if step % MEASUREMENT_STRIDE == 0 # I don't think we care whether G or Z sector?
            rc, gr = radial_distribution(cfg, sys; nbins=nbins, r_max=r_max)
            gr_accum .+= gr
            count += 1
            if isempty(r_centers)
                r_centers = rc
            end
        end
    end

    return r_centers, gr_accum ./ count
end

# ═══════════════════════════════════════════════════════════════════════════════
# Main Execution
# ═══════════════════════════════════════════════════════════════════════════════

println("Box Length L = $L_box Å")

results = Dict()
temperatures = [4.0, 1.0]

for T in temperatures
    r, gr = run_gr_simulation(T)
    results[T] = (r, gr)
end

# Plotting
@gp "reset"
@gp :- "set title 'Radial Distribution Function g(r) for 4He'"
@gp :- "set xlabel 'r (Å)'"
@gp :- "set ylabel 'g(r)'"
@gp :- "set grid"
@gp :- "set yrange [0:1.5]"
@gp :- "set xrange [0:8]"
@gp :- "set size square"

# Plot T=4.0K
data_4K = results[4.0]
@gp :- data_4K[1] data_4K[2] "w l lw 2 t 'T=4.0K'"

# Plot T=1.0K
data_1K = results[1.0]
@gp :- data_1K[1] data_1K[2] "w l lw 2 dt 2 t 'T=1.0K'"

fname = "Ceperley1995_Fig4_gr.png"
Gnuplot.save(fname, term="pngcairo size 800,600 enhanced font 'Helvetica,12'")
println("\nSaved plot to $fname")
