#!/usr/bin/env julia
# Dornheim et al. (permutation paper) fig:column_trap — 2D harmonic dot + Coulomb, N=10, β ∈ {0.5,1,3}.
# Ref: refs/Dornheim2019Permutation/main.tex §2D quantum dot.

using Halcyon
using Printf
using Statistics
using ProgressMeter


const N_PART = 10
const D = 2
const λ_τ = 0.5          # ℏ²/(2m) in oscillator units
const K_TRAP = 0.5       # V = k r² = ½ r² ⇒ k = 0.5
const λ_COULOMB = 0.5    # pair +λ/r ⇒ CoulombPotential(g=-λ)
const L_BOX = 50.0 # BIG box ~ vacuum
const RADIAL_DR = 0.08
const RADIAL_RMAX = 4.5

function trap_worm_params(M::Int)
    WormParams(C=1.0, j_max=max(20, M ÷ 2), r_max=min(8.0, 1.2 * L_BOX))
end

function run_trap_β(β::Float64, M::Int, equil::Int, steps::Int, measure_every::Int, outdir::String;
                    radial_origin::Tuple{Float64,Float64}=(0.0, 0.0))
    sys = System(M, N_PART; D=D, β=β, λ=λ_τ, L=L_BOX,
        V=HarmonicPotential(k=K_TRAP), U=CoulombPotential(g=-λ_COULOMB), statistics=Fermions)
    params = trap_worm_params(M)
    cfg = WormConfiguration(sys)

    acc_P = zeros(N_PART)
    acc_W1 = zeros(N_PART)
    acc_W3 = zeros(N_PART)
    signs = Int[]
    nb = max(1, Int(ceil(RADIAL_RMAX / RADIAL_DR)))
    hist_b = zeros(Float64, nb)
    hist_sf = zeros(Float64, nb)
    sum_S = 0.0
    n_z = 0
    n_radial = 0

    acc_Wmatrix = zeros(N_PART, N_PART) # full matrix

    @showprogress dt = 1 desc = "Equilibration" for _ in 1:equil
        worm_step!(cfg, sys, params)
    end
    @showprogress dt = 1 desc = "Equilibration" for t in 1:steps
        worm_step!(cfg, sys, params)
        t % measure_every != 0 && continue
        cfg.sector != Z_SECTOR && continue
        w = permutation_pl_weights(cfg)
        acc_P .+= w
        W = permutation_pcf_weights(cfg)
        acc_Wmatrix .+= W
        acc_W1 .+= W[1, :]
        acc_W3 .+= W[3, :]
        S = permutation_sign(cfg, sys)
        push!(signs, S)
        sum_S += S
        accumulate_trap_radial_2d!(hist_b, hist_sf, cfg, sys, S; dr=RADIAL_DR, r_max=RADIAL_RMAX,
            origin=radial_origin)
        n_z += 1
        n_radial += 1
    end

    n_z == 0 && error("No Z-sector samples for β=$β")

    # normalise Permutations weights by number of samples (Z-sector only)
    P = acc_P ./ n_z
    Wmatrix = acc_Wmatrix ./ n_z
    W1 = acc_W1 ./ n_z
    W3 = acc_W3 ./ n_z
    Pu1 = [P[1] * P[l] for l in 1:N_PART]
    Pu3 = [P[3] * P[l] for l in 1:N_PART]
    sum_lp = permutation_pl_sum_rule(P)
    # Blocking analysis of sign following Dornheim
    mS, eS = blocking_mean_stderr(Float64.(signs); n_blocks=16)
#   Now spit out the data for GNUPLOT'ing
    base = joinpath(outdir, @sprintf("trap_beta%.4g", β))
    pl_path = base * "_pl.tsv"
    open(pl_path, "w") do io
        println(io, "l\tP_l\tP_l_times_l\tP_1l\tP_3l\tPu_1l\tPu_3l")
        for l in 1:N_PART
            println(io, "$l\t$(P[l])\t$(P[l]*l)\t$(W1[l])\t$(W3[l])\t$(Pu1[l])\t$(Pu3[l])")
        end
    end

    # P(k,l) heat-map figures, like Fig 5 onwards; saved for GNUPLOT
    # first uncorrelated from multiplying the vectors
    open(base * "permutematrix_uncorrelated.tsv", "w") do io
    for k in 1:N_PART
        for l in 1:N_PART
            p=P[k]*P[l]
            @printf(io, "%8.6f ", p)
        end
        println(io)
    end
    # correlated, directly from accumulated matrix
    open(base * "permutematrix_correlated.tsv", "w") do io
        for k in 1:N_PART
            for l in 1:N_PART
                @printf(io, "%8.6f ", Wmatrix[k, l])
            end
            println(io)
        end
    end

end 


    # Radial density analysis 
    r, n_b, n_f = finalize_trap_radial_2d(hist_b, hist_sf, sum_S, n_radial, N_PART, M, RADIAL_DR)
    n_b*=N_PART # Dornheim plots a particle density, so x N
    n_f*=N_PART
    # And shove in another GNUPLOT script
    dens_path = base * "_density.tsv"
    open(dens_path, "w") do io
        println(io, "r\tn_bose\tn_fermi")
        for i in eachindex(r)
            println(io, "$(r[i])\t$(n_b[i])\t$(n_f[i])")
        end
    end

    # log simulation info, because why not
    meta_path = base * "_meta.txt"
    open(meta_path, "w") do io
        println(io, "beta = ", β)
        println(io, "M = ", M, " N = ", N_PART, " L = ", L_BOX)
        println(io, "equil = ", equil, " steps = ", steps, " measure_every = ", measure_every)
        println(io, "Z_samples = ", n_z)
        println(io, "sum_l l*P(l) = ", sum_lp)
        println(io, "mean_S = ", mS, " stderr_S_blocking = ", eS)
        println(io, "radial_origin = ", radial_origin, " dr = ", RADIAL_DR, " r_max = ", RADIAL_RMAX)
    end

    @printf("β=%g  Z_samples=%d  ⟨S⟩=%.5f±%.5f  ∑l·P(l)=%.5f  wrote %s\n", β, n_z, mS, eS, sum_lp, pl_path)
    return (; P, W1, W3, mS, eS, sum_lp, pl_path, dens_path)
end

function main()
    fast = get(ENV, "HALCYON_TRAP_FAST", "") == "1"
    M = fast ? 32 : 80
    equil = fast ? 2_000 : 500_000
    steps = fast ? 8_000 : 10_000_000
    measure_every = fast ? 40 : 100
    outdir = "./" 
    mkpath(outdir)

    println("Dornheim fig:column_trap — 2D dot N=$N_PART, λ_coulomb=$λ_COULOMB (FAST=$fast)")
    println("Output directory: ", outdir)

    for β in (0.5, 1.0, 3.0)
        run_trap_β(β, M, equil, steps, measure_every, outdir; radial_origin=(0.0, 0.0))
    end
    println("Done! Now gnuplot ./trap.gpt...")
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
