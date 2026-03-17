# src/analysis.jl - Post-processing analysis utilities for PIMC configurations

"""
    radial_distribution(cfg::WormConfiguration, sys::System; nbins=100, r_max=nothing) -> (r, gr)

Compute the radial distribution function g(r) from a configuration.
Uses minimum image convention for periodic boundary conditions.
Returns bin centers `r` and histogram values `gr`.
"""
function radial_distribution(cfg::WormConfiguration, sys::System; nbins=100, r_max=nothing)
    N, M, D = sys.N, sys.M, sys.D
    L = sys.L
    r_max = isnothing(r_max) ? L / 2 : r_max
    dr = r_max / nbins

    hist = zeros(nbins)

    # Sum over all time slices and particle pairs
    for j in 1:M
        for i in 1:N-1
            for k in i+1:N
                # Minimum image distance
                d2 = 0.0
                for d in 1:D
                    dx = cfg.r[i, j, d] - cfg.r[k, j, d]
                    dx -= L * round(dx / L)
                    d2 += dx * dx
                end
                r = sqrt(d2)

                if r < r_max
                    bin = min(Int(floor(r / dr)) + 1, nbins)
                    hist[bin] += 2  # Count both i-k and k-i
                end
            end
        end
    end

    # Normalize to ideal gas
    r_centers = [(i - 0.5) * dr for i in 1:nbins]
    n = N / L^D
    for i in 1:nbins
        r = r_centers[i]
        if D == 2
            shell_volume = 2 * π * r * dr
        else
            shell_volume = 4 * π * r^2 * dr
        end
        hist[i] /= (N * M * n * shell_volume)
    end

    return r_centers, hist
end

"""
    accumulate_density_matrix!(hist::AbstractArray, cfg::WormConfiguration, sys::System)

In G-sector, accumulate the head-to-tail displacement into a histogram.
This samples the single-particle density matrix rho_1(r, r').
`hist` should be a 3D array of size (nbins, nbins, nbins) representing the box.
"""
function accumulate_density_matrix!(hist::AbstractArray, cfg::WormConfiguration, sys::System)
    if cfg.sector != G_SECTOR
        return
    end

    # Head position (compact)
    r_head = cfg.r_c_head
    # Tail position (bead 0 of i_tail)
    r_tail = cfg.r[cfg.i_tail, 1, :]

    L = sys.L
    D = sys.D
    nbins = size(hist, 1) # Assumes cubic grid

    # wrapped displacement vector r_head - r_tail
    # We want to bin this into the grid [0, L]^D

    for d in 1:D
        dx = r_head[d] - r_tail[d]
        # Minimum image? No, for n(k) we want the full extent?
        # Actually standard PIMC n(k) uses separation.
        # But with PBC, standard is usually defined within cell or unwrapped?
        # Usually we bin in [-L/2, L/2].
        dx -= L * round(dx / L)

        # Map to bin index 1..nbins
        # shift to [0, L]
        x_shifted = dx + L / 2
        bin = floor(Int, x_shifted / L * nbins) + 1
        bin = clamp(bin, 1, nbins)

        # We need a way to index D-dim array
        # This implementation assumes D=3 for simplicity of the loop below, or generalized?
        # Let's support D=3 specifically or use CartesianIndex.
    end

    # 3D specific implementation for efficiency/simplicity
    if D == 3
        d1 = r_head[1] - r_tail[1]
        d1 -= L * round(d1 / L)
        b1 = clamp(floor(Int, (d1 + L / 2) / L * nbins) + 1, 1, nbins)
        d2 = r_head[2] - r_tail[2]
        d2 -= L * round(d2 / L)
        b2 = clamp(floor(Int, (d2 + L / 2) / L * nbins) + 1, 1, nbins)
        d3 = r_head[3] - r_tail[3]
        d3 -= L * round(d3 / L)
        b3 = clamp(floor(Int, (d3 + L / 2) / L * nbins) + 1, 1, nbins)
        hist[b1, b2, b3] += 1.0
    elseif D == 2
        d1 = r_head[1] - r_tail[1]
        d1 -= L * round(d1 / L)
        b1 = clamp(floor(Int, (d1 + L / 2) / L * nbins) + 1, 1, nbins)
        d2 = r_head[2] - r_tail[2]
        d2 -= L * round(d2 / L)
        b2 = clamp(floor(Int, (d2 + L / 2) / L * nbins) + 1, 1, nbins)
        hist[b1, b2] += 1.0
    end
end

using FFTW

"""
    momentum_distribution(rho1_hist, sys::System) -> (k, nk)

Fourier transform the accumulated density matrix histogram to get n(k).
Returns radial k magnitudes and spherically averaged n(k).
"""
function momentum_distribution(rho1_hist::AbstractArray, sys::System)
    # FFT to get n(k) (unnormalized)
    nk_grid = fftshift(fft(rho1_hist))
    nk_grid = abs.(nk_grid) # n(k) is real and positive (averaged)

    # Radial average
    nbins = size(rho1_hist, 1)
    L = sys.L
    dk = 2π / L
    k_max = nbins / 2 * dk

    # Radial bins for output - use full nbins for finer k-resolution
    num_k_bins = nbins  # Previously nbins÷2, now full range
    k_edges = range(0, stop=k_max, length=num_k_bins + 1)
    k_centers = (k_edges[1:end-1] .+ k_edges[2:end]) ./ 2

    nk_radial = zeros(num_k_bins)
    count_radial = zeros(num_k_bins)

    center = nbins ÷ 2 + 1

    D = sys.D
    indices = CartesianIndices(rho1_hist)

    for I in indices
        k_vec = [(I[d] - center) * dk for d in 1:D]
        k_mag = sqrt(sum(abs2, k_vec))

        # Find bin
        bin = floor(Int, k_mag / (k_max / num_k_bins)) + 1
        if bin >= 1 && bin <= num_k_bins
            nk_radial[bin] += nk_grid[I]
            count_radial[bin] += 1
        end
    end

    # Average
    for i in 1:num_k_bins
        if count_radial[i] > 0
            nk_radial[i] /= count_radial[i]
        end
    end

    # Normalize? 
    # Usually we want n(k=0) to reflect N0.
    # The FFT scaling depends on implementation.
    # We should normalize such that Sum n(k) = N (or similar).
    # For now return raw (relative values are key).

    return k_centers, nk_radial
end


"""
    cycle_length_distribution(cfg::WormConfiguration, sys::System) -> (lengths, probs)

Compute the probability distribution P(m) that an atom belongs to a permutation cycle of length m.
Returns unique cycle lengths found (sorted) and their probabilities.
Ref: Ceperley 1995, Figure 12.
"""
function cycle_length_distribution(cfg::WormConfiguration, sys::System)
    N = sys.N
    visited = zeros(Bool, N)
    counts = Dict{Int,Int}() # Length -> Count of cycles

    for i in 1:N
        if !visited[i]
            cycle = get_cycle(cfg, i)
            m = length(cycle)
            counts[m] = get(counts, m, 0) + 1
            for p in cycle
                visited[p] = true
            end
        end
    end

    # Probability P(m) = (m * NumberOfCyclesOfLengthM) / N
    lengths = sort(collect(keys(counts)))
    probs = Float64[]
    for m in lengths
        nm = counts[m]
        push!(probs, (m * nm) / N)
    end

    return lengths, probs
end
"""
    smooth_momentum_distribution(rho1_grid::AbstractArray, sys::System; 
                                k_max=5.0, nk_bins=200, radial_bins=500) -> (k, nk)

Compute a smooth momentum distribution n(k) by first radially averaging 
the density matrix rho1(r) and then performing a numerical radial Fourier transform.
This avoids k-grid discretization from finite box size.
"""
function smooth_momentum_distribution(rho1_grid::AbstractArray, sys::System;
    k_max=5.0, nk_bins=200, radial_bins=500)
    nbins = size(rho1_grid, 1)
    center = nbins ÷ 2 + 1
    dr_bin = sys.L / nbins
    r_max_plot = sys.L / 2

    # 1. Radial averaging
    rho1_radial = zeros(radial_bins)
    count_radial = zeros(radial_bins)
    dr_hist = r_max_plot / radial_bins

    indices = CartesianIndices(rho1_grid)
    for I in indices
        dx = (I[1] - center) * dr_bin
        dy = (I[2] - center) * dr_bin
        dz = (I[3] - center) * dr_bin
        r_mag = sqrt(dx^2 + dy^2 + dz^2)

        bin = floor(Int, r_mag / dr_hist) + 1
        if bin >= 1 && bin <= radial_bins
            rho1_radial[bin] += rho1_grid[I]
            count_radial[bin] += 1
        end
    end

    for i in 1:radial_bins
        if count_radial[i] > 0
            rho1_radial[i] /= count_radial[i]
        end
    end

    # Normalize rho1(r) such that rho1(0) = 1.0 (internal probability scale)
    scale = rho1_radial[1] > 0 ? 1.0 / rho1_radial[1] : 1.0
    rho1_radial .*= scale
    r_gen = [(i - 0.5) * dr_hist for i in 1:radial_bins]

    # 2. Radial Fourier Transform for n(k)
    # n(k) = 4π/k * Integral[ r * rho1(r) * sin(kr) dr ]
    k_smooth = collect(range(0.001, stop=k_max, length=nk_bins))
    nk_smooth = zeros(nk_bins)

    for (ik, k) in enumerate(k_smooth)
        integral = 0.0
        for i in 1:radial_bins
            r = r_gen[i]
            integral += r * rho1_radial[i] * sin(k * r) * dr_hist
        end
        nk_smooth[ik] = (4 * π / k) * integral
    end

    return k_smooth, nk_smooth, r_gen, rho1_radial
end
