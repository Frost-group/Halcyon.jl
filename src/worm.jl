# worm.jl - Worm algorithm for PIMC with periodic boundary conditions
# Ref: Spada et al. 2022, "Path-integral Monte Carlo worm algorithm for Bose systems
#      with periodic boundary conditions", Condens. Matter 2022, 7, 30
# arXiv:2203.00010

# Type definitions (Sector, WormConfiguration, WormParams) are now in types.jl

function WormConfiguration(sys::System)
    N, M, D, L = sys.N, sys.M, sys.D, sys.L
    λ, δτ = sys.λ, sys.τ

    # Initialize positions: beads 0 to M-1
    r = zeros(N, M, D)

    for i in 1:N
        # Bead 0 uniformly in [0, L)^D
        r[i, 1, :] .= L .* rand(D)

        # Sample beads 1 to M-1 using a temporary closed loop Levy bridge
        # from bead 0 (index 1) to its image (also index 1)
        for j in 2:M
            # steps from j-1 to M
            Δj_fwd = M - j + 1
            Δj_back = 1
            denom = Δj_fwd + Δj_back

            # Mean: bridge from r[j-1] to r[1]
            r_mean = (Δj_fwd .* r[i, j-1, :] .+ Δj_back .* r[i, 1, :]) ./ denom
            σ = sqrt(2λ * δτ * Δj_back * Δj_fwd / denom)
            r[i, j, :] .= r_mean .+ σ .* randn(D)
        end
    end

    # Identity permutation: each polymer closes on itself
    next = collect(1:N)
    prev = collect(1:N)
    w = zeros(Int, N, D)

    return WormConfiguration(
        sector=Z_SECTOR,
        next=next,
        prev=prev,
        i_head=1,
        i_tail=1,
        r_c_head=vec(copy(r[1, 1, :])),
        r=r,
        w=w
    )
end

function Base.show(io::IO, cfg::WormConfiguration)
    N, M, D = size(cfg.r)
    print(io, "WormConfiguration($(cfg.sector), N=$N, M=$M, D=$D")
    if cfg.sector == G_SECTOR
        print(io, ", head=$(cfg.i_head), tail=$(cfg.i_tail)")
    end
    print(io, ")")
end


# ═══════════════════════════════════════════════════════════════════════════════
# Helpers for implicit endpoints and consistency
# ═══════════════════════════════════════════════════════════════════════════════

"""
    get_endpoint(cfg, i, d, L) -> Float64

Return the d-th coordinate of the M-th bead (endpoint) of polymer i.
Zero-allocation accessor.
"""
@inline function get_endpoint(cfg::WormConfiguration, i::Int, d::Int, L::Float64)
    if cfg.sector == G_SECTOR && i == cfg.i_head
        return cfg.r_c_head[d] + cfg.w[i, d] * L
    else
        target = cfg.next[i]
        return cfg.r[target, 1, d] + cfg.w[i, d] * L
    end
end

"""
    get_endpoint(cfg, i, L) -> Vector

Return the coordinate of the M-th bead (endpoint) of polymer i.
This is a derived quantity in Option A.
NOTE: This allocates. Use the dimension-aware version for inner loops.
"""
function get_endpoint(cfg::WormConfiguration, i::Int, L::Float64)
    D = size(cfg.r, 3)
    out = zeros(D)
    for d in 1:D
        out[d] = get_endpoint(cfg, i, d, L)
    end
    return out
end

"""
    get_bead(cfg, i, j, L) -> Vector (copy or view)

Return the coordinate of bead j (0 to M) of polymer i.
NOTE: This allocates for j = M. Use the dimension-aware version for inner loops.
"""
function get_bead(cfg::WormConfiguration, i::Int, j::Int, L::Float64)
    M = size(cfg.r, 2)
    if j < M
        return @view cfg.r[i, j+1, :]
    elseif j == M
        return get_endpoint(cfg, i, L)
    else
        throw(BoundsError("Bead index $j out of range [0, $M]"))
    end
end

"""
    get_bead(cfg, i, j, d, L) -> Float64

Return the d-th coordinate of bead j (0 to M) of polymer i.
Zero-allocation accessor.
"""
@inline function get_bead(cfg::WormConfiguration, i::Int, j::Int, d::Int, L::Float64)
    M = size(cfg.r, 2)
    if j < M
        return cfg.r[i, j+1, d]
    elseif j == M
        return get_endpoint(cfg, i, d, L)
    else
        throw(BoundsError("Bead index $j out of range [0, $M]"))
    end
end

@inline function _wrap_compact_and_winding(x::Float64, L::Float64)
    n = floor(Int, x / L)
    return x - n * L, n
end

function update_head_compact!(cfg::WormConfiguration, L::Float64)
    D = size(cfg.r, 3)
    i_H = cfg.i_head
    # head endpoint position
    r_M = get_endpoint(cfg, i_H, L)
    for d in 1:D
        x_c, n = _wrap_compact_and_winding(r_M[d], L)
        cfg.r_c_head[d] = x_c
        cfg.w[i_H, d] = n
    end
    return nothing
end

"""
    get_cycle(cfg::WormConfiguration, i::Int) -> Vector{Int}

Find all particle indices in the permutation cycle containing particle i.
"""
function get_cycle(cfg::WormConfiguration, i::Int)
    N = length(cfg.next)
    cycle = [i]
    j = cfg.next[i]
    count = 0
    while j != i && count < N
        push!(cycle, j)
        j = cfg.next[j]
        count += 1
    end
    return cycle
end

"""
    count_cycles(cfg::WormConfiguration) -> Int

Count permutation cycles by reusing get_cycle().
Each particle belongs to exactly one cycle.
"""
function count_cycles(cfg::WormConfiguration)
    N = length(cfg.next)
    visited = falses(N)
    num_cycles = 0

    for i in 1:N
        if !visited[i]
            num_cycles += 1
            cycle = get_cycle(cfg, i)  # Reuse existing function
            for p in cycle
                visited[p] = true
            end
        end
    end
    return num_cycles
end

"""
    permutation_sign(cfg::WormConfiguration, sys::System) -> Int

Return permutation sign based on quantum statistics:
- Boltzmannons: Always +1 (distinguishable)
- Bosons: Always +1 (symmetric)
- Fermions: (-1)^(N - num_cycles) (antisymmetric)
"""
function permutation_sign(cfg::WormConfiguration, sys::System)
    if sys.statistics == Boltzmannons || sys.statistics == Bosons
        return 1
    elseif sys.statistics == Fermions
        N = length(cfg.next)
        nc = count_cycles(cfg)
        return iseven(N - nc) ? 1 : -1
    else
        error("Unknown quantum statistics: $(sys.statistics)")
    end
end

# Convenience: permutation_sign without System (assumes Fermions)
permutation_sign(cfg::WormConfiguration) = permutation_sign_fermion(cfg)

function permutation_sign_fermion(cfg::WormConfiguration)
    N = length(cfg.next)
    nc = count_cycles(cfg)
    return iseven(N - nc) ? 1 : -1
end

"""
    recenter!(cfg::WormConfiguration, i::Int, L::Float64)

Shift polymer i so that bead 0 is within the fundamental cell [0, L)^D.
Updates winding numbers of polymer i and all polymers that connect to it.
"""
function recenter!(cfg::WormConfiguration, i::Int, L::Float64)
    D = size(cfg.r, 3)
    M = size(cfg.r, 2)

    # Compute shift needed to put bead 0 in [0, L)^D
    for d in 1:D
        Δw_d = floor(Int, cfg.r[i, 1, d] / L)
        if Δw_d != 0
            # 1. Shift beads 0..M-1 of polymer i
            for j in 1:M
                cfg.r[i, j, d] -= Δw_d * L
            end

            # 2. Update windings
            # A translation by ΔR = -Δw_d * L affects two springs:
            # - Spring i connects i (bead M-1) to next[i] (bead 0).
            #   r_M_i = r_0_next_i + w_i * L
            #   If r_0_i shifted by -Δw_d*L, then r_M_i shifted by -Δw_d*L.
            #   So r_0_next_i + w_i' * L = (r_0_i - Δw_d*L + ...) = r_M_i - Δw_d*L
            #   So w_i' = w_i - Δw_d.
            #
            # - Spring k connects prev[i] (bead M-1) to i (bead 0).
            #   r_M_k = r_0_i + w_k * L
            #   If r_0_i shifted by -Δw_d*L, then to keep r_M_k fixed:
            #   r_0_i - Δw_d*L + w_k' * L = r_0_i + w_k * L
            #   So w_k' = w_k + Δw_d.

            # Update i's own winding (connects out from i)
            # In G-sector, if i is the head, its winding relates to the compact r_c_head,
            # which we treat as its "endpoint". The logic remains: r_Head = r_0_i + w_i*L.
            # If r_0_i moves, w_i must move in opposite to keep r_Head fixed.
            cfg.w[i, d] -= Δw_d

            # Update prev[i]'s winding (connects in to i)
            # In G-sector, if i is the tail, then prev[i] is NOT connected to it via a spring.
            # However, the tail's winding w[i_tail] is not used?
            # Actually Spada Eq 19-21: tail is special.
            # Let's check prev[i]. If we are in G-sector, prev[tail] is head? No, they are open.
            # BUT i_tail = next[i_head]. So prev[i_tail] = i_head.
            # If we shift the tail, we shift its bead 0.
            # In G-sector, the head is NOT connected to the tail's bead 0.
            # So if we shift the tail (i), we don't update w[i_head].
            k = cfg.prev[i]
            if !(cfg.sector == G_SECTOR && i == cfg.i_tail)
                cfg.w[k, d] += Δw_d
            end
        end
    end
    return nothing
end

"""
    extract_extended_path(cfg::WormConfiguration, sys::System, i_start::Int) -> Matrix

Extract the continuous path of a permutation cycle, unwrapping periodic boundary
conditions using winding numbers. Returns a matrix of points [bead_index, dimension].
"""
function extract_extended_path(cfg::WormConfiguration, sys::System, i_start::Int)
    L = sys.L
    M = sys.M
    D = sys.D
    cycle = get_cycle(cfg, i_start)
    k = length(cycle)

    # pts: (k*M + 1) points in D dimensions
    pts = zeros(k * M + 1, D)

    current_offset = zeros(D)
    idx = 1
    for p in cycle
        # Beads 0 to M-1 (Julia indices 1 to M)
        for j in 1:M
            for d in 1:D
                pts[idx, d] = cfg.r[p, j, d] + current_offset[d]
            end
            idx += 1
        end
        # Update offset for next particle in cycle using winding number of p
        for d in 1:D
            current_offset[d] += cfg.w[p, d] * L
        end
    end

    # Final point to complete the trace (bead 0 of i_start shifted by total winding)
    for d in 1:D
        pts[idx, d] = cfg.r[i_start, 1, d] + current_offset[d]
    end

    return pts
end

"""
    fix_all_windings!(cfg::WormConfiguration, L::Float64)

Reset winding numbers for all particles to minimize spring lengths.
"""
function fix_all_windings!(cfg::WormConfiguration, L::Float64)
    N, M, D = size(cfg.r)
    for i in 1:N
        # Current bead M-1
        r_M_minus_1 = cfg.r[i, M, :]

        # Target for closure
        if cfg.sector == G_SECTOR && i == cfg.i_head
            target_0 = cfg.r_c_head
        else
            target_0 = cfg.r[cfg.next[i], 1, :]
        end

        # Choose w to minimize |r_M_minus_1 - (target_0 + w*L)|
        for d in 1:D
            cfg.w[i, d] = round(Int, (r_M_minus_1[d] - target_0[d]) / L)
        end
    end
end

# ═══════════════════════════════════════════════════════════════════════════════
# Interaction Helpers
# ═══════════════════════════════════════════════════════════════════════════════

"""
    get_external_action(cfg, sys, i, j) -> Float64

External potential action for particle i between slices j and j+1.
Uses trapezoidal rule (primitive factorization).
"""
function get_external_action(cfg::WormConfiguration, sys::System, i::Int, j::Int)
    # Early exit for null potential
    if sys.V isa HarmonicPotential && sys.V.k == 0.0
        return 0.0
    end

    L, D, δτ = sys.L, sys.D, sys.τ

    # Get beads j and j+1
    r_j = zeros(D)
    r_next = zeros(D)
    for d in 1:D
        r_j[d] = get_bead(cfg, i, j, d, L)
        r_next[d] = get_bead(cfg, i, j + 1, d, L)
    end

    # Trapezoidal: 0.5 * δτ * (V(r_j) + V(r_{j+1}))
    return 0.5 * δτ * (sys.V(r_j) + sys.V(r_next))
end

"""
    get_interaction_action(cfg, sys, i, j) -> Float64

Total action for particle i at time slice j (between bead j and j+1):
- Pair potential with all other particles

Ref: Spada et al. 2022, Eq. 57 (pair potential only).
"""
function get_interaction_action(cfg::WormConfiguration, sys::System, i::Int, j::Int)
    N, L, D, M = sys.N, sys.L, sys.D, sys.M

    # External potential contribution
    action = get_external_action(cfg, sys, i, j)

    # Pair potential: early exit if single particle or null
    if N == 1 || sys.U isa NullPairPotential
        return action
    end

    λ, δτ = sys.λ, sys.τ
    a_hs = sys.U isa HardSpherePotential ? sys.U.a : 0.0

    # BRANCH REMOVAL: Specialise the loop based on whether j+1 is an endpoint
    if j < M - 1
        # FAST PATH: k,j and k,j+1 are both internal beads (stored in cfg.r)
        @inbounds for k in 1:N
            k == i && continue

            d2_j = 0.0
            d2_next = 0.0
            dot_prod = 0.0
            for d in 1:D
                rk_j_d = cfg.r[k, j+1, d]
                rk_n_d = cfg.r[k, j+2, d]
                ri_j_d = cfg.r[i, j+1, d]
                ri_n_d = cfg.r[i, j+2, d]

                dx_j = ri_j_d - rk_j_d
                dx_j -= L * round(dx_j / L)
                dx_n = ri_n_d - rk_n_d
                dx_n -= L * round(dx_n / L)

                d2_j += dx_j * dx_j
                d2_next += dx_n * dx_n
                dot_prod += dx_j * dx_n
            end

            if sys.U isa HardSpherePotential
                ratio = cao_berne_ratio(sqrt(d2_j), sqrt(d2_next), dot_prod, a_hs, λ, δτ)
                if ratio <= 0.0
                    return Inf
                end
                action -= log(ratio)
            else
                action += (sys.U(sqrt(d2_j)) + sys.U(sqrt(d2_next))) * 0.5 * δτ
            end
        end
    else
        # SLOW PATH: At least one bead is an endpoint (derived)
        @inbounds for k in 1:N
            k == i && continue

            d2_j = 0.0
            d2_next = 0.0
            dot_prod = 0.0
            for d in 1:D
                rk_j_d = get_bead(cfg, k, j, d, L)
                rk_n_d = get_bead(cfg, k, j + 1, d, L)
                ri_j_d = get_bead(cfg, i, j, d, L)
                ri_n_d = get_bead(cfg, i, j + 1, d, L)

                dx_j = ri_j_d - rk_j_d
                dx_j -= L * round(dx_j / L)
                dx_n = ri_n_d - rk_n_d
                dx_n -= L * round(dx_n / L)

                d2_j += dx_j * dx_j
                d2_next += dx_n * dx_n
                dot_prod += dx_j * dx_n
            end

            if sys.U isa HardSpherePotential
                ratio = cao_berne_ratio(sqrt(d2_j), sqrt(d2_next), dot_prod, a_hs, λ, δτ)
                if ratio <= 0.0
                    return Inf
                end
                action -= log(ratio)
            else
                action += (sys.U(sqrt(d2_j)) + sys.U(sqrt(d2_next))) * 0.5 * δτ
            end
        end
    end
    return action
end

"""
    get_interaction_action_external(cfg, sys, i, j, cycle_mask) -> Float64

Interaction action for particle i at slice j with all particles NOT in cycle_mask.
Used to optimize translate! move.
"""
function get_interaction_action_external(cfg::WormConfiguration, sys::System, i::Int, j::Int, cycle_mask::AbstractVector{Bool})
    N, L, D, M = sys.N, sys.L, sys.D, sys.M

    # Include external potential (trap moves with particles)
    action = get_external_action(cfg, sys, i, j)

    # Pair potential: early exit if single particle or null
    if N == 1 || sys.U isa NullPairPotential
        return action
    end

    λ, δτ = sys.λ, sys.τ
    a_hs = sys.U isa HardSpherePotential ? sys.U.a : 0.0

    if j < M - 1
        @inbounds for k in 1:N
            cycle_mask[k] && continue

            d2_j = 0.0
            d2_next = 0.0
            dot_prod = 0.0
            for d in 1:D
                rk_j_d = cfg.r[k, j+1, d]
                rk_n_d = cfg.r[k, j+2, d]
                ri_j_d = cfg.r[i, j+1, d]
                ri_n_d = cfg.r[i, j+2, d]

                dx_j = ri_j_d - rk_j_d
                dx_j -= L * round(dx_j / L)
                dx_n = ri_n_d - rk_n_d
                dx_n -= L * round(dx_n / L)

                d2_j += dx_j * dx_j
                d2_next += dx_n * dx_n
                dot_prod += dx_j * dx_n
            end

            if sys.U isa HardSpherePotential
                ratio = cao_berne_ratio(sqrt(d2_j), sqrt(d2_next), dot_prod, a_hs, λ, δτ)
                if ratio <= 0.0
                    return Inf
                end
                action -= log(ratio)
            else
                action += (sys.U(sqrt(d2_j)) + sys.U(sqrt(d2_next))) * 0.5 * δτ
            end
        end
    else
        @inbounds for k in 1:N
            cycle_mask[k] && continue

            d2_j = 0.0
            d2_next = 0.0
            dot_prod = 0.0
            for d in 1:D
                rk_j_d = get_bead(cfg, k, j, d, L)
                rk_n_d = get_bead(cfg, k, j + 1, d, L)
                ri_j_d = get_bead(cfg, i, j, d, L)
                ri_n_d = get_bead(cfg, i, j + 1, d, L)

                dx_j = ri_j_d - rk_j_d
                dx_j -= L * round(dx_j / L)
                dx_n = ri_n_d - rk_n_d
                dx_n -= L * round(dx_n / L)

                d2_j += dx_j * dx_j
                d2_next += dx_n * dx_n
                dot_prod += dx_j * dx_n
            end

            if sys.U isa HardSpherePotential
                ratio = cao_berne_ratio(sqrt(d2_j), sqrt(d2_next), dot_prod, a_hs, λ, δτ)
                if ratio <= 0.0
                    return Inf
                end
                action -= log(ratio)
            else
                action += (sys.U(sqrt(d2_j)) + sys.U(sqrt(d2_next))) * 0.5 * δτ
            end
        end
    end
    return action
end

"""
    get_interaction_action_segment(cfg, sys, i, j0, j1) -> Float64

Sum of interaction actions for particle i for slices j in j0:j1-1.
"""
function get_interaction_action_segment(cfg::WormConfiguration, sys::System, i::Int, j0::Int, j1::Int)
    S = 0.0
    for j in j0:(j1-1)
        S_slice = get_interaction_action(cfg, sys, i, j)
        if isinf(S_slice)
            return Inf
        end
        S += S_slice
    end
    return S
end

# ═══════════════════════════════════════════════════════════════════════════════
# Lévy Construction for Staging
# ═══════════════════════════════════════════════════════════════════════════════

"""
    levy_sample!(cfg, i, j0, j1, λ, δτ, L)

Sample intermediate beads j0+1 to j1-1 of polymer i via Lévy construction.
j0, j1 in 0:M.
"""
function levy_sample!(cfg::WormConfiguration, i::Int, j0::Int, j1::Int, λ::Float64, δτ::Float64, L::Float64)
    D = size(cfg.r, 3)

    # Sample beads sequentially from j0+1 to j1-1
    for j in (j0+1):(j1-1)
        Δj_fwd = j1 - j
        Δj_back = 1
        denom = Δj_fwd + Δj_back
        σ = sqrt(2λ * δτ * Δj_back * Δj_fwd / denom)

        for d in 1:D
            r_prev_d = get_bead(cfg, i, j - 1, d, L)
            r_end_d = get_bead(cfg, i, j1, d, L)

            # Mean: bridge from r_prev to r_end
            r_star_d = (Δj_fwd * r_prev_d + Δj_back * r_end_d) / denom

            # Update physical storage (beads 0..M-1)
            # Bead j is at index j+1 in Julia
            cfg.r[i, j+1, d] = r_star_d + σ * randn()
        end
    end
end

"""
    ρ_free_sp(r1, r2, λ, Δτ) -> Float64

Single-particle free propagator (unnormalized).
"""
function ρ_free_sp(r1::AbstractVector, r2::AbstractVector, λ::Float64, Δτ::Float64)
    Δr² = sum(abs2, r1 .- r2)
    return exp(-Δr² / (4λ * Δτ))
end

# ═══════════════════════════════════════════════════════════════════════════════
# Monte Carlo Moves
# ═══════════════════════════════════════════════════════════════════════════════

@kwdef mutable struct WormParams
    C::Float64 = 1.0
    j_max::Int = 10
    j_max_open::Int = 10
    j_max_swap::Int = 10
    r_max::Float64 = 0.5
end

"""
    translate!(cfg, sys, params) -> Bool
"""
function translate!(cfg::WormConfiguration, sys::System, params::WormParams)
    N, M, D, L = sys.N, sys.M, sys.D, sys.L
    i = rand(1:N)
    cycle = get_cycle(cfg, i)
    Δr = params.r_max .* (2 .* rand(D) .- 1)

    # Interaction energy change
    in_cycle = zeros(Bool, N)
    for p in cycle
        in_cycle[p] = true
    end

    S_int_old = 0.0
    for p in cycle, j in 0:M-1
        S_int_old += get_interaction_action_external(cfg, sys, p, j, in_cycle)
    end

    # Store old positions and head state
    r_old = copy(cfg.r[cycle, :, :])
    r_c_H_old = copy(cfg.r_c_head)
    w_old = copy(cfg.w[cycle, :])

    # Shift
    for p in cycle
        for j in 1:M
            for d in 1:D
                cfg.r[p, j, d] += Δr[d]
            end
        end
    end

    if cfg.sector == G_SECTOR && any(==(cfg.i_head), cycle)
        i_H = cfg.i_head
        for d in 1:D
            x = cfg.r_c_head[d] + Δr[d]
            x_c, n = _wrap_compact_and_winding(x, L)
            cfg.r_c_head[d] = x_c
            cfg.w[i_H, d] += n
        end
    end

    for p in cycle
        recenter!(cfg, p, L)
    end

    S_int_new = 0.0
    for p in cycle, j in 0:M-1
        S_int_new += get_interaction_action_external(cfg, sys, p, j, in_cycle)
    end

    if rand() < exp(S_int_old - S_int_new)
        return true
    else
        # Revert
        cfg.r[cycle, :, :] .= r_old
        cfg.r_c_head .= r_c_H_old
        cfg.w[cycle, :] .= w_old
        return false
    end
end

"""
    redraw!(cfg, sys, params) -> Bool
"""
function redraw!(cfg::WormConfiguration, sys::System, params::WormParams)
    N, M, D, L = sys.N, sys.M, sys.D, sys.L
    λ, δτ = sys.λ, sys.τ

    if M < 2
        return false
    end

    i = rand(1:N)
    j0 = rand(0:M-1)
    Δj = rand(2:min(params.j_max, M))
    j1 = j0 + Δj

    if j1 > M
        j1 = M
    end

    r_old = copy(cfg.r[i, (j0+1):(min(j1, M)), :])
    S_int_old = get_interaction_action_segment(cfg, sys, i, j0, j1)

    levy_sample!(cfg, i, j0, j1, λ, δτ, L)
    S_int_new = get_interaction_action_segment(cfg, sys, i, j0, j1)

    ΔS = S_int_new - S_int_old

    if rand() < exp(-ΔS)
        if j0 == 0
            recenter!(cfg, i, L)
        end
        return true
    else
        cfg.r[i, (j0+1):(min(j1, M)), :] .= r_old
        return false
    end
end

"""
    open!(cfg, sys, params) -> Bool
"""
function open!(cfg::WormConfiguration, sys::System, params::WormParams)
    if cfg.sector == G_SECTOR
        return false
    end

    N, M, D, L = sys.N, sys.M, sys.D, sys.L
    λ, δτ = sys.λ, sys.τ
    V = L^D

    i_H = rand(1:N)
    j0 = rand(0:M-1)
    Δj = M - j0
    Δ = min(sqrt(2λ * Δj * δτ), L / 2)

    r_old_M = get_endpoint(cfg, i_H, L)
    r_j0 = get_bead(cfg, i_H, j0, L)

    # Propose new head position
    r_new_M = zeros(D)
    for d in 1:D
        r_new_M[d] = r_old_M[d] + Δ * (2 * rand() - 1)
    end

    # Store old state
    r_old_segment = copy(cfg.r[i_H, (j0+1):M, :])
    w_old = copy(cfg.w[i_H, :])
    S_int_old = get_interaction_action_segment(cfg, sys, i_H, j0, M)

    # Switch to G-sector temporarily
    cfg.sector = G_SECTOR
    cfg.i_head = i_H
    cfg.i_tail = cfg.next[i_H]

    # Initialize r_c_head and head winding
    for d in 1:D
        x_c, n = _wrap_compact_and_winding(r_new_M[d], L)
        cfg.r_c_head[d] = x_c
        cfg.w[i_H, d] = n
    end

    levy_sample!(cfg, i_H, j0, M, λ, δτ, L)
    S_int_new = get_interaction_action_segment(cfg, sys, i_H, j0, M)

    # Free propagator ratio
    ρ_new = ρ_free_sp(r_j0, r_new_M, λ, Δj * δτ)
    ρ_old = ρ_free_sp(r_j0, r_old_M, λ, Δj * δτ)

    prefactor = params.C * N * (2 * Δ)^D / V
    A_O = prefactor * (ρ_new / ρ_old) * exp(S_int_old - S_int_new)

    if rand() < min(1.0, A_O)
        return true
    else
        # Revert
        cfg.sector = Z_SECTOR
        cfg.r[i_H, (j0+1):M, :] .= r_old_segment
        cfg.w[i_H, :] .= w_old
        return false
    end
end

"""
    close!(cfg, sys, params) -> Bool
"""
function close!(cfg::WormConfiguration, sys::System, params::WormParams)
    if cfg.sector == Z_SECTOR
        return false
    end

    N, M, D, L = sys.N, sys.M, sys.D, sys.L
    λ, δτ = sys.λ, sys.τ
    V = L^D

    i_H = cfg.i_head
    i_T = cfg.i_tail

    j0 = rand(0:M-1)
    Δj = M - j0
    Δ = min(sqrt(2λ * Δj * δτ), L / 2)

    r_head_M = get_endpoint(cfg, i_H, L)
    r_tail_0 = cfg.r[i_T, 1, :]

    # Find nearest image of tail for closure
    r_T = zeros(D)
    for d in 1:D
        n = round(Int, (r_head_M[d] - r_tail_0[d]) / L)
        r_T[d] = r_tail_0[d] + n * L
    end

    if any(abs.(r_head_M .- r_T) .> Δ)
        return false
    end

    # Store old state
    r_old_segment = copy(cfg.r[i_H, (j0+1):M, :])
    w_old = copy(cfg.w[i_H, :])
    r_old_M = copy(r_head_M)
    r_j0 = get_bead(cfg, i_H, j0, L)
    S_int_old = get_interaction_action_segment(cfg, sys, i_H, j0, M)

    # Propose closure: update winding to match r_T
    for d in 1:D
        cfg.w[i_H, d] = round(Int, (r_T[d] - r_tail_0[d]) / L)
    end

    # Temporarily switch to Z-sector
    cfg.sector = Z_SECTOR

    levy_sample!(cfg, i_H, j0, M, λ, δτ, L)
    S_int_new = get_interaction_action_segment(cfg, sys, i_H, j0, M)

    ρ_new = ρ_free_sp(r_j0, r_T, λ, Δj * δτ)
    ρ_old = ρ_free_sp(r_j0, r_old_M, λ, Δj * δτ)

    prefactor = V / (params.C * N * (2 * Δ)^D)
    A_C = prefactor * (ρ_new / ρ_old) * exp(S_int_old - S_int_new)

    if rand() < min(1.0, A_C)
        cfg.i_head = 1
        cfg.i_tail = 1
        return true
    else
        cfg.sector = G_SECTOR
        cfg.r[i_H, (j0+1):M, :] .= r_old_segment
        cfg.w[i_H, :] .= w_old
        return false
    end
end

"""
    move_head!(cfg, sys, params) -> Bool
"""
function move_head!(cfg::WormConfiguration, sys::System, params::WormParams)
    if cfg.sector == Z_SECTOR
        return false
    end
    M, D, L = sys.M, sys.D, sys.L
    λ, δτ = sys.λ, sys.τ
    i_H = cfg.i_head

    j0 = rand(0:M-1)
    Δj = M - j0

    r_old_segment = copy(cfg.r[i_H, (j0+1):M, :])
    w_old = copy(cfg.w[i_H, :])
    r_c_head_old = copy(cfg.r_c_head)

    r_j0 = get_bead(cfg, i_H, j0, L)
    S_int_old = get_interaction_action_segment(cfg, sys, i_H, j0, M)

    # Sample new head endpoint
    σ = sqrt(2λ * Δj * δτ)
    r_new_M = zeros(D)
    for d in 1:D
        r_new_M[d] = r_j0[d] + σ * randn()
        x_c, n = _wrap_compact_and_winding(r_new_M[d], L)
        cfg.r_c_head[d] = x_c
        cfg.w[i_H, d] = n
    end

    levy_sample!(cfg, i_H, j0, M, λ, δτ, L)
    S_int_new = get_interaction_action_segment(cfg, sys, i_H, j0, M)

    if rand() < exp(S_int_old - S_int_new)
        return true
    else
        # Revert
        cfg.r[i_H, (j0+1):M, :] .= r_old_segment
        cfg.w[i_H, :] .= w_old
        cfg.r_c_head .= r_c_head_old
        return false
    end
end

"""
    move_tail!(cfg, sys, params) -> Bool
"""
function move_tail!(cfg::WormConfiguration, sys::System, params::WormParams)
    if cfg.sector == Z_SECTOR
        return false
    end
    M, D, L = sys.M, sys.D, sys.L
    λ, δτ = sys.λ, sys.τ
    i_T = cfg.i_tail

    j1 = rand(1:M)
    Δj = j1

    r_old_segment = copy(cfg.r[i_T, 1:j1, :])
    r_j1 = get_bead(cfg, i_T, j1, L)
    S_int_old = get_interaction_action_segment(cfg, sys, i_T, 0, j1)

    # Sample new bead 0
    σ = sqrt(2λ * Δj * δτ)
    for d in 1:D
        cfg.r[i_T, 1, d] = r_j1[d] + σ * randn()
    end

    levy_sample!(cfg, i_T, 0, j1, λ, δτ, L)
    S_int_new = get_interaction_action_segment(cfg, sys, i_T, 0, j1)

    if rand() < exp(S_int_old - S_int_new)
        recenter!(cfg, i_T, L)
        return true
    else
        cfg.r[i_T, 1:j1, :] .= r_old_segment
        return false
    end
end

"""
    swap!(cfg, sys, params) -> Bool
"""
function swap!(cfg::WormConfiguration, sys::System, params::WormParams)
    if cfg.sector == Z_SECTOR
        return false
    end

    N, M, D, L = sys.N, sys.M, sys.D, sys.L
    λ, δτ = sys.λ, sys.τ
    i_H = cfg.i_head
    i_T = cfg.i_tail

    # Select pivot time slice j_P from 1 to M
    j_max_possible = min(params.j_max_swap, M)
    if j_max_possible < 1
        return false
    end
    j_P = rand(1:j_max_possible)
    r_c_H = cfg.r_c_head

    # Tower sampling for target i_0 (Spada Eq. 31)
    weights = zeros(N)
    for i in 1:N
        r_i_jP = get_bead(cfg, i, j_P, L)
        weights[i] = ρ_free_sp(r_c_H, r_i_jP, λ, j_P * δτ)
    end

    Σ_P = sum(weights)
    if Σ_P ≤ 0
        return false
    end

    r_sample = rand() * Σ_P
    cumsum_w = 0.0
    i_0 = 0
    for i in 1:N
        cumsum_w += weights[i]
        if r_sample < cumsum_w
            i_0 = i
            break
        end
    end

    if i_0 == 0 || i_0 == i_T
        return false
    end

    # Σ_0 for inverse process
    r_c_i0_0 = copy(cfg.r[i_0, 1, :])
    weights_inv = zeros(N)
    for i in 1:N
        r_i_jP = get_bead(cfg, i, j_P, L)
        weights_inv[i] = ρ_free_sp(r_c_i0_0, r_i_jP, λ, j_P * δτ)
    end
    Σ_0 = sum(weights_inv)
    if Σ_0 ≤ 0
        return false
    end

    i_star = cfg.prev[i_0]
    S_int_old = get_interaction_action_segment(cfg, sys, i_0, 0, j_P)

    r_old_segment = copy(cfg.r[i_0, 1:j_P, :])
    r_c_H_old = copy(cfg.r_c_head)

    # 1. Relocate i_0 bead 0 to the old head position
    cfg.r[i_0, 1, :] .= r_c_H_old

    # 2. Relocate head compact coordinate to the old i_0 bead 0 position
    cfg.r_c_head .= r_c_i0_0

    # 3. Redraw segment of i_0 from 0 to j_P
    levy_sample!(cfg, i_0, 0, j_P, λ, δτ, L)
    S_int_new = get_interaction_action_segment(cfg, sys, i_0, 0, j_P)

    A_SW = (Σ_P / Σ_0) * exp(S_int_old - S_int_new)

    if rand() < min(1.0, A_SW)
        # Update permutations
        cfg.next[i_H] = i_0
        cfg.prev[i_0] = i_H
        cfg.next[i_star] = i_T
        cfg.prev[i_T] = i_star
        cfg.i_head = i_star

        # Update windings:
        # i_H used to connect to head (broken). Now it connects to i_0.
        # r_{M, i_H} = r_c_H + w_{i_H} * L
        # New target i_0 is AT r_c_H. So r_{0, i_0} = r_c_H.
        # So w_{i_H} remains the same to keep r_{M, i_H} fixed.
        # Likewise for i_star (new head) relating to its target (broken).
        # We don't need to change cfg.w values because of our relocation strategy.

        recenter!(cfg, i_0, L)
        update_head_compact!(cfg, L)
        return true
    else
        # Revert
        cfg.r[i_0, 1:j_P, :] .= r_old_segment
        cfg.r_c_head .= r_c_H_old
        return false
    end
end

"""
    energy_estimators(cfg, sys) -> (E_thermo, E_virial)

Thermodynamic and Virial energy estimators (Spada et al. 2022, Eq. A1-A2).
Returns (E_thermo, E_virial) tuple.
"""
function energy_estimators(cfg::WormConfiguration, sys::System)
    N, M, D = sys.N, sys.M, sys.D
    λ, δτ, β, L = sys.λ, sys.τ, sys.β, sys.L
    r = cfg.r  # Direct array access: r[particle, bead+1, dim]

    # Accumulators
    K_spring = 0.0   # Spring term (thermodynamic)
    Virial_G1 = 0.0  # Virial Term 2: endpoint/winding correction
    Virial_G2 = 0.0  # Virial Term 3: force × displacement
    V_total = 0.0    # Potential energy
    Term4 = 0.0      # Cao-Berne ∂U/∂δτ

    has_ext = !(sys.V isa HarmonicPotential && sys.V.k == 0.0)
    has_pair = N > 1 && !(sys.U isa NullPairPotential)
    is_hs = sys.U isa HardSpherePotential
    a_hs = is_hs ? sys.U.a : 0.0

    # Temp arrays for position vectors (allocate once)
    r_vec = zeros(Float64, D)
    rij = zeros(Float64, D)
    grad_vec = zeros(Float64, D)

    # =========================================================================
    # SINGLE-PARTICLE TERMS
    # =========================================================================
    @inbounds for i in 1:N
        # --- Virial Term 2: (R_{M-1} - R_M) · (R_M - R_0) ---
        for d in 1:D
            rm1 = r[i, M, d]                        # bead M-1
            rm = get_endpoint(cfg, i, d, L)          # bead M (endpoint)
            r0 = r[i, 1, d]                          # bead 0
            Virial_G1 += (rm1 - rm) * (rm - r0)
        end

        # --- Spring term and external potential ---
        for j in 1:M  # j is array index (bead index = j-1)
            # Spring: |r[j] - r[j+1]|² with MIC
            for d in 1:D
                if j < M
                    dx = r[i, j, d] - r[i, j+1, d]
                else
                    dx = r[i, M, d] - get_endpoint(cfg, i, d, L)
                end
                dx -= L * round(dx / L)  # MIC
                K_spring += dx * dx
            end

            # External potential V(r)
            if has_ext
                for d in 1:D
                    r_vec[d] = r[i, j, d]
                end
                V_total += sys.V(r_vec)

                # Virial: (r - r_0) · ∇V
                potential_derivative!(grad_vec, sys.V, r_vec)
                for d in 1:D
                    Virial_G2 += (r[i, j, d] - r[i, 1, d]) * grad_vec[d]
                end
            end
        end
    end

    # =========================================================================
    # PAIR INTERACTION TERMS
    # =========================================================================
    if has_pair
        @inbounds for j in 1:M
            for i in 1:N-1, k in i+1:N
                # Distance |r_i - r_k| with MIC
                d2_j = 0.0
                for d in 1:D
                    dx = r[i, j, d] - r[k, j, d]
                    dx -= L * round(dx / L)
                    d2_j += dx * dx
                end
                dist_j = sqrt(d2_j)

                if is_hs
                    # --- Hard-sphere (Cao-Berne) ---
                    # Next slice distance and dot product
                    d2_n, dot_jn = 0.0, 0.0
                    for d in 1:D
                        r_i_j = r[i, j, d]
                        r_k_j = r[k, j, d]
                        dx_j = r_i_j - r_k_j
                        dx_j -= L * round(dx_j / L)

                        if j < M
                            r_i_n = r[i, j+1, d]
                            r_k_n = r[k, j+1, d]
                        else
                            r_i_n = get_endpoint(cfg, i, d, L)
                            r_k_n = get_endpoint(cfg, k, d, L)
                        end
                        dx_n = r_i_n - r_k_n
                        dx_n -= L * round(dx_n / L)

                        d2_n += dx_n * dx_n
                        dot_jn += dx_j * dx_n
                    end
                    dist_n = sqrt(d2_n)

                    # Cao-Berne derivatives
                    du_dr, du_dr_n, du_dcos, du_dτ = cao_berne_derivatives(dist_j, dist_n, dot_jn, a_hs, λ, δτ)
                    isinf(du_dr) && return (NaN, NaN)
                    Term4 += du_dτ

                    # Virial force terms
                    cos_θ = clamp(dot_jn / (dist_j * dist_n), -1.0, 1.0)
                    f1 = (du_dr - du_dcos * cos_θ / dist_j) / dist_j
                    f2 = du_dcos / (dist_j * dist_n)
                    f3 = (du_dr_n - du_dcos * cos_θ / dist_n) / dist_n
                    f4 = f2

                    for d in 1:D
                        dx_j = r[i, j, d] - r[k, j, d]
                        dx_j -= L * round(dx_j / L)

                        if j < M
                            ri_n, rk_n = r[i, j+1, d], r[k, j+1, d]
                        else
                            ri_n = get_endpoint(cfg, i, d, L)
                            rk_n = get_endpoint(cfg, k, d, L)
                        end
                        dx_n = ri_n - rk_n
                        dx_n -= L * round(dx_n / L)

                        # Force vectors
                        g_j = f1 * dx_j + f2 * dx_n
                        g_n = f3 * dx_n + f4 * dx_j

                        # Virial: ((r_i - r_{i,0}) - (r_k - r_{k,0})) · g
                        diff_j = (r[i, j, d] - r[i, 1, d]) - (r[k, j, d] - r[k, 1, d])
                        diff_n = (ri_n - r[i, 1, d]) - (rk_n - r[k, 1, d])
                        Virial_G2 += diff_j * g_j + diff_n * g_n
                    end

                else
                    # --- Soft pair potential ---
                    V_total += sys.U(dist_j)

                    for d in 1:D
                        dx = r[i, j, d] - r[k, j, d]
                        rij[d] = dx - L * round(dx / L)
                    end
                    potential_derivative!(grad_vec, sys.U, rij)

                    for d in 1:D
                        diff = (r[i, j, d] - r[i, 1, d]) - (r[k, j, d] - r[k, 1, d])
                        Virial_G2 += diff * grad_vec[d]
                    end
                end
            end
        end
    end

    # =========================================================================
    # ASSEMBLY (Spada Eq. A1 and A2)
    # =========================================================================
    E_thermo = D * N / (2δτ) - K_spring / (4λ * δτ^2 * M) + Term4 / M + V_total / M
    E_virial = D * N / (2β) + Virial_G1 / (4λ * δτ^2 * M) + Virial_G2 / (2β * M) + Term4 / M + V_total / M

    return E_thermo, E_virial
end

"""
    worm_step!(cfg, sys, params) -> Symbol
"""
function worm_step!(cfg::WormConfiguration, sys::System, params::WormParams)
    r = rand()
    N = sys.N

    if N == 1
        if r < 0.15
            return translate!(cfg, sys, params) ? :translate : :reject
        elseif r < 0.45
            return redraw!(cfg, sys, params) ? :redraw : :reject
        elseif r < 0.55
            return open!(cfg, sys, params) ? :open : :reject
        elseif r < 0.65
            return close!(cfg, sys, params) ? :close : :reject
        elseif r < 0.825
            return move_head!(cfg, sys, params) ? :move_head : :reject
        else
            return move_tail!(cfg, sys, params) ? :move_tail : :reject
        end
    else
        if r < 0.10
            return translate!(cfg, sys, params) ? :translate : :reject
        elseif r < 0.35
            return redraw!(cfg, sys, params) ? :redraw : :reject
        elseif r < 0.45
            return open!(cfg, sys, params) ? :open : :reject
        elseif r < 0.55
            return close!(cfg, sys, params) ? :close : :reject
        elseif r < 0.70
            return swap!(cfg, sys, params) ? :swap : :reject
        elseif r < 0.85
            return move_head!(cfg, sys, params) ? :move_head : :reject
        else
            return move_tail!(cfg, sys, params) ? :move_tail : :reject
        end
    end
end

"""
    total_winding(cfg::WormConfiguration) -> Vector{Int}

Return the total winding vector W = sum_i w_i across all particles.
"""
function total_winding(cfg::WormConfiguration)
    return vec(sum(cfg.w, dims=1))
end

"""
    superfluid_fraction(cfg::WormConfiguration, sys::System) -> Float64

Estimate the instantaneous superfluid fraction from the winding number.
Note: This should be averaged over many Z-sector configurations.
Ref: Ceperley 1995, Eq. 34
"""
function superfluid_fraction(cfg::WormConfiguration, sys::System)
    W = total_winding(cfg)
    W2 = sum(abs2, W)
    # W is integer winding. W_physical = W * L.
    # rho_s/rho = <W_physical^2> / (2 * D * lambda * beta * N)
    #           = <W^2> * L^2 / (2 * D * lambda * beta * N)
    return (W2 * sys.L^2) / (2 * sys.D * sys.λ * sys.β * sys.N)
end

