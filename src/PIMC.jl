# PIMC.jl
using ForwardDiff
# Initial location for all the path integral bits and bobs

"""
    total_energy(sys::System, path::Path) -> Float64

Primitive (? is that the correct term) PIMC energy estimator: spring kinetic +
external one-body + pairwise interactions, averaged over beads.
"""
function total_energy(sys::System, path::Path)
    τ, λ, M, N = sys.τ, sys.λ, sys.M, sys.N
    kinetic = 0.0
    v_ext   = 0.0
    v_pair  = 0.0

    # Kinetic (springs) and external potential
    @inbounds for b in 1:M
        for p in 1:N
            r  = @view path.r[p, b, :]
            rp = @view path.r[p, mod1(b - 1, M), :]
            kinetic += sys.m * sum(abs2, r .- rp) / (4λ * τ)
            v_ext   += sys.V(r)
        end
        # Pair interactions counted once per pair at bead b
        if N > 1
            for p in 1:N-1
                rp = @view path.r[p, b, :]
                for q in p+1:N
                    rq = @view path.r[q, b, :]
                    v_pair += sys.U(rp .- rq)
                end
            end
        end
    end

    return (kinetic + v_ext + v_pair) / M
end

#### Local moves ####

function localMove!(sys::System, path::Path; moves::Integer=1, verbose::Bool=false, stepsize::Float64=0.4)
    ACCEPT = 0
    REJECT = 0
    τ = sys.τ
    λ = sys.λ

    for m in 1:moves # this loop brought within the function
        # I don't understand why, but otherwise you get a lot of allocations (12?!) from
        # dereferencing system.potential for every run of the function.
        
        p = rand(1:sys.N) # (p)article
        b = rand(1:sys.M)     # (b)ead
        Δ = stepsize * randn(sys.D)

        @inbounds begin
            r_old = @view path.r[p,b,:]
            r_new = r_old + Δ
            
            # Get neighboring beads
            prev_bead = @view path.r[path.prev[p,b], mod1(b-1, sys.M), :]
            next_bead = @view path.r[path.next[p,b], mod1(b+1, sys.M), :]

            # Calculate spring terms with correct factors for both adjacent links to bead b
            Sold = sum(abs2, (r_old .- prev_bead)) / (4λ*τ) +
                   sum(abs2, (next_bead .- r_old)) / (4λ*τ)
            Snew = sum(abs2, (r_new .- prev_bead)) / (4λ*τ) +
                   sum(abs2, (next_bead .- r_new)) / (4λ*τ)

            # Add (single body) potential terms
            Sold += τ * sys.V(r_old)
            Snew += τ * sys.V(r_new)

            # Add interaction (two-body potential) terms if multiple particles
            if sys.N > 1
                @inbounds for q in 1:sys.N
                    q == p && continue
                    rq = @view path.r[q,b,:]
                    Sold += τ * sys.U(r_old .- rq)
                    Snew += τ * sys.U(r_new .- rq)
                end
            end
        end

        ΔS = (Snew - Sold)

        if verbose
            println("localMove! p=$p b=$b Δ=$Δ r_old=$r_old r_new=$r_new Sold=$Sold Snew=$Snew ΔS=$ΔS")
        end

        # Metropolis criterion for path integral action
        if ΔS ≤ 0 || rand() < exp(-ΔS)
            if verbose println("Accept!") end
            ACCEPT+=1
            @inbounds path.r[p,b,:]=r_new
        else 
            REJECT+=1
        end
    end

    println("Moves: $moves Accept: $ACCEPT Reject: $REJECT Ratio: $(ACCEPT/moves)")
end



#### Worm algorithm ####
# Sisters Ira and Masha do their magic #



#### G-sector (Matsubara frequencies) moves ####




