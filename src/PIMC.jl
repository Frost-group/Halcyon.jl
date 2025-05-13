# PIMC.jl
# Initial location for all the path integral bits and bobs

# Generic as possible path

# Assumes SliceA, SliceB
function KineticAction(sA,sB)
    KE=(sA.r .- sB.r).^2
    KE=KE/4*Lambda*tau
    KE
end

function PotentialAction(sA,sB) #sliceA, sliceB
    # Primitive action
    PE=Vext(sA)+Vext(sB) + Vint(sA,sB)
    PE=PE*0.5*tau
    PE
end

function total_energy(sys, path; V=Harmonic, U=Coulomb)
    # Initialize energy components
    kinetic = 0.0
    potential = 0.0
    
    # Constants
    τ = sys.β / sys.M
    
    # Loop over all particles and beads
    for p in 1:sys.N
        for b in 1:sys.M
            @inbounds begin
                # Get current bead and next/previous beads
                sA = @view path.r[p,b,:]
                prev_bead = @view path.r[path.prev[p,b], mod1(b-1, sys.M), :]
                next_bead = @view path.r[path.next[p,b], mod1(b+1, sys.M), :]

                # Spring terms with correct factors
                kinetic  += ( sum(abs2, (sA .- prev_bead)) + 
                              sum(abs2, (sA .- next_bead)) ) / 4τ
                
                # External potential (factor of 2 (!?) as in Thijssen)
                potential += 2.0 * V(sA)
                
                # Interaction potential if multiple particles
                if sys.N > 1
                    potential += sum(U.(reinterpret(SVector{3,Float64}, (sA' .- @view path.r[p,:,:])')))
                end
            end
        end
    end
    
    # Average over beads, and particles, for comparison with QHO answer
    total = (kinetic + potential) / ( sys.M * sys.N )
    
    return total
end

#### Local moves ####

function localMove!(sys, path; moves=1, verbose=false, stepsize=0.4, V=Harmonic, U=NullPotential)
    ACCEPT = 0
    REJECT = 0
    τ = sys.β / sys.M

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

            # Calculate spring terms with correct factors
            Sold = sum(abs2, τ* (r_old .- prev_bead)) + 
                  sum(abs2, τ* (r_old .- next_bead))
            Snew = sum(abs2, τ* (r_new .- prev_bead)) + 
                  sum(abs2, τ* (r_new .- next_bead))

            # Add (single body) potential terms
            Sold += 2τ * V(r_old)
            Snew += 2τ * V(r_new)

            # Add interaction (two-body potential) terms if multiple particles
            if sys.N > 1
                Sold += τ * sum(U.(reinterpret(SVector{3,Float64}, (r_old' .- @view path.r[p,:,:])')))
                Snew += τ * sum(U.(reinterpret(SVector{3,Float64}, (r_new' .- @view path.r[p,:,:])')))
            end
        end

        ΔS = 0.5 * (Snew - Sold) # Factor of 1/2 from primitive approximation

        if verbose
            println("localMove! p=$p b=$b Δ=$Δ r_old=$r_old r_new=$r_new Sold=$Sold Snew=$Snew ΔS=$ΔS")
        end

        # Metropolis critereon
        if ΔS ≤ 0 || rand() < exp(-sys.β * ΔS)
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




