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
    τ = sys.β / sys.nbeads
    
    # Loop over all particles and beads
    for p in 1:sys.nparticles
        for b in 1:sys.nbeads
            @inbounds begin
                # Get current bead and neighbors
                sA = @view path.r[p,b,:]
                prev_bead = @view path.r[path.prev[p,b], mod1(b-1, sys.nbeads), :]
                next_bead = @view path.r[path.next[p,b], mod1(b+1, sys.nbeads), :]

                # Spring terms with correct factors
                spring_energy = sum(abs2, τ* (sA .- prev_bead)) + 
                              sum(abs2, τ* (sA .- next_bead))
                kinetic += spring_energy / (4τ)
                
                # External potential (factor of 2 (!?) as in Thijssen)
                potential += 2.0 * V(sA)
                
                # Interaction potential if multiple particles
                if sys.nparticles > 1
                    potential += sum(U.(reinterpret(SVector{3,Float64}, (sA' .- @view path.r[p,:,:])')))
                end
            end
        end
    end
    
    # Average over beads, and particles, for comparison with QHO answer
    total = (kinetic + potential) / ( sys.nbeads * sys.nparticles )
    
    return total
end

function localMove!(sys, path; moves=1, verbose=false, stepsize=0.2, V=Harmonic, U=NullPotential)
    ACCEPT = 0
    REJECT = 0
    τ = sys.β / sys.nbeads

    for m in 1:moves # this loop brought within the function
        # I don't understand why, but otherwise you get a lot of allocations (12?!) from
        # dereferencing system.potential for every run of the function.
        
        p = rand(1:sys.nparticles) # (p)article
        b = rand(1:sys.nbeads)     # (b)ead
        Δ = stepsize * randn(sys.spatialdimensions)
# reinterpret(SVector{3, Float64}, from 
# https://discourse.julialang.org/t/broadcasting-across-columns-of-a-matrix/18496/3?u=jarvist

# I was being far too clever for my own good here.
#        Sold = 
#        V(pold) +
#        sum(U.(reinterpret(SVector{3, Float64}, (pold' .- @view path.r[p,:,:])'))) +
#        sum(abs2, (pold .- @view path.r[path.prev[p, b], mod1(b- 1, sys.nbeads), :])) +
#            sum(abs2, (pold .- @view path.r[path.next[p, b], mod1(b+ 1, sys.nbeads), :]))

#        Snew = 
#        V(pnew) +
#        sum(U.(reinterpret(SVector{3, Float64}, (pnew' .- @view path.r[p,:,:])'))) +            
#        sum(abs2, (pnew .- @view path.r[path.prev[p, b], mod1(b- 1, sys.nbeads), :])) +
#        sum(abs2, (pnew .- @view path.r[path.next[p, b], mod1(b+ 1, sys.nbeads), :]))


        @inbounds begin
            r_old = @view path.r[p,b,:]
            r_new = r_old + Δ
            
            # Get neighboring beads
            prev_bead = @view path.r[path.prev[p,b], mod1(b-1, sys.nbeads), :]
            next_bead = @view path.r[path.next[p,b], mod1(b+1, sys.nbeads), :]

            # Calculate spring terms with correct factors
            Sold = sum(abs2, τ* (r_old .- prev_bead)) + 
                  sum(abs2, τ* (r_old .- next_bead))
            Snew = sum(abs2, τ* (r_new .- prev_bead)) + 
                  sum(abs2, τ* (r_new .- next_bead))

            # Add potential terms
            Sold += 2τ * V(r_old)
            Snew += 2τ * V(r_new)

            # Add interaction terms if multiple particles
            if sys.nparticles > 1
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




