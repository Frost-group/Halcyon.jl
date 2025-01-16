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
    ω=ℏ=1 # fudge
    
    
    # Loop over all particles and beads
    for p in 1:sys.nparticles
        for b in 1:sys.nbeads
            next_b = mod1(b+1, sys.nbeads)
            
            @inbounds begin
                # Get current and next slice
                sA = @view path.r[p,b,:]
                sB = @view path.r[p,next_b,:]

                # Spring term for kinetic energy
                Δr² = sum(abs2, sA .- sB)
                kinetic += sys.mass / (2τ * ℏ^2) * Δr²
                
                # External harmonic potential
                potential += 0.5 * sys.mass * ω^2 * sum(abs2, sA)
            end
        end
    end
    
    total = kinetic + potential
    total /= sys.nbeads
    
    return total
end

function localMove!(sys,path; moves=1, verbose=false, stepsize=0.5, V=Harmonic, U=Coulomb)
    ACCEPT=0
    REJECT=0

    for m in 1:moves # this loop brought within the function
        # I don't understand why, but otherwise you get a lot of allocations (12?!) from
        # dereferencing system.potential for every run of the function. 

        # pick random particle, random bead, attempted 'simple' move
        p=rand(1:sys.nparticles) # (p)article
        b=rand(1:sys.nbeads)     # (b)ead
        Δ=stepsize*randn(sys.spatialdimensions)

        @inbounds begin # hold my beer
        pold=@view path.r[p,b,:]
        pnew=pold+Δ
        
# reinterpret(SVector{3, Float64}, from 
# https://discourse.julialang.org/t/broadcasting-across-columns-of-a-matrix/18496/3?u=jarvist

# I was being far too clever for my own good here.
        Sold = 
        V(pold) +
        sum(U.(reinterpret(SVector{3, Float64}, (pold' .- @view path.r[p,:,:])'))) +
        sum(abs2, (pold .- @view path.r[path.prev[p, b], mod1(b- 1, sys.nbeads), :])) +
            sum(abs2, (pold .- @view path.r[path.next[p, b], mod1(b+ 1, sys.nbeads), :]))

        Snew = 
        V(pnew) +
        sum(U.(reinterpret(SVector{3, Float64}, (pnew' .- @view path.r[p,:,:])'))) +            
        sum(abs2, (pnew .- @view path.r[path.prev[p, b], mod1(b- 1, sys.nbeads), :])) +
        sum(abs2, (pnew .- @view path.r[path.next[p, b], mod1(b+ 1, sys.nbeads), :]))
        end

    ΔS=Snew-Sold
    
    if verbose
        println("localMove! particle=$p bead=$b Δ=$Δ pold=$pold pnew=$pnew Sold=$Sold Snew=$Snew ΔS=$ΔS")
    end

    # Metropolis critereon
    if ΔS ≤ 0 || rand() < exp(-sys.β * ΔS)
        if verbose println("Accept!") end
        ACCEPT+=1
        @inbounds path.r[p,b,:]=pnew
    else 
        REJECT+=1
    end

    end

    println("Moves: $moves Accept: $ACCEPT Reject: $REJECT Ratio: $(ACCEPT/moves)")
end




