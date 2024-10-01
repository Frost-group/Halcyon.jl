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

function localMove!(sys,path; moves=1, verbose=false, stepsize=0.1)
    ACCEPT=0
    REJECT=0

    for m in 1:moves # this loop brought within the function
        # I don't understand why, but otherwise you get a lot of allocations (12?!) from
        # dereferencing system.potential for every run of the function. 

        # pick random particle, random bead, attempted 'simple' move
        p=rand(1:sys.nparticles) # (p)article
        b=rand(1:sys.nbeads)     # (b)ead
        Δ=stepsize*randn(sys.spatialdimensions)

        @inbounds begin
        pold=@view path.r[p,b,:]
        pnew=pold+Δ
        
        Sold = Harmonic(pold) +
            sum(abs2, (pold .- @view path.r[path.prev[p, b], mod1(b- 1, sys.nbeads), :])) +
            sum(abs2, (pold .- @view path.r[path.next[p, b], mod1(b+ 1, sys.nbeads), :]))

        Snew = Harmonic(pnew) +
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


