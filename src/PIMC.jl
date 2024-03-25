# PIMC.jl
# Initially location for all the path integral bits and bobs

# Very generic path

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

function localMove!(system,path; moves=1, verbose=false, stepsize=0.1)
    ACCEPT=0
    REJECT=0

    for m in 1:moves # this loop brought within the function
        # I don't understand why, but otherwise you get a lot of allocations (12!) from
        # dereferencing system.potential for every run of the function. 

    particle=rand(1:system.nparticles)
    bead=rand(1:system.nbeads)
    Δ=stepsize*randn(system.spatialdimensions)

    @inbounds begin
    pold=@view path.r[particle,bead,:]
    pnew=pold+Δ

    
    Sold = Harmonic(pold) +
        sum(abs2, (pold .- @view path.r[path.prev[particle, bead], mod1(bead - 1, system.nbeads), :])) +
        sum(abs2, (pold .- @view path.r[path.next[particle, bead], mod1(bead + 1, system.nbeads), :]))

    Snew = Harmonic(pnew) +
        sum(abs2, (pnew .- @view path.r[path.prev[particle, bead], mod1(bead - 1, system.nbeads), :])) +
        sum(abs2, (pnew .- @view path.r[path.next[particle, bead], mod1(bead + 1, system.nbeads), :]))
    end

    ΔS=Snew-Sold
    
    if verbose
        println("localMove! particle=$particle bead=$bead Δ=$Δ pold=$pold pnew=$pnew Sold=$Sold Snew=$Snew ΔS=$ΔS")
    end

    if ΔS ≤ 0 || rand() < exp(-system.β * ΔS)
        if verbose println("Accept!") end
        ACCEPT+=1
        @inbounds path.r[particle,bead,:]=pnew
    else 
        REJECT+=1
    end

    end

    println("Moves: $moves Accept: $ACCEPT Reject: $REJECT Ratio: $(ACCEPT/moves)")
end


