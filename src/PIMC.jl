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

function localMove!(sys,path)
    particle=rand(1:sys.nparticles)
    bead=rand(1:sys.nbeads)
    Δ=randn(sys.spatialdimensions)

    pold=path.r[particle,bead,:]
    pnew=pold+Δ

    Sold=Harmonic(pold)+
            (sum(pold.-path.r[path.prev[particle,bead],mod1(bead-1,sys.nbeads),:])).^2
            + (sum(pold.-path.r[path.next[particle,bead],mod1(bead+1,sys.nbeads),:])).^2

    Snew=Harmonic(pnew)+
    (sum(pnew.-path.r[path.prev[particle,bead],mod1(bead-1,sys.nbeads),:])).^2
    + (sum(pnew.-path.r[path.next[particle,bead],mod1(bead+1,sys.nbeads),:])).^2

    ΔS=Snew-Sold
    
    println("localMove! particle=$particle bead=$bead Δ=$Δ pold=$pold pnew=$pnew Sold=$Sold Snew=$Snew ΔS=$ΔS")

    if ΔS ≤ 0 || rand() < exp(-sys.β * ΔS)
        println("Accept!")
        path.r[particle,bead,:]=pnew
    end
end


