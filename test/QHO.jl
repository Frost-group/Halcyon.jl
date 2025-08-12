using Test
using Halcyon

@testset "Analytic: QHO" begin
    # Units: m=1, set harmonic V(x)=0.5*x^2, so ω=1

    for D in (1, 2, 3)
        println("QHO Dimension: D = $D")
        sys = System(20, 1; D=D, β=10.0, V=HarmonicPotential(k=0.5), U=NullPairPotential(), λ=0.5)
        # Beads ≈ 20 × β × ω for tighter Trotter error
        ω = sqrt(2*sys.V.k / sys.m)
        M = ceil(Int, 20 * sys.β * ω)
        println("Calculated beads for 1% error: M = $M")
        
        sys = System(M, 1; D=D, β=10.0, V=HarmonicPotential(k=0.5), U=NullPairPotential(), λ=0.5)
        path = Path(sys)

        # Warm up chain
        localMove!(sys, path, moves=200_000)
        # Sample virial energy along the chain
        sweeps = 20
        spacing = 20_000
        vir_vals = Float64[]
        prim_vals = Float64[]

        for _ in 1:sweeps
            localMove!(sys, path, moves=spacing)
            push!(prim_vals, total_energy(sys, path))
#            push!(vir_vals, energy_virial(sys, path)) # TODO: fix this
        end

#        E_virial = sum(vir_vals) / length(vir_vals)
        E_primitive = sum(prim_vals) / length(prim_vals)

        # Exact D-D QHO energy
        E_exact = D * 0.5 * coth(sys.β/2)
        N=1
        E_equipartition = D * 0.5 * N / sys.β

        println(sys)
        println("E_primitive = $E_primitive, E_exact = $E_exact, E_equipartition = $E_equipartition ")
        @test isapprox(E_primitive, E_exact; rtol=0.05)
    end
end

 
