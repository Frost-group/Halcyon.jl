### A Pluto.jl notebook ###
# v0.19.46

using Markdown
using InteractiveUtils

# ╔═╡ cf33684e-7ff3-11ef-1222-113dfe69a52e
begin
    import Pkg
    # activate the shared project environment
    Pkg.activate(Base.current_project())
    # instantiate, i.e. make sure that all packages are downloaded
    Pkg.instantiate()
    using  LinearAlgebra
end


# ╔═╡ 0c480bfd-fd2a-472c-88b4-55e3b7bf555e
using Halcyon

# ╔═╡ bf1f3e90-6307-41cb-8301-52ce793295b5
using StaticArrays

# ╔═╡ aa117241-6058-4c37-92ea-ed7098a38bf0
using Gnuplot

# ╔═╡ e7a57d59-47e6-4f0e-aa7c-f1f81ebcf82b
system=Halcyon.System(200,10)

# ╔═╡ 09f5f99e-69e8-417b-81c4-0b0cddd25e39
path=Halcyon.Path(system)

# ╔═╡ 467dc73d-5175-42a8-a4f4-7409266596a0
Halcyon.localMove!(system,path,verbose=true)

# ╔═╡ 3a76a88d-2697-43c1-a5f9-c324acc652f7
@time Halcyon.localMove!(system,path)

# ╔═╡ 3dea0870-c89f-4e1f-9406-f6ba6db79f04
println("Now 1_000_000 moves...")

# ╔═╡ fded6861-dfd9-44d2-84ca-b4b95ef563fb
@time Halcyon.localMove!(system,path,moves=1_000_000)

# ╔═╡ 62cafe35-e2ca-4692-bac7-b69f120934c0
Halcyon.localMove!(system,path,verbose=true)

# ╔═╡ b02f75ea-71bb-429f-bfeb-a8c2b4acfad6
println("Rattle rattle... should be lower in energy.")

# ╔═╡ 01b8fd34-df2d-413c-9650-bbbef34a28b0
path

# ╔═╡ d18addfd-82d2-4f1d-9a1e-4643433ec77a
system

# ╔═╡ 0b7188b2-2f84-4f98-94df-d8bb801faefe
function localMove!(sys,path; moves=1, verbose=false, stepsize=0.01, V=Harmonic, U=Coulomb)
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

        Sold = V(pold) +
        sum(U.(reinterpret(SVector{3, Float64}, (pold' .- @view path.r[p,:,:])'))) +
            sum(abs2, (pold .- @view path.r[path.prev[p, b], mod1(b- 1, sys.nbeads), :])) +
            sum(abs2, (pold .- @view path.r[path.next[p, b], mod1(b+ 1, sys.nbeads), :]))

        Snew = V(pnew) +
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

# ╔═╡ ec95e935-5838-45c5-b1d1-d1ab5163673b
localMove!(system,path,verbose=true)

# ╔═╡ 9fd2eed4-7c9d-4491-8d1c-cf0345b17dbc
@time Halcyon.localMove!(system,path,moves=1_000_000)

# ╔═╡ fe17ea8d-d27e-4b96-9782-8925fcd1567b
pold=@view path.r[1,1,:]

# ╔═╡ 0cf56308-16dc-45f8-b27f-8fa0ae23d933
sum(Coulomb.(reinterpret(SVector{3, Float64},(pold' .- @view path.r[1,:,:])')))

# ╔═╡ b64f5bf2-60e1-4a7a-b699-0e3a8441470e
path.r

# ╔═╡ 2600b1a8-63a4-47ef-a48d-ef99df38a706
@gp path.r[:,:,1] path.r[:,:,2]  "w lp"

# ╔═╡ ea7a4e94-b4bd-44f4-9796-bb1e516ed328
null(x)=Coulomb(x, g=10)

# ╔═╡ 471ea4a0-9aac-4b77-baec-7526fc11b26f
@time localMove!(system,path,moves=1_000_000, U=null)

# ╔═╡ 10654cc8-21ff-4cd6-ad49-f8fa0abd01c6
@gp path.r[:,:,1] path.r[:,:,2]  "w lp"

# ╔═╡ a8f90f68-3321-48c1-9949-79531edba647
begin
	# histo of particle posn across x,y,z
	@gp    hist(path.r[:,:,1] |> vec, nbins=25) #"lc rgb 'red'"
	@gp :- hist(path.r[:,:,2] |> vec, nbins=25) #"lc rgb 'blue'"
	@gp :- hist(path.r[:,:,3] |> vec, nbins=25) #"lc rgb 'green'"
end

# ╔═╡ c4e8bdea-23e6-427c-8585-92b1f66b4a1a
@gp hist(path.r[:,:,1] |> vec, path.r[:,:,2] |> vec, nbins1=20, nbins2=20)

# ╔═╡ d8364f2c-e04f-44bb-81f0-f8341f37411c


# ╔═╡ 9aa3126f-58b8-478d-81fa-6223f40211f4


# ╔═╡ b2690a76-8a2c-41d5-99b8-4fb1b086bf12


# ╔═╡ f5fa0f46-3b81-4faa-9da8-2938432b45f5


# ╔═╡ df07a2e3-4205-4451-8541-88067d360377


# ╔═╡ a545f8e4-8fe2-40a8-8aa9-bfaae57587d0


# ╔═╡ 0d3a8d18-93f3-42a4-9d5f-1506143f9810


# ╔═╡ 049923da-5cc9-4f89-a84c-74c0628c0328


# ╔═╡ bca8dbe4-ccc9-405f-8ca3-620dcc3cffb6
begin
	track=[]
	for i in 1:10
		localMove!(system,path,moves=100_000, U=null)
	   append!(track,[path.r])
	end
end

# ╔═╡ d2482769-bd0f-4293-b78c-8824c12a1c07
track

# ╔═╡ 761faced-6d9d-4e0c-aa09-4a92bffd3283
@time Halcyon.localMove!(system,path,moves=1_000_000, U=Coulomb(x;g=1000000))

# ╔═╡ d5edf04c-075a-4f02-a3d2-6f8e15c11471


# ╔═╡ Cell order:
# ╠═cf33684e-7ff3-11ef-1222-113dfe69a52e
# ╠═0c480bfd-fd2a-472c-88b4-55e3b7bf555e
# ╠═e7a57d59-47e6-4f0e-aa7c-f1f81ebcf82b
# ╠═09f5f99e-69e8-417b-81c4-0b0cddd25e39
# ╠═467dc73d-5175-42a8-a4f4-7409266596a0
# ╠═3a76a88d-2697-43c1-a5f9-c324acc652f7
# ╠═3dea0870-c89f-4e1f-9406-f6ba6db79f04
# ╠═fded6861-dfd9-44d2-84ca-b4b95ef563fb
# ╠═62cafe35-e2ca-4692-bac7-b69f120934c0
# ╠═b02f75ea-71bb-429f-bfeb-a8c2b4acfad6
# ╠═01b8fd34-df2d-413c-9650-bbbef34a28b0
# ╠═d18addfd-82d2-4f1d-9a1e-4643433ec77a
# ╠═0b7188b2-2f84-4f98-94df-d8bb801faefe
# ╠═ec95e935-5838-45c5-b1d1-d1ab5163673b
# ╠═9fd2eed4-7c9d-4491-8d1c-cf0345b17dbc
# ╠═fe17ea8d-d27e-4b96-9782-8925fcd1567b
# ╠═bf1f3e90-6307-41cb-8301-52ce793295b5
# ╠═0cf56308-16dc-45f8-b27f-8fa0ae23d933
# ╠═b64f5bf2-60e1-4a7a-b699-0e3a8441470e
# ╠═aa117241-6058-4c37-92ea-ed7098a38bf0
# ╠═2600b1a8-63a4-47ef-a48d-ef99df38a706
# ╠═ea7a4e94-b4bd-44f4-9796-bb1e516ed328
# ╠═471ea4a0-9aac-4b77-baec-7526fc11b26f
# ╠═10654cc8-21ff-4cd6-ad49-f8fa0abd01c6
# ╠═a8f90f68-3321-48c1-9949-79531edba647
# ╠═c4e8bdea-23e6-427c-8585-92b1f66b4a1a
# ╠═d8364f2c-e04f-44bb-81f0-f8341f37411c
# ╠═9aa3126f-58b8-478d-81fa-6223f40211f4
# ╠═b2690a76-8a2c-41d5-99b8-4fb1b086bf12
# ╠═f5fa0f46-3b81-4faa-9da8-2938432b45f5
# ╠═df07a2e3-4205-4451-8541-88067d360377
# ╠═a545f8e4-8fe2-40a8-8aa9-bfaae57587d0
# ╠═0d3a8d18-93f3-42a4-9d5f-1506143f9810
# ╠═049923da-5cc9-4f89-a84c-74c0628c0328
# ╠═bca8dbe4-ccc9-405f-8ca3-620dcc3cffb6
# ╠═d2482769-bd0f-4293-b78c-8824c12a1c07
# ╠═761faced-6d9d-4e0c-aa09-4a92bffd3283
# ╠═d5edf04c-075a-4f02-a3d2-6f8e15c11471
