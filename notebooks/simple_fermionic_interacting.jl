### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ c96a6ec8-fd82-4f77-83eb-8f2f31e8fb22
using Random, LinearAlgebra, Statistics

# ╔═╡ d274124c-a3ab-4339-949f-723f05706582
using ForwardDiff: gradient

# ╔═╡ 31ca10cf-a1ca-409c-93bc-91735939334f
using UnicodePlots

# ╔═╡ 5076bffe-7a6a-4c09-a140-7b6c01b6f121
begin
	# Define the constants
	const β = 1.0     # inverse temperature
	const L = 1.0     # length of the box
	const N = 10      # number of particles
	const m = 1.0     # particle m
	const q = 1.0     # particle charge
	const ε0 = 1.0    # vacuum permittivity
	const e2 = q^2 / (4π * ε0)  # Coulomb constant
	const dτ = 0.1    # imaginary time step
	const dx = 0.05   # position step
	const Nτ = Int(β/dτ)  # number of imaginary time slices

	const delta = 0.01 # size of displacement / change in momentum move
end

# ╔═╡ 6d68fd3e-4377-453d-a8d7-5a458f3304f6
Nτ

# ╔═╡ 1d80c4f5-0d5a-42e1-ba21-8e51d0d89b1f
# Define the potential energy function
function V(x)
    if any(x .< 0) || any(x .> L)
        return Inf
    else
        return 0.0
    end
end

# ╔═╡ ec430cd7-6676-407d-b6dd-7ac1d6f43bfd
# Define the Coulomb interaction energy function
function U(x1, x2)
    r = norm(x1 - x2)
    if r == 0
        return Inf
    else
        return e2 / r
    end
end

# ╔═╡ 9d1d682e-8d12-42d5-9b8b-cbd8e3b4c8b5
# Define the kinetic energy function
KE(p) = sum(p.^2) / (2m)

# ╔═╡ 8ea44d99-2b31-4556-bf82-cac09adb9cee
# Define the action function
function action(x, p)
    V(x) * β / Nτ + 
	KE(p) * dτ / 2 +
    sum([U(x[i,:], x[j,:]) for i in 1:N for j in 1:i-1]) * β / Nτ
end

# ╔═╡ 6c31ad77-ba6a-4811-98ae-31efa56b3f56
# Define a type for the particle configuration
mutable struct ParticleConfig
    x::Matrix{Float64}  # N x 3 matrix of particle positions
    p::Matrix{Float64}  # N x 3 matrix of particle momenta
end

# ╔═╡ e6e04a87-1e9d-478f-a864-5b4073b61071
# Define a function to update the particle configuration
function SteepestDescentAutoDiff_update!(config::ParticleConfig)
    # Update the momenta
    config.p = randn(N, 3) * sqrt(m / (β * Nτ * dτ))

    # Update the positions
    for τ in 1:Nτ
        config.p[:, :] -= dτ / 2 * gradient(x -> action(x, config.p), config.x)  # half-step momentum update
        config.x[:, :] += dx * config.p[:, :] / m                                 # full-step position update
        config.p[:, :] -= dτ / 2 * gradient(x -> action(x, config.p), config.x)  # half-step momentum update
    end
end

# ╔═╡ c1de2974-1870-4c91-9f59-d163e68701b6
function MetropolisHastings_update!(config::ParticleConfig)

	old_action = action(config.x, config.p)
    
	# Propose a new configuration
	# Generate a random displacement vector for each particle
    delta_r = delta * randn(N, 3)
    config.x[:, :] += delta_r
	delta_p = delta * randn(N, 3)
    config.p[:, :] += delta_p
	
    # Compute the acceptance probability
    new_action = action(config.x, config.p)
    acceptance_prob = exp(-β* (new_action - old_action))

    # Decide whether to accept the new configuration
    if rand() > acceptance_prob
        # Return config to old state; move not accepted
        config.x[:, :] -= delta_r
		config.p[:, :] -= delta_p
    end
end


# ╔═╡ 8b0aa768-d9ec-4c0b-92e6-25782248ad49


# ╔═╡ 1aecabcb-4ec6-4837-b430-32b0aa7b1a10
function update!(config::ParticleConfig)
	MetropolisHastings_update!(config)
end

# ╔═╡ bbabf971-02f5-4ddc-aecc-a40a4708582b
# Define a function to initialize the particle configuration
function init_config()
    x = rand(N, 3) * L
    p = randn(N, 3) * sqrt(m / (β * Nτ * dτ))
    ParticleConfig(x, p)
end

# ╔═╡ da721ab7-8fcf-4800-87ac-171f22f807ba
# Define a function to compute the observable of interest
function potential_energy(config::ParticleConfig)
    sum([V(config.x[i,:]) for i in 1:N] + [KE(config.p[i,:]) for i in 1:N]) / N
end

# ╔═╡ eea2632a-0d89-415d-b76b-2fe64241c88e
# Perform the simulation
Random.seed!(1234)

# ╔═╡ d254124e-e1af-48c5-8c0e-7b36d2254706
# Define a function to run the PIMC simulation
function run_pimc(nsteps::Int)
    # Initialize the particle configuration
    config = init_config()

    # Run the PIMC simulation
    energies = zeros(nsteps)
	
    for step in 1:nsteps
        # Update the particle configuration
        update!(config)

        # Compute the observable of interest
        energy = potential_energy(config)
        energies[step] = energy
		println(config.x)
    end

    return energies, config
end


# ╔═╡ 8d633b90-e1e5-404b-802f-da5cefbbadc6
energy,finalconfig=run_pimc(1000)

# ╔═╡ 41a4c881-0f87-41e4-8ecb-2a1f8ff33894
# Print the final result
println("Average  energy: ", mean(energy))

# ╔═╡ 6849412c-ce59-456f-a27b-ad96abbaa0ae
plot(finalconfig.x)

# ╔═╡ 27836834-c564-4823-82fc-2af52051ea33
UnicodePlots.scatterplot(finalconfig.x[:,1], finalconfig.x[:,2])	

# ╔═╡ 00000000-0000-0000-0000-000000000001
PLUTO_PROJECT_TOML_CONTENTS = """
[deps]
ForwardDiff = "f6369f11-7733-5829-9624-2563aa707210"
LinearAlgebra = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"
Random = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"
Statistics = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
UnicodePlots = "b8865327-cd53-5732-bb35-84acbb429228"

[compat]
ForwardDiff = "~0.10.35"
UnicodePlots = "~3.4.0"
"""

# ╔═╡ 00000000-0000-0000-0000-000000000002
PLUTO_MANIFEST_TOML_CONTENTS = """
# This file is machine-generated - editing it directly is not advised

julia_version = "1.9.0"
manifest_format = "2.0"
project_hash = "c82d1877fded7c392f5073ee216186c952604d0e"

[[deps.ArgTools]]
uuid = "0dad84c5-d112-42e6-8d28-ef12dabb789f"
version = "1.1.1"

[[deps.Artifacts]]
uuid = "56f22d72-fd6d-98f1-02f0-08ddc0907c33"

[[deps.Base64]]
uuid = "2a0f44e3-6c83-55bd-87e4-b1978d98bd5f"

[[deps.ColorSchemes]]
deps = ["ColorTypes", "ColorVectorSpace", "Colors", "FixedPointNumbers", "Random", "SnoopPrecompile"]
git-tree-sha1 = "aa3edc8f8dea6cbfa176ee12f7c2fc82f0608ed3"
uuid = "35d6a980-a343-548e-a6ea-1d62b119f2f4"
version = "3.20.0"

[[deps.ColorTypes]]
deps = ["FixedPointNumbers", "Random"]
git-tree-sha1 = "eb7f0f8307f71fac7c606984ea5fb2817275d6e4"
uuid = "3da002f7-5984-5a60-b8a6-cbb66c0b333f"
version = "0.11.4"

[[deps.ColorVectorSpace]]
deps = ["ColorTypes", "FixedPointNumbers", "LinearAlgebra", "SpecialFunctions", "Statistics", "TensorCore"]
git-tree-sha1 = "600cc5508d66b78aae350f7accdb58763ac18589"
uuid = "c3611d14-8923-5661-9e6a-0046d554d3a4"
version = "0.9.10"

[[deps.Colors]]
deps = ["ColorTypes", "FixedPointNumbers", "Reexport"]
git-tree-sha1 = "fc08e5930ee9a4e03f84bfb5211cb54e7769758a"
uuid = "5ae59095-9a9b-59fe-a467-6f913c188581"
version = "0.12.10"

[[deps.CommonSubexpressions]]
deps = ["MacroTools", "Test"]
git-tree-sha1 = "7b8a93dba8af7e3b42fecabf646260105ac373f7"
uuid = "bbf7d656-a473-5ed7-a52c-81e309532950"
version = "0.3.0"

[[deps.Compat]]
deps = ["Dates", "LinearAlgebra", "UUIDs"]
git-tree-sha1 = "61fdd77467a5c3ad071ef8277ac6bd6af7dd4c04"
uuid = "34da2185-b29b-5c13-b0c7-acf172513d20"
version = "4.6.0"

[[deps.CompilerSupportLibraries_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "e66e0078-7015-5450-92f7-15fbd957f2ae"
version = "1.0.2+0"

[[deps.Contour]]
git-tree-sha1 = "d05d9e7b7aedff4e5b51a029dced05cfb6125781"
uuid = "d38c429a-6771-53c6-b99e-75d170b6e991"
version = "0.6.2"

[[deps.Crayons]]
git-tree-sha1 = "249fe38abf76d48563e2f4556bebd215aa317e15"
uuid = "a8cc5b0e-0ffa-5ad4-8c14-923d3ee1735f"
version = "4.1.1"

[[deps.DataAPI]]
git-tree-sha1 = "e8119c1a33d267e16108be441a287a6981ba1630"
uuid = "9a962f9c-6df0-11e9-0e5d-c546b8b5ee8a"
version = "1.14.0"

[[deps.DataStructures]]
deps = ["Compat", "InteractiveUtils", "OrderedCollections"]
git-tree-sha1 = "d1fff3a548102f48987a52a2e0d114fa97d730f0"
uuid = "864edb3b-99cc-5e75-8d2d-829cb0a9cfe8"
version = "0.18.13"

[[deps.Dates]]
deps = ["Printf"]
uuid = "ade2ca70-3891-5945-98fb-dc099432e06a"

[[deps.DiffResults]]
deps = ["StaticArraysCore"]
git-tree-sha1 = "782dd5f4561f5d267313f23853baaaa4c52ea621"
uuid = "163ba53b-c6d8-5494-b064-1a9d43ac40c5"
version = "1.1.0"

[[deps.DiffRules]]
deps = ["IrrationalConstants", "LogExpFunctions", "NaNMath", "Random", "SpecialFunctions"]
git-tree-sha1 = "a4ad7ef19d2cdc2eff57abbbe68032b1cd0bd8f8"
uuid = "b552c78f-8df3-52c6-915a-8e097449b14b"
version = "1.13.0"

[[deps.DocStringExtensions]]
deps = ["LibGit2"]
git-tree-sha1 = "2fb1e02f2b635d0845df5d7c167fec4dd739b00d"
uuid = "ffbed154-4ef7-542d-bbb7-c09d3a79fcae"
version = "0.9.3"

[[deps.Downloads]]
deps = ["ArgTools", "FileWatching", "LibCURL", "NetworkOptions"]
uuid = "f43a241f-c20a-4ad4-852c-f6b1247861c6"
version = "1.6.0"

[[deps.FileWatching]]
uuid = "7b1f6079-737a-58dc-b8bc-7a2ca5c1b5ee"

[[deps.FixedPointNumbers]]
deps = ["Statistics"]
git-tree-sha1 = "335bfdceacc84c5cdf16aadc768aa5ddfc5383cc"
uuid = "53c48c17-4a7d-5ca2-90c5-79b7896eea93"
version = "0.8.4"

[[deps.ForwardDiff]]
deps = ["CommonSubexpressions", "DiffResults", "DiffRules", "LinearAlgebra", "LogExpFunctions", "NaNMath", "Preferences", "Printf", "Random", "SpecialFunctions"]
git-tree-sha1 = "00e252f4d706b3d55a8863432e742bf5717b498d"
uuid = "f6369f11-7733-5829-9624-2563aa707210"
version = "0.10.35"
weakdeps = ["StaticArrays"]

    [deps.ForwardDiff.extensions]
    ForwardDiffStaticArraysExt = "StaticArrays"

[[deps.InteractiveUtils]]
deps = ["Markdown"]
uuid = "b77e0a4c-d291-57a0-90e8-8db25a27a240"

[[deps.IrrationalConstants]]
git-tree-sha1 = "630b497eafcc20001bba38a4651b327dcfc491d2"
uuid = "92d709cd-6900-40b7-9082-c6be49f344b6"
version = "0.2.2"

[[deps.JLLWrappers]]
deps = ["Preferences"]
git-tree-sha1 = "abc9885a7ca2052a736a600f7fa66209f96506e1"
uuid = "692b3bcd-3c85-4b1f-b108-f13ce0eb3210"
version = "1.4.1"

[[deps.LibCURL]]
deps = ["LibCURL_jll", "MozillaCACerts_jll"]
uuid = "b27032c2-a3e7-50c8-80cd-2d36dbcbfd21"
version = "0.6.3"

[[deps.LibCURL_jll]]
deps = ["Artifacts", "LibSSH2_jll", "Libdl", "MbedTLS_jll", "Zlib_jll", "nghttp2_jll"]
uuid = "deac9b47-8bc7-5906-a0fe-35ac56dc84c0"
version = "7.84.0+0"

[[deps.LibGit2]]
deps = ["Base64", "NetworkOptions", "Printf", "SHA"]
uuid = "76f85450-5226-5b5a-8eaa-529ad045b433"

[[deps.LibSSH2_jll]]
deps = ["Artifacts", "Libdl", "MbedTLS_jll"]
uuid = "29816b5a-b9ab-546f-933c-edad1886dfa8"
version = "1.10.2+0"

[[deps.Libdl]]
uuid = "8f399da3-3557-5675-b5ff-fb832c97cbdb"

[[deps.LinearAlgebra]]
deps = ["Libdl", "OpenBLAS_jll", "libblastrampoline_jll"]
uuid = "37e2e46d-f89d-539d-b4ee-838fcccc9c8e"

[[deps.LogExpFunctions]]
deps = ["DocStringExtensions", "IrrationalConstants", "LinearAlgebra"]
git-tree-sha1 = "0a1b7c2863e44523180fdb3146534e265a91870b"
uuid = "2ab3a3ac-af41-5b50-aa03-7779005ae688"
version = "0.3.23"

    [deps.LogExpFunctions.extensions]
    LogExpFunctionsChainRulesCoreExt = "ChainRulesCore"
    LogExpFunctionsChangesOfVariablesExt = "ChangesOfVariables"
    LogExpFunctionsInverseFunctionsExt = "InverseFunctions"

    [deps.LogExpFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"
    ChangesOfVariables = "9e997f8a-9a97-42d5-a9f1-ce6bfc15e2c0"
    InverseFunctions = "3587e190-3f89-42d0-90ee-14403ec27112"

[[deps.Logging]]
uuid = "56ddb016-857b-54e1-b83d-db4d58db5568"

[[deps.MacroTools]]
deps = ["Markdown", "Random"]
git-tree-sha1 = "42324d08725e200c23d4dfb549e0d5d89dede2d2"
uuid = "1914dd2f-81c6-5fcd-8719-6d5c9610ff09"
version = "0.5.10"

[[deps.MarchingCubes]]
deps = ["SnoopPrecompile", "StaticArrays"]
git-tree-sha1 = "55aaf3fdf414b691a15875cfe5edb6e0daf4625a"
uuid = "299715c1-40a9-479a-aaf9-4a633d36f717"
version = "0.1.6"

[[deps.Markdown]]
deps = ["Base64"]
uuid = "d6f4376e-aef5-505a-96c1-9c027394607a"

[[deps.MbedTLS_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "c8ffd9c3-330d-5841-b78e-0817d7145fa1"
version = "2.28.2+0"

[[deps.Missings]]
deps = ["DataAPI"]
git-tree-sha1 = "f66bdc5de519e8f8ae43bdc598782d35a25b1272"
uuid = "e1d29d7a-bbdc-5cf2-9ac0-f12de2c33e28"
version = "1.1.0"

[[deps.MozillaCACerts_jll]]
uuid = "14a3606d-f60d-562e-9121-12d972cd8159"
version = "2022.10.11"

[[deps.NaNMath]]
deps = ["OpenLibm_jll"]
git-tree-sha1 = "0877504529a3e5c3343c6f8b4c0381e57e4387e4"
uuid = "77ba4419-2d1f-58cd-9bb1-8ffee604a2e3"
version = "1.0.2"

[[deps.NetworkOptions]]
uuid = "ca575930-c2e3-43a9-ace4-1e988b2c1908"
version = "1.2.0"

[[deps.OpenBLAS_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "Libdl"]
uuid = "4536629a-c528-5b80-bd46-f80d51c5b363"
version = "0.3.21+4"

[[deps.OpenLibm_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "05823500-19ac-5b8b-9628-191a04bc5112"
version = "0.8.1+0"

[[deps.OpenSpecFun_jll]]
deps = ["Artifacts", "CompilerSupportLibraries_jll", "JLLWrappers", "Libdl", "Pkg"]
git-tree-sha1 = "13652491f6856acfd2db29360e1bbcd4565d04f1"
uuid = "efe28fd5-8261-553b-a9e1-b2916fc3738e"
version = "0.5.5+0"

[[deps.OrderedCollections]]
git-tree-sha1 = "85f8e6578bf1f9ee0d11e7bb1b1456435479d47c"
uuid = "bac558e1-5e72-5ebc-8fee-abe8a469f55d"
version = "1.4.1"

[[deps.Pkg]]
deps = ["Artifacts", "Dates", "Downloads", "FileWatching", "LibGit2", "Libdl", "Logging", "Markdown", "Printf", "REPL", "Random", "SHA", "Serialization", "TOML", "Tar", "UUIDs", "p7zip_jll"]
uuid = "44cfe95a-1eb2-52ea-b672-e2afdf69b78f"
version = "1.9.0"

[[deps.Preferences]]
deps = ["TOML"]
git-tree-sha1 = "47e5f437cc0e7ef2ce8406ce1e7e24d44915f88d"
uuid = "21216c6a-2e73-6563-6e65-726566657250"
version = "1.3.0"

[[deps.Printf]]
deps = ["Unicode"]
uuid = "de0858da-6303-5e67-8744-51eddeeeb8d7"

[[deps.REPL]]
deps = ["InteractiveUtils", "Markdown", "Sockets", "Unicode"]
uuid = "3fa0cd96-eef1-5676-8a61-b3b8758bbffb"

[[deps.Random]]
deps = ["SHA", "Serialization"]
uuid = "9a3f8284-a2c9-5f02-9a11-845980a1fd5c"

[[deps.Reexport]]
git-tree-sha1 = "45e428421666073eab6f2da5c9d310d99bb12f9b"
uuid = "189a3867-3050-52da-a836-e630ba90ab69"
version = "1.2.2"

[[deps.Requires]]
deps = ["UUIDs"]
git-tree-sha1 = "838a3a4188e2ded87a4f9f184b4b0d78a1e91cb7"
uuid = "ae029012-a4dd-5104-9daa-d747884805df"
version = "1.3.0"

[[deps.SHA]]
uuid = "ea8e919c-243c-51af-8825-aaa63cd721ce"
version = "0.7.0"

[[deps.Serialization]]
uuid = "9e88b42a-f829-5b0c-bbe9-9e923198166b"

[[deps.SnoopPrecompile]]
deps = ["Preferences"]
git-tree-sha1 = "e760a70afdcd461cf01a575947738d359234665c"
uuid = "66db9d55-30c0-4569-8b51-7e840670fc0c"
version = "1.0.3"

[[deps.Sockets]]
uuid = "6462fe0b-24de-5631-8697-dd941f90decc"

[[deps.SortingAlgorithms]]
deps = ["DataStructures"]
git-tree-sha1 = "a4ada03f999bd01b3a25dcaa30b2d929fe537e00"
uuid = "a2af1166-a08f-5f64-846c-94a0d3cef48c"
version = "1.1.0"

[[deps.SparseArrays]]
deps = ["Libdl", "LinearAlgebra", "Random", "Serialization", "SuiteSparse_jll"]
uuid = "2f01184e-e22b-5df5-ae63-d93ebab69eaf"

[[deps.SpecialFunctions]]
deps = ["IrrationalConstants", "LogExpFunctions", "OpenLibm_jll", "OpenSpecFun_jll"]
git-tree-sha1 = "ef28127915f4229c971eb43f3fc075dd3fe91880"
uuid = "276daf66-3868-5448-9aa4-cd146d93841b"
version = "2.2.0"

    [deps.SpecialFunctions.extensions]
    SpecialFunctionsChainRulesCoreExt = "ChainRulesCore"

    [deps.SpecialFunctions.weakdeps]
    ChainRulesCore = "d360d2e6-b24c-11e9-a2a3-2a2ae2dbcce4"

[[deps.StaticArrays]]
deps = ["LinearAlgebra", "Random", "StaticArraysCore", "Statistics"]
git-tree-sha1 = "2d7d9e1ddadc8407ffd460e24218e37ef52dd9a3"
uuid = "90137ffa-7385-5640-81b9-e52037218182"
version = "1.5.16"

[[deps.StaticArraysCore]]
git-tree-sha1 = "6b7ba252635a5eff6a0b0664a41ee140a1c9e72a"
uuid = "1e83bf80-4336-4d27-bf5d-d5a4f845583c"
version = "1.4.0"

[[deps.Statistics]]
deps = ["LinearAlgebra", "SparseArrays"]
uuid = "10745b16-79ce-11e8-11f9-7d13ad32a3b2"
version = "1.9.0"

[[deps.StatsAPI]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "f9af7f195fb13589dd2e2d57fdb401717d2eb1f6"
uuid = "82ae8749-77ed-4fe6-ae5f-f523153014b0"
version = "1.5.0"

[[deps.StatsBase]]
deps = ["DataAPI", "DataStructures", "LinearAlgebra", "LogExpFunctions", "Missings", "Printf", "Random", "SortingAlgorithms", "SparseArrays", "Statistics", "StatsAPI"]
git-tree-sha1 = "d1bf48bfcc554a3761a133fe3a9bb01488e06916"
uuid = "2913bbd2-ae8a-5f71-8c99-4fb6c76f3a91"
version = "0.33.21"

[[deps.SuiteSparse_jll]]
deps = ["Artifacts", "Libdl", "Pkg", "libblastrampoline_jll"]
uuid = "bea87d4a-7f5b-5778-9afe-8cc45184846c"
version = "5.10.1+6"

[[deps.TOML]]
deps = ["Dates"]
uuid = "fa267f1f-6049-4f14-aa54-33bafae1ed76"
version = "1.0.3"

[[deps.Tar]]
deps = ["ArgTools", "SHA"]
uuid = "a4e569a6-e804-4fa4-b0f3-eef7a1d5b13e"
version = "1.10.0"

[[deps.TensorCore]]
deps = ["LinearAlgebra"]
git-tree-sha1 = "1feb45f88d133a655e001435632f019a9a1bcdb6"
uuid = "62fd8b95-f654-4bbd-a8a5-9c27f68ccd50"
version = "0.1.1"

[[deps.Test]]
deps = ["InteractiveUtils", "Logging", "Random", "Serialization"]
uuid = "8dfed614-e22c-5e08-85e1-65c5234f0b40"

[[deps.UUIDs]]
deps = ["Random", "SHA"]
uuid = "cf7118a7-6976-5b1a-9a39-7adc72f591a4"

[[deps.Unicode]]
uuid = "4ec0a83e-493e-50e2-b9ac-8f72acf5a8f5"

[[deps.UnicodePlots]]
deps = ["ColorSchemes", "ColorTypes", "Contour", "Crayons", "Dates", "LinearAlgebra", "MarchingCubes", "NaNMath", "Printf", "Requires", "SnoopPrecompile", "SparseArrays", "StaticArrays", "StatsBase"]
git-tree-sha1 = "ef00b38d086414a54d679d81ced90fb7b0f03909"
uuid = "b8865327-cd53-5732-bb35-84acbb429228"
version = "3.4.0"

    [deps.UnicodePlots.extensions]
    FreeTypeExt = ["FileIO", "FreeType"]
    ImageInTerminalExt = "ImageInTerminal"
    UnitfulExt = "Unitful"

    [deps.UnicodePlots.weakdeps]
    FileIO = "5789e2e9-d7fb-5bc7-8068-2c6fae9b9549"
    FreeType = "b38be410-82b0-50bf-ab77-7b57e271db43"
    ImageInTerminal = "d8c32880-2388-543b-8c61-d9f865259254"
    Unitful = "1986cc42-f94f-5a68-af5c-568840ba703d"

[[deps.Zlib_jll]]
deps = ["Libdl"]
uuid = "83775a58-1f1d-513f-b197-d71354ab007a"
version = "1.2.13+0"

[[deps.libblastrampoline_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850b90-86db-534c-a0d3-1478176c7d93"
version = "5.7.0+0"

[[deps.nghttp2_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "8e850ede-7688-5339-a07c-302acd2aaf8d"
version = "1.48.0+0"

[[deps.p7zip_jll]]
deps = ["Artifacts", "Libdl"]
uuid = "3f19e933-33d8-53b3-aaab-bd5110c3b7a0"
version = "17.4.0+0"
"""

# ╔═╡ Cell order:
# ╠═c96a6ec8-fd82-4f77-83eb-8f2f31e8fb22
# ╠═d274124c-a3ab-4339-949f-723f05706582
# ╠═5076bffe-7a6a-4c09-a140-7b6c01b6f121
# ╠═6d68fd3e-4377-453d-a8d7-5a458f3304f6
# ╠═1d80c4f5-0d5a-42e1-ba21-8e51d0d89b1f
# ╠═ec430cd7-6676-407d-b6dd-7ac1d6f43bfd
# ╠═9d1d682e-8d12-42d5-9b8b-cbd8e3b4c8b5
# ╠═8ea44d99-2b31-4556-bf82-cac09adb9cee
# ╠═6c31ad77-ba6a-4811-98ae-31efa56b3f56
# ╠═e6e04a87-1e9d-478f-a864-5b4073b61071
# ╠═c1de2974-1870-4c91-9f59-d163e68701b6
# ╠═8b0aa768-d9ec-4c0b-92e6-25782248ad49
# ╠═1aecabcb-4ec6-4837-b430-32b0aa7b1a10
# ╠═bbabf971-02f5-4ddc-aecc-a40a4708582b
# ╠═da721ab7-8fcf-4800-87ac-171f22f807ba
# ╠═eea2632a-0d89-415d-b76b-2fe64241c88e
# ╠═d254124e-e1af-48c5-8c0e-7b36d2254706
# ╠═8d633b90-e1e5-404b-802f-da5cefbbadc6
# ╠═41a4c881-0f87-41e4-8ecb-2a1f8ff33894
# ╠═31ca10cf-a1ca-409c-93bc-91735939334f
# ╠═6849412c-ce59-456f-a27b-ad96abbaa0ae
# ╠═27836834-c564-4823-82fc-2af52051ea33
# ╟─00000000-0000-0000-0000-000000000001
# ╟─00000000-0000-0000-0000-000000000002
