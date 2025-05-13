### A Pluto.jl notebook ###
# v0.19.22

using Markdown
using InteractiveUtils

# ╔═╡ 18fc3cb1-77d1-49e0-97d0-d53dabe691e7
using LinearAlgebra

# ╔═╡ 7a704a36-a05c-4248-bad0-04ed7d0a745d
using Random

# ╔═╡ 65eda18b-5db7-4c1b-ad7f-f1f6faa71c8a
using StatsBase

# ╔═╡ 5cbc3649-9784-48a0-9598-744fa9f33279
using Zygote

# ╔═╡ 1a0f12ad-86c3-4abd-a038-8ddc021e934e
using BenchmarkTools

# ╔═╡ b73556c6-e692-4eca-8c80-75d80ecf231e
using Plots

# ╔═╡ b2abcf83-77a1-4f75-89d8-8f6eec2b8cba
using Gnuplot

# ╔═╡ c5c96c64-eff2-11ed-04e9-e1642f1e52ee
# Unreasonable Expectation of Zero Point Energy

# ╔═╡ 022fa92a-570d-45c6-b50c-10c13b741c4a
begin
	struct Params
	    Ne::Int  # number of electrons
	    Nt::Int  # number of time slices
	    β::Float64  # inverse temperature
	    ω::Float64  # frequency of harmonic potential
	    Δ::Float64  # step size for Metropolis updates
	    rₙ::Float64  # nodal radius
	    rₑ::Float64  # exchange radius
	end
	Params(;Ne=2, Nt=100, β=1.0, ω=1.0, Δ=0.1, rₙ=1.0,rₑ=1.0)=Params(Ne,Nt,β,ω,Δ,rₙ,rₑ)
end

# ╔═╡ add7f101-a827-4dc6-ab6c-93f00b4681d7
begin
	struct State
	    params::Params
	    path::Array{Float64,3}  # path[electron, dimension, time slice]
	end
	State(p::Params) = State(p, rand(p.Ne, 3, p.Nt))
end

# ╔═╡ fe4c179a-bbeb-46a8-8bc5-6fdca2f358f5
function V(r::Vector{Float64}, ω::Float64)
    0.5 * ω^2 * sum(r.^2)
end

# ╔═╡ df1e127b-7fc3-43ca-a05b-59e5381d75a2
function U(r₁::Vector{Float64}, r₂::Vector{Float64})
    1.0 / norm(r₁ - r₂)
end

# ╔═╡ 83f804d0-2d93-413b-9dab-e5bef166d502
function action(s::State, i::Int, j::Int, r::Vector{Float64})
    p = s.params
    T = (sum(s.path[i, :, (j)% p.Nt + 1] - r) .^ 2 
	     + sum((s.path[i, :, (j-2+p.Nt)% p.Nt + 1] - r) .^ 2 )) / 
		(2 * p.β / p.Nt)
    Vₑ = V(r, p.ω)
    Uₑ = sum(U(r, s.path[k, :, j]) for k in 1:p.Ne if k != i ; init=0)
    #Xₑ = sum((norm(s.path[k, :, j] - r) < p.rₑ && k != i) ? -1.0 : 0.0 for k in 1:p.Ne)
    T + Vₑ + Uₑ #+ Xₑ
end

# ╔═╡ 759916b1-a2c3-417e-a578-48162bdeba30
action_gradient(r::Vector{Float64}, s::State, i::Int, j::Int) = gradient(r -> action(r,s,i,j))

# ╔═╡ 8654b316-cf71-45d1-b2fd-9ceded5d2a5d
function centreofm_update!(s::State)
	p=s.params
	i=rand(1:p.Ne)

	rΔ=p.Δ * randn(3)

    Sₒ = sum( action(s, i, j, s.path[i, :, j]) for j in 1:p.Nt) 
	Sₙ = sum( action(s, i, j, rΔ+s.path[i, :, j]) for j in 1:p.Nt)
    ΔS = Sₙ - Sₒ
	
    if ΔS<0 || rand() < exp(-ΔS) #&& !crosses_nodal_surface(s, i, j, r)
        s.path[i, :, :] = s.path[i, :, :] .+ rΔ
    end
end

# ╔═╡ 756efc67-b4ce-431e-9f4d-14926260c63d
let
	ptest = Params(Ne=2, Nt=100, β=1.0, ω=1.0, Δ=0.1, rₙ=1.0, rₑ=1.0)
	stest = State(ptest)
	centreofm_update!(stest)
end

# ╔═╡ f749dd08-0f8c-423a-9c4a-79119ae5358d
function metropolis_update!(s::State)
    p = s.params
    i = rand(1:p.Ne)
    j = rand(1:p.Nt)
    r = s.path[i, :, j] + p.Δ * randn(3)
    Sₒ = action(s, i, j, s.path[i, :, j])
    Sₙ = action(s, i, j, r)
	ΔS = Sₙ - Sₒ
    if ΔS<0 || rand() < exp(-ΔS) #&& !crosses_nodal_surface(s, i, j, r)
        s.path[i, :, j] = r
    end
end

# ╔═╡ 1a68ff25-4d75-4c62-a01b-cea826d3ff9c
function bisection_update!(s::State, i::Int, τ_start::Int, τ_end::Int)
    p = s.params
    τ_mid = (τ_start + τ_end) ÷ 2
    if (τ_end - τ_start) > 1
        bisection_update!(s, i, τ_start, τ_mid)
        bisection_update!(s, i, τ_mid, τ_end)
    end

    r = (s.path[i, :, τ_start] + s.path[i, :, τ_end]) / 2 + sqrt((τ_end - τ_start) * p.β / p.Nt) * randn(3)
    S_old = action(s, i, τ_start, s.path[i, :, τ_start]) + action(s, i, τ_end, s.path[i, :, τ_end])
    S_new = action(s, i, τ_start, r) + action(s, i, τ_end, r)
    ΔS = S_new - S_old
    if ΔS < 0 || rand() < exp(-ΔS)
        s.path[i, :, τ_mid] .= r
    end
end

# ╔═╡ 255a9be8-bd63-44cd-9cc2-546b123b5127
function bisection_update!(s::State)
    p = s.params
    i = rand(1:p.Ne)
    τ_start = rand(1:p.Nt)
    τ_end = τ_start + 2^rand(0:log2(p.Nt-1))
    if τ_end > p.Nt
        τ_end = p.Nt
    end
    bisection_update!(s, i, τ_start, τ_end)
end

# ╔═╡ 28a2c299-89d6-47d3-8474-ba10c0531cb0
begin
	ptest = Params(Ne=2, Nt=100, β=1.0, ω=1.0, Δ=0.1, rₙ=1.0, rₑ=1.0)
	stest = State(ptest)
	bisection_update!(stest)
end

# ╔═╡ f1123350-7079-412f-9771-867598477957
function crosses_nodal_surface(s::State, i::Int, j::Int, r::Vector{Float64})
    p = s.params
    any(k -> (norm(s.path[k, :, j] - r) < p.rₙ && k != i), 1:p.Ne)
end

# ╔═╡ 6df6a3d8-a6bd-4baa-8f12-6ae18c52f751
function energy(s::State)
    p = s.params
    T = sum(sum(
		sum((s.path[i, :, j] - s.path[i, :, j% p.Nt + 1]) .^ 2) 
			for i in 1:p.Ne) 
				for j in 1:p.Nt) / (p.β / p.Nt)
    Vₑ = sum(sum(
		V(s.path[i, :, j], p.ω) 
			for i in 1:p.Ne) 
				for j in 1:p.Nt) / p.Nt
    Uₑ = sum(sum(sum(
			U(s.path[i, :, j], s.path[k, :, j]) 
			for k in 1:p.Ne if k > i ; init=0) 
				for i in 1:p.Ne ) 
					for j in 1:p.Nt ) / p.Nt
	T + Vₑ + Uₑ
end

# ╔═╡ 5727c703-246d-4b5d-b4b1-368719f299c9
function run_simulation(p::Params, moves::Dict{Function, Float64}, n_steps::Int, stride::Int)
    s = State(p)
    E_avg = energy(s)
	traj=[]
    for t in 1:n_steps
        sample([centreofm_update!, metropolis_update!, bisection_update!], Weights([moves[centreofm_update!], moves[metropolis_update!], moves[bisection_update!]]))(s)
        if t % stride == 0
            E_avg = (E_avg * (t / stride - 1) + energy(s)) / (t / stride)
			append!(traj, [s.path])
		end
    end
    E_avg, s, traj
end

# ╔═╡ 924dda10-f85a-42d6-962b-d9e92cf741de


# ╔═╡ 194f526c-7956-426f-aeac-938c0299c2c3
@benchmark metropolis_update!(State(Params(Ne=3, Nt=40, β=0.2, ω=1.0, Δ=0.01, rₙ=1.0, rₑ=1.0)))

# ╔═╡ 017020dc-933b-4991-aeeb-028285b19ac6
function plot_traj_Plotsjl(s,traj)
    p = s.params
	colors = [:blue, :red, :green, :purple, :orange] 
	Ncolors=length(colors)

	a=Animation()
	for t in traj
		plt=plot3d()
		for i in 1:p.Ne
        	x = append!(t[i,1,:], t[i,1,1]) # loop the paths
        	y = append!(t[i,2,:], t[i,2,1])
        	z = append!(t[i,3,:], t[i,3,1])

        	plot3d!(x, y, z,
				linewidth = 2, color = colors[i%Ncolors + 1],label = "Electron $i")
    	end
		frame(a,plt)
	end
	a
end

# ╔═╡ 9b1fcb8d-3ac7-43f2-ae62-20a8d1530db3
begin
	p = Params(Ne=1, Nt=10, β=1, ω=1.0, Δ=0.01, rₙ=1.0, rₑ=1.0)
	#moves = Dict(metropolis_update! => 0.5, bisection_update! => 0.5)
	moves = Dict(centreofm_update! => 0.1, metropolis_update! => 1.0, bisection_update! => 0.0)
	n_steps = 1_000_000
	stride = n_steps ÷ 1000
	
	E_avg, s, traj=run_simulation(p, moves, n_steps, stride)
	
	println("Average energy: $E_avg")
	#plot_paths_gnuplot(s)
	a=plot_traj_Plotsjl(s,traj)
	gif(a)
end

# ╔═╡ d5bb6b3a-ac23-4e95-9323-e4ca00725144
let
	ptest = Params(Ne=2, Nt=100, β=1.0, ω=1.0, Δ=0.1, rₙ=1.0, rₑ=1.0)
	stest = State(ptest)

    p = stest.params
    i = rand(1:p.Ne)
    j = rand(1:p.Nt)
    r = stest.path[i, :, j] + p.Δ * randn(3)

	action_gradient(r, s, i, j)
end

# ╔═╡ cfa86086-a635-4870-9092-578f7fab0383
function plot_paths_Plotsjl(s::State)
    p = s.params
    plot3d()
	colors = [:blue, :red, :green, :purple, :orange] 
	Ncolors=length(colors)
    for i in 1:p.Ne
        x = append!(s.path[i,1,:], s.path[i,1,1]) # loop the paths
        y = append!(s.path[i,2,:], s.path[i,2,1])
        z = append!(s.path[i,3,:], s.path[i,3,1])

        plot3d!(x, y, z,
			linewidth = 2, color = colors[i%Ncolors + 1],label = "Electron $i")
    end
    current()
end

# ╔═╡ 5652f370-7246-49df-a0d4-6fc47db787f5
function plot_paths_gnuplot(s::State)
    p = s.params

	colors = ["blue", "red", "green", "purple", "orange"]
	Ncolors=length(colors)

	@gp "set terminal svg enhanced size 600,600"
	@gp "set size square"
	@gp :- "set xlabel 'X' "
	@gp :- "set ylabel 'Y' "
	@gp :- "set zlabel 'Z' "

    for i in 1:p.Ne
        x = append!(s.path[i, 1, :], s.path[i, 1, 1])
        y = append!(s.path[i, 2, :], s.path[i, 2, 1])
        z = append!(s.path[i, 3, :], s.path[i, 3, 1])
        
		lc=colors[i%Ncolors + 1]
        @gsp :- x y z " with lp linewidth 2 lc rgb '$lc' t 'Electron $i' "
    end
	return @gp
end

# ╔═╡ Cell order:
# ╠═c5c96c64-eff2-11ed-04e9-e1642f1e52ee
# ╠═18fc3cb1-77d1-49e0-97d0-d53dabe691e7
# ╠═7a704a36-a05c-4248-bad0-04ed7d0a745d
# ╠═65eda18b-5db7-4c1b-ad7f-f1f6faa71c8a
# ╠═022fa92a-570d-45c6-b50c-10c13b741c4a
# ╠═add7f101-a827-4dc6-ab6c-93f00b4681d7
# ╠═fe4c179a-bbeb-46a8-8bc5-6fdca2f358f5
# ╠═df1e127b-7fc3-43ca-a05b-59e5381d75a2
# ╠═83f804d0-2d93-413b-9dab-e5bef166d502
# ╠═5cbc3649-9784-48a0-9598-744fa9f33279
# ╠═759916b1-a2c3-417e-a578-48162bdeba30
# ╠═d5bb6b3a-ac23-4e95-9323-e4ca00725144
# ╠═8654b316-cf71-45d1-b2fd-9ceded5d2a5d
# ╠═756efc67-b4ce-431e-9f4d-14926260c63d
# ╠═f749dd08-0f8c-423a-9c4a-79119ae5358d
# ╠═1a68ff25-4d75-4c62-a01b-cea826d3ff9c
# ╠═255a9be8-bd63-44cd-9cc2-546b123b5127
# ╠═28a2c299-89d6-47d3-8474-ba10c0531cb0
# ╠═f1123350-7079-412f-9771-867598477957
# ╠═6df6a3d8-a6bd-4baa-8f12-6ae18c52f751
# ╠═5727c703-246d-4b5d-b4b1-368719f299c9
# ╠═9b1fcb8d-3ac7-43f2-ae62-20a8d1530db3
# ╠═924dda10-f85a-42d6-962b-d9e92cf741de
# ╠═1a0f12ad-86c3-4abd-a038-8ddc021e934e
# ╠═194f526c-7956-426f-aeac-938c0299c2c3
# ╠═b73556c6-e692-4eca-8c80-75d80ecf231e
# ╠═017020dc-933b-4991-aeeb-028285b19ac6
# ╠═cfa86086-a635-4870-9092-578f7fab0383
# ╠═b2abcf83-77a1-4f75-89d8-8f6eec2b8cba
# ╠═5652f370-7246-49df-a0d4-6fc47db787f5
