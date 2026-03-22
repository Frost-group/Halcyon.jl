#!/usr/bin/env julia
# Playground for `Halcyon` permutation-family API (S_N cycle type, padded λ).
# `N` = particle count (e.g. `sys.N`). Not worm Z/G sector.
#
# Run: julia --project=. examples/permutation/permutation-scratchpad.jl

using Halcyon

function basic_test()
    N = 4
    P = integer_partition_count_table(N)
    λ = permutation_family_lambda(N, [2, 1, 1])
    @show permutation_family_index(λ, P, N)
    @show permutation_family_multiplicity_for_display(λ)
    println("scratchpad: OK (import Halcyon + permutation_family_*).")
end

function print_permutation_families(N)
    P = integer_partition_count_table(N)
    indices=[]
    for k in 1:permutation_family_count(N, P)
        λ = permutation_family_lambda_from_rank(k, N, P)
        @show λ
        @show C = permutation_family_multiplicity_for_display(λ)
        @show sum(C[i]*i for i in 1:length(C)) == N
        push!(indices, permutation_family_index(λ, P, N))
        @show indices[end]
    end

    println("Full sorted indicex: ", sort(indices))

    println("Young diagrams.")
    for k in 1:permutation_family_count(N, P)
        λ = permutation_family_lambda_from_rank(k, N, P)
        @show(λ)
        for l in λ
            l==0 && break
            println("⬜"^l)
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    basic_test()
    print_permutation_families(10)
end
