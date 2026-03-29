#!/usr/bin/env julia
# Playground for `Halcyon` permutation-family API (Sₙ conjugacy classes, C-vectors).
# `N` = particle count. C[ℓ] = number of cycles of length ℓ.
#
# Run: julia --project=. examples/PermutationFamily/permutation-scratchpad.jl

using Halcyon

function basic_test()
    N = 4
    P = integer_partition_count_table(N)
    C = permutation_family_C(N, [2, 1, 1])
    @show C_to_rank(C, P, N)
    @show C
    println("scratchpad: OK (import Halcyon + C-vector API).")
end

function print_permutation_families(N)
    P = integer_partition_count_table(N)
    indices = []
    for k in 1:permutation_family_count(N, P)
        C = C_from_rank(k, N, P)
        @show C
        @show sum(C[i]*i for i in 1:length(C)) == N
        push!(indices, C_to_rank(C, P, N))
        @show indices[end]
    end

    println("Full sorted indices: ", sort(indices))

    println("Young diagrams.")
    for k in 1:permutation_family_count(N, P)
        C = C_from_rank(k, N, P)
        @show(C)
        # draw rows from the descending partition (largest cycles first)
        for ℓ in N:-1:1
            for _ in 1:C[ℓ]
                println("⬜"^ℓ)
            end
        end
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    basic_test()
    print_permutation_families(10)
end
