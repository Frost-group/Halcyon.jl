using Test
using Halcyon
using BenchmarkTools

@testset "permutation_family" begin
    N = 4
    P = integer_partition_count_table(N)
    @test permutation_family_count(N, P) == 5

    # C-vector for two 1-cycles and one 2-cycle: [2,1,1] → C = [2,1,0,0]
    C_211 = permutation_family_C(N, [2, 1, 1])
    @test C_211 == [2, 1, 0, 0]
    @test C_to_rank(C_211, P, N) == 4

    # Identity permutation (all 1-cycles)
    C_1111 = permutation_family_C(N, [1, 1, 1, 1])
    @test C_1111 == [4, 0, 0, 0]
    @test C_from_rank(5, N, P) == C_1111

    # Round-trip bijection for all p(10) = 42 partitions
    P10 = integer_partition_count_table(10)
    for k in 1:permutation_family_count(10, P10)
        C = C_from_rank(k, 10, P10)
        @test sum(ℓ * C[ℓ] for ℓ in 1:10) == 10   # constraint check
        @test C_to_rank(C, P10, 10) == k             # round-trip
    end

    # Observation accumulation using C-vectors
    acc = DensePermutationFamilyStats(N)
    observe_permutation_family!(acc, C_211, 3.14)
    observe_permutation_family!(acc, permutation_family_C(N, [2, 1, 1]), 2.0)
    @test acc.count[4] == 2
end

@testset "PermutationFamilyBenchmark" begin
    for N in (12, 24)
        P = integer_partition_count_table(N)

        for cycle_lengths in [fill(1, N), fill(4, N ÷ 4)]
            C = permutation_family_C(N, cycle_lengths)
            idx = C_to_rank(C, P, N)

            println("Benchmarking C_to_rank(C=$C, ..., N=$N)")
            b = @benchmark C_to_rank($C, $P, $N) seconds=0.25
            show(stdout, MIME("text/plain"), b)
            println()
        end
    end
end