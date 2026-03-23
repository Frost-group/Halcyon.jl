using Test
using Halcyon
using BenchmarkTools

# Not very sophisticant tests currently

@testset "permutation_family" begin
    N = 4
    P = integer_partition_count_table(N)
    @test permutation_family_count(N, P) == 5

    λ211 = permutation_family_lambda(N, [2, 1, 1])
    @test permutation_family_index(λ211, P, N) == 4
    @test permutation_family_lambda_from_rank(5, N, P) == permutation_family_lambda(N, [1, 1, 1, 1])

    P10 = integer_partition_count_table(10)
    lambdas = [permutation_family_lambda_from_rank(k, 10, P10) for k in 1:permutation_family_count(10, P10)]
    @test length(lambdas) == 42
    inds = [permutation_family_index(λ, P10, 10) for λ in lambdas]
    @test inds == collect(1:42)

    # This bit is mostly just mocked, will get rewritten
    acc = DensePermutationFamilyStats(N)
    observe_permutation_family!(acc, λ211, 3.14)
    observe_permutation_family!(acc, permutation_family_lambda(N, [2, 1, 1]), 2.0)
    @test acc.count[4] == 2

    @test C_permutation_sector(λ211) == [2, 1, 0, 0]

#    bad = [2, 1, 0, 1] # got rid of this guard check for performance, so don't screw up!
#    @test_throws ArgumentError permutation_family_index(bad, P, N)
end

@testset "PermutationFamilyBenchmark" begin
    for N in (12, 24)
        P = integer_partition_count_table(N)

        # so consider non-permuting case, which we want to be be fast (happens a lot!)
        #  and then a very permutting case, which is less common, so accept slow down 
        for cycle_lengths in [fill(1, N), fill(4, N ÷ 4)]
            λ = permutation_family_lambda(N, cycle_lengths)
            idx = permutation_family_index(λ, P, N)

            t = @belapsed permutation_family_index($λ, $P, $N) seconds=0.25
            println("permutation_family_index N=$N: idx=$idx, λ=$λ, elapsed=$(round(t * 1e6; digits=2)) μs")
        end
    end
end