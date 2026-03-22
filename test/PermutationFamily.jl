using Test
using Halcyon

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

    bad = [2, 1, 0, 1]
    @test_throws ArgumentError permutation_family_index(bad, P, N)
end
