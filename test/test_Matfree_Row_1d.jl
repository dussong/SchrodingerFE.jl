using LinearAlgebra
using Combinatorics
using StaticArrays, SparseArrays
using Test

L = 5.0;
N = 20;
ham = ham1d(L,N)
C = ham.C
for ne in [2, 3, 4]
    Ψ = rand(binomial(2C.n, ne))
    k = rand(1:binomial(2 * C.n, ne), 10)
    k1 = map(x -> num2seq(2C.n, ne, x), k)
    k1 = map(x -> seq2num_ns(2C.n, ne, x), k1)
    k2 = collect(combinations(1:2C.n, ne))
    k2 = map(x -> seq2num_ns(2C.n, ne, x), k2)
    Ψ2 = sparsevec(k2, Ψ, (2C.n)^ne)
    println("\n $(ne)-particle", "  N = $(N)")
    println("-------------------------")
    println(" Testing 1d hamiltonian")
    println("-------------------------")
    println(" Matfree_Tensor:")
    @time ah, am = ham_free_tensor(ne, Ψ, ham)
    println(" Matfree_Row:")
    @time bh, bm = ham_row(k1, ne, Ψ2, ham)
    e1 = norm(ah[k] - bh)
    e2 = norm(am[k] - bm)
    @test(abs(e1) < 1e-12)
    @test(abs(e2) < 1e-12)
end
