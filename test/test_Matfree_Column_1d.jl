using LinearAlgebra
using StatsBase
using Test

L = 5.0;
N = 10;
ham = ham1d(L, N);
C = ham.C
k = 10
for ne = 2:5
    K = sample(1:binomial(2 * C.n, ne), k, replace=false)
    comb = map(x -> num2seq(2C.n, ne, x), K)
    Ψs = rand(k)
    Ψ = zeros(binomial(2 * C.n, ne))
    @. Ψ[K] = Ψs
    println("\n $(ne)-particle", "  N = $(N)")
    println("-------------------------")
    println(" Testing 1d hamiltonian")
    println("-------------------------")
    println(" Matfree_Tensor:")
    @time ah, am = ham_free_tensor(ne, Ψ, ham)
    println(" Matfree_Column:")
    @time bh, bm = ham_column(ne, comb, Ψs, ham)
    i1 = map(x -> num2seq_ns(2C.n, ne, x), bh.nzind)
    i1 = map(x -> seq2num(2C.n, ne, x), i1)
    bh = sparsevec(i1, bh.nzval, binomial(2C.n, ne))
    i2 = map(x -> num2seq_ns(2C.n, ne, x), bm.nzind)
    i2 = map(x -> seq2num(2C.n, ne, x), i2)
    bm = sparsevec(i2, bm.nzval, binomial(2C.n, ne))
    e1 = norm(ah - bh)
    @test(abs(e1) < 1e-12)
    println("  error:$(e1)\n")
    e2 = norm(am - bm)
    @test(abs(e2) < 1e-12)
    println("  error:$(e2)\n")
end

println("---------------------------------")
println(" Testing function ham_column_nonz")
println("---------------------------------")
L = 5.0;
N = 10;
ham = ham1d(L, N);
C = ham.C
for ne in [2, 3, 4]
    println("\n $(ne)-particle", "  N = $(N)")
    H, M = hamiltonian(ne,ham)
    H = Array(H)
    for j in sample(1:size(H, 1), 10, replace=false)
        l1 = findall(!iszero, H[:, j])
        l2 = ham_column_nonz(ne, ham, num2seq(2C.n, ne, j))
        l2 = map(x -> num2seq_ns(2C.n, ne, x), l2)
        l2 = map(x -> seq2num(2C.n, ne, x), l2)
        l2 = sort(l2)
        @test l1 == l2
    end
end
