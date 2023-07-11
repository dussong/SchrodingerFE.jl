using Test

Lx = 5.0;
Ly = 3.0;
Nx = 10;
Ny = 5;
L = [Lx,Ly]
N = [Nx,Ny]
ham = ham2d(L,N);
C = ham.C
k = 10
for ne in [2, 3]
    K = sample(1:binomial(2 * C.n, ne), k, replace=false)
    comb = map(x -> num2seq(2C.n, ne, x), K)
    Ψs = rand(k)
    Ψ = zeros(binomial(2 * C.n, ne))
    @. Ψ[K] = Ψs
    println("\n $(ne)-particle", "  N = $(N)")
    println("-------------------------")
    println(" Testing 2d hamiltonian")
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
