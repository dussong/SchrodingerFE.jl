using LinearAlgebra
using Arpack

# vext = 1/x^2

# hamiltonian
ne1 = [2, 3];
N1 = [10, 5];
L = 5.0;

for (ne, N) in zip(ne1, N1)
    H, M = hamiltonian(ne, ham1d(L, N))
    println("\n $(ne)-particle", "  N = $(N)")
    println("-----------------------")
    println(" Testing Hamiltonian with inv_power")
    println("-----------------------")
    # directly compute the minimum eigenvalue and eigenvector of Hamiltonian
    println(" no matrix free:")
    @time begin
        E1, Ψ = eigs(H, M, nev=2, which=:SM)
    end
    Ψ1 = Ψ[:, 1]
    gap = abs(E1[1] - E1[2])
    println(" gap = $(gap)")
    if gap < 1e-4
        println(" eigenvalues degenerate")
    end
    # using aitken inverse power method through matries free
    println(" power method:")
    @time E2, Ψ2, res2 = inv_power(ne, H, M; max_iter=500, tol=1.0e-10)
    e1 = norm(E2 - E1[1])
    @test(abs(e1) < 1e-3)
    println("  eigenvalue's error = $(e1)")
    Ψ1 = Ψ1 ./ norm(Ψ1)
    Ψ2 = Ψ2 ./ norm(Ψ2)
    e2 = min(norm(Ψ2 + Ψ1), norm(Ψ2 - Ψ1))
    #@test(abs(e2) < 1e-2)
    println("  eigenvector's error = $(e2)")
end