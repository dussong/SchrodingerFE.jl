
# Testing the Nbody matrices free
ne1 = [2,3,4]
N1 = [10,10,10]
L = 5.0;
println("-----------------------------------------------")
println(" Testing matrix-vector multiplication")
println("-----------------------------------------------")
for (ne, N) in zip(ne1, N1)
    ham = ham1d(L, N)
    n = 2 * ham.C.n
    Ψ = rand(binomial(n, ne))
    println("\n $(ne)-particle", "  N = $(N)")
    println("-------------------------")
    println(" Testing 1-body operator")
    println("-------------------------")
    println(" no matrix free:")
    @time begin
        HΔ = ham_1B_sp(ne, ham.AΔ, ham.C)
        W1 = HΔ * Ψ
    end
    println(" matrix free:")
    @time begin
        W2 = ham_1B_free_tensor(ne, Ψ, ham.AΔ, ham.C)
    end
    e1 = norm(W2 - W1)
    @test(abs(e1) < 1e-12)
    println("  error = $(e1)")

    println("-------------------------")
    println(" Testing 2-body operator")
    println("-------------------------")
    println(" no matrix free:")
    @time begin
        Hee = ham_2B_sp(ne, ham.Bee, ham.C)
        W3 = Hee * Ψ
    end
    println(" matrix free:")
    @time begin
        W4 = ham_2B_free_tensor(ne, Ψ, ham.Bee, ham.C)
    end
    e2 = norm(W4 - W3)
    @test(abs(e2) < 1e-12)
    println("  error = $(e2)")

    println("-------------------------")
    println(" Testing full hamiltonian")
    println("-------------------------")
    println(" no matrix free:")
    @time begin
        H, M = hamiltonian(ne,ham)
        w1 = H * Ψ
        m1 = M * Ψ
    end
    println(" matrix free:")
    @time begin
        w2, m2 = ham_free_tensor(ne, Ψ, ham)
    end
    e3 = norm(w2-w1)
    e4 = norm(m2-m1)
    @test(abs(e3) < 1e-12 && abs(e4) < 1e-12)
    println("  ham's error = $(e3), overlap's error = $(e4)")
end