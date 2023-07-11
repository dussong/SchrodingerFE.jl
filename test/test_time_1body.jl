using SchrodingerFE
using SparseArrays
using LinearAlgebra

using SchrodingerFE: ham_1B_sp

# test running time for 1-body Hamiltonian
L = 5.0;

# 2-particle
println("---------- 2-particle ----------\n")
ne = 2
for k = 0 : 2
    N = 4 * 2^k;
    println("\n ----- N = ", N, " -----\n");
    A = oneB_lap(L, N);
    C = oneB_over(L, N);
    @time begin
        H = ham_1B_sp(ne, A, C);
    end
    @test norm(H-H') < 1e-12
end

# 3-particle
println("---------- 3-particle ----------\n")
ne = 3
for k = 0 : 1
    N = 4 * 2^k;
    println("\n ----- N = ", N, " -----\n");
    A = oneB_lap(L, N);
    C = oneB_over(L, N);
    @time begin
        H = ham_1B_sp(ne, A, C);
    end
    @test norm(H-H') < 1e-12
end

# 4-particle
println("---------- 4-particle ----------\n")
ne = 4
for k = 0 : 1
    N = 4 * 2^k;
    println("\n ----- N = ", N, " -----\n");
    A = oneB_lap(L, N);
    C = oneB_over(L, N);
    @time begin
        H = ham_1B_sp(ne, A, C);
    end
    @test norm(H-H') < 1e-12
end
