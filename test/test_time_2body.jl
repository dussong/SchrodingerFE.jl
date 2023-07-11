using SchrodingerFE
using SchrodingerFE: ham_2B_sp
using SparseArrays
using LinearAlgebra

# test running time for 1-body Hamiltonian
L = 5.0;
nx = 3;
ny = 4;

# 2-particle
println("---------- 2-particle ----------\n")
ne = 2
for k = 0 : 4
    N = 4 * 2^k;
    println("\n ----- N = ", N, " -----\n");
    @time begin
        B = twoB_V(L, N, x->x[1]+x[2], nx, ny);
        C = oneB_over(L, N);
        H = ham_2B_sp(ne, B, C);
    end
    @assert norm(H-H') < 1e-12
end

# 3-particle
println("---------- 3-particle ----------\n")
ne = 3
for k = 0 : 2
    N = 4 * 2^k;
    println("\n ----- N = ", N, " -----\n");
    @time begin
        B = twoB_V(L, N,  x->x[1]+x[2], nx, ny);
        C = oneB_over(L, N);
        H = ham_2B_sp(ne, B, C);
    end
    @assert norm(H-H') < 1e-12
end

# 4-particle
println("---------- 4-particle ----------\n")
ne = 4
for k = 0 : 2
    N = 4 * 2^k;
    println("\n ----- N = ", N, " -----\n");
    @time begin
        B = twoB_V(L, N, x->x[1]+x[2], nx, ny);
        C = oneB_over(L, N);
        H = ham_2B_sp(ne, B, C);
    end
    @assert norm(H-H') < 1e-12
end
