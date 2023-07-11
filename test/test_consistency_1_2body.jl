using SchrodingerFE
using SparseArrays
using Printf
using LinearAlgebra
using Test

include("check_coulomb.jl")

# Check that the matrices from the old and new implementation match
L = 5.
N = 30

# --------------------------------------------------------------
# One-body tests
# --------------------------------------------------------------
println("-------------------------------------------")
println("         Testing matching one-body matrices")
println("-------------------------------------------")

println("         Testing overlap matrix")

println("---------------")
println("n_gauss_points | error")
println("---------------")
errs = []
for nx in 1:20
    M1 = oneB_over(L, N)
    M2 = oneB_V(L,N,x->1,nx)
    push!(errs, norm(M1-M2))
    @printf(" %d | %.2e \n", nx, errs[end])
end
println("---------------")
@test minimum(errs) <= 1e-3 * maximum(errs)

# Laplace one-body matrix, just checking that the function runs (no real test)
mat_lap_1B = oneB_lap(L,N)



# --------------------------------------------------------------
# Two-body tests
# --------------------------------------------------------------
println("-------------------------------------------")
println("         Testing matching two-body matrices")
println("-------------------------------------------")

println("         Testing coulomb matrix:")


# Defining Coulomb
f(x) = 1.0./abs.(x)

# Coulomb matrix from old Matlab implementation turned to Julia
@time M1 = mat_2d_coulomb(L, N)
@time M2 = twoB_V(L,N,f,4,3)
@test norm(M1-M2) < 1e-8



@time M3 = twoB_V_sp(L,N,f,4,3)
M4 = zeros(N-1,N-1,N-1,N-1)
for i in 1:size(M3[1],2)
    M4[M3[1][1,i],M3[1][2,i],M3[1][3,i],M3[1][4,i]] = M3[2][i]
end
M4
@test norm(M4-M2) < 1e-8
