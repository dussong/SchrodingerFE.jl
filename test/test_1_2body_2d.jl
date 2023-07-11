using LinearAlgebra
using Test

ne = 2
L = 5.0
N = 3
vext(x, y) = 0.1 * (x^2 + y^2)
vee(x, y) = sqrt(1e-3+x^2+y^2)
AΔ = oneB_lap_2d(L, N)
AV = oneB_V_2d(L, N, vext, 4, 3)
C = oneB_over_2d(L, N)
B = twoB_V_2d_sp(L, N, vee, 3, 3, 3, 3)
B2 = B'
@test norm(B2 - B) < 1e-12

#laplace
HΔ = ham_1B_sp(ne, AΔ, C)
@test norm(HΔ - HΔ') < 1e-12
# external
HV = ham_1B_sp(ne, AV, C)
@test norm(HV - HV') < 1e-12

# 2body interaction
Hee = ham_2B_sp(ne, B, C)
@test norm(Hee-Hee') < 1e-12
# total
H = HΔ + HV + Hee
@test norm(H-H') < 1e-12
