using SchrodingerFE
using Test

@testset "SchrodingerFE.jl" begin
   @testset "consistency_1_2body" begin include("test_consistency_1_2body.jl") end
   @testset "full_ham" begin include("test_full_ham.jl") end
   @testset "time_1body" begin include("test_time_1body.jl") end
   @testset "time_2body" begin include("test_time_2body.jl") end
   @testset "MatFreeTensor" begin include("test_MatFreeTensor.jl") end
   @testset "1_2body_2d" begin include("test_1_2body_2d.jl") end
   @testset "geneigsolve" begin include("test_geneigsolve.jl") end
   @testset "Matfree_Column_1d" begin include("test_Matfree_Column_1d.jl") end
   @testset "Matfree_Column_2d" begin include("test_Matfree_Column_2d.jl") end
   @testset "Matfree_Row_1d" begin include("test_Matfree_Row_1d.jl") end
   @testset "Matfree_Row_2d" begin include("test_Matfree_Row_2d.jl") end
   @testset "power_method" begin include("test_power_method.jl") end

end
