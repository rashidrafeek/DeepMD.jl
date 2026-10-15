using DeepMD
using Test
using PythonCall

@testset "DeepMD" begin
    @testset "Managed Python environment" begin
        @test pyconvert(String, DeepMD.deepmd.__version__) != ""
        @test pyhasattr(DeepMD.dpdata, "LabeledSystem")
        @test pyhasattr(pyimport("tensorflow"), "Graph")
        @test pyhasattr(pyimport("ase.io"), "read")
        @test pyhasattr(DeepMD.DeepPotPy, "eval")
    end
    include("helpers.jl")
    include("atomsbase.jl")
end
