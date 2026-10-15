# Run in a fresh Julia process: julia --startup-file=no test/install.jl
# Uses Pkg.add on the committed local repository, like installing its Git URL.
using Pkg
using Test

root = dirname(@__DIR__)
Pkg.activate(mktempdir())
Pkg.add(PackageSpec(url=root))
using DeepMD

@testset "Pkg.add installation" begin
    @test !isempty(DeepMD.pyconvert(String, DeepMD.deepmd.__version__))
    mktempdir() do dir
        path = joinpath(dir, "model.pb")
        DeepMD.pyimport("deepmd.tf.utils.convert").convert_pbtxt_to_pb(
            joinpath(root, "test", "fixtures", "deeppot.pbtxt"), path)
        pot = DeepPotential(path)
        np = DeepMD.np
        data = DeepMD.pydict(
            atom_names=DeepMD.pylist(["O", "H"]), atom_numbs=DeepMD.pylist([1, 2]),
            atom_types=np.array([0, 1, 1]), orig=np.zeros(3),
            cells=(np.eye(3) * 10).reshape((1, 3, 3)),
            coords=np.array([0. 0. 0.; 1. 0. 0.; 0. 1. 0.]).reshape((1, 3, 3)),
            energies=np.zeros(1))
        sys = DPLabeledSystem(dpdata.LabeledSystem(; data))
        result = DeepMD.energy_and_force(sys, pot)
        @test isfinite(DeepMD.ustrip(result.e))
        @test length(result.f) == 3
    end
end
