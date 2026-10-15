@testset "Quantum ESPRESSO readers" begin
    out = joinpath(@__DIR__, "fixtures", "scf.out")
    a = DeepMD.getlabeledsystemdp_qescf(out)
    b = DeepMD.getdplabeledsystem_qescf(out, atomic_system())
    for key in ("energies", "forces", "virials", "coords", "cells")
        @test pyconvert(Array, a.data[key]) ≈ pyconvert(Array, b.data[key])
    end
    sys = DPLabeledSystem(a)
    @test ustrip(u"hartree", sys[:energy]) ≈ -5 atol=1e-6
    @test ustrip.(u"hartree/bohr", sys[:force][1]) ≈ [0.005, 0.01, 0.015] rtol=1e-6
    # Positive QE stress multiplied by cell volume, with no extra sign flip.
    @test ustrip(u"eV", sys[:virial][1][1]) ≈ ustrip(u"eV", 1u"kbar" * 1000u"Å^3") rtol=1e-6
    rotated = FlexibleSystem(
        [Atom(:H, [0., 0., 0.]u"Å"), Atom(:O, [0., 1., 0.]u"Å"), Atom(:H, [-1., 0., 0.]u"Å")],
        [[0., 10., 0.], [-10., 1., 0.], [0., 0., 10.]] .* u"Å", (true, true, true))
    rotated_labels = DPLabeledSystem(DeepMD.getdplabeledsystem_qescf(out, rotated))
    @test position(rotated_labels, 2) ≈ [1., 0., 0.]u"Å"
    @test ustrip.(u"hartree/bohr", rotated_labels[:force][1]) ≈ [0.01, -0.005, 0.015] rtol=1e-6
    @test DeepMD.getlabeledsystemdp_qescf!(a, out) === a
    @test length(a) == 2
    mktempdir() do dir
        # Directory names containing "out" must not be changed when finding input.
        subdir = joinpath(dir, "output"); mkdir(subdir)
        cp(out, joinpath(subdir, "scf.out"))
        cp(joinpath(@__DIR__, "fixtures", "scf.in"), joinpath(subdir, "scf.in"))
        @test length(DeepMD.getlabeledsystemdp_qescf(joinpath(subdir, "scf.out"))) == 1
        no_stress = joinpath(dir, "no_stress.out")
        write(no_stress, first(split(read(out, String), "     total   stress")))
        ls = DeepMD.getdplabeledsystem_qescf(no_stress, atomic_system(; pbc=(false, false, false)))
        @test !haskey(DPLabeledSystem(ls), :virial)
        @test periodicity(DPLabeledSystem(ls)) == (false, false, false)
        @test length(DeepMD.getlabeledsystemdp_qescf(no_stress; inname=joinpath(@__DIR__, "fixtures", "scf.in"))) == 1
    end
end
