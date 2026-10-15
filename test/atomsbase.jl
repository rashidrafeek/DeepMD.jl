@testset "AtomsBase and dpdata conversion" begin
    for nonperiodic in (false, true), virials in (false, true)
        ls = labeled_frames(; nopbc=nonperiodic, virials)
        sys = DPLabeledSystem(ls)
        @test length(sys) == 3
        @test size(sys) == (3,)
        @test periodicity(sys) == ntuple(_ -> !nonperiodic, 3)
        @test periodicity(AtomsBase.cell(sys)) == periodicity(sys)
        @test atomic_symbol(sys) == [:H, :O, :H]
        @test Symbol.(atomic_symbol(sys, :)) == [:H, :O, :H]
        @test atomic_number(sys) == [1, 8, 1]
        @test atomic_number(sys[2]) == 8
        @test position(sys[2]) == [1., 0., 0.]u"Å"
        @test AtomsBase.mass(sys[2]) ≈ atomic_mass(sys)[2]
        @test sys[2, :species] == AtomsBase.species(sys, 2)
        @test sys[2, :mass] == AtomsBase.mass(sys, 2)
        @test sys[:, :position] == position(sys, :)
        @test sys[[1, 3], :atomic_number] == [1, 1]
        @test sys[:energy] == 0u"eV"
        @test length(sys[:force]) == 3
        @test sys[2, :force] == zeros(3)u"eV/Å"
        @test all(k -> haskey(sys, k), keys(sys))
        @test haskey(sys, :virial) == virials
        @test (:virial in keys(sys)) == virials
        if virials
            @test sys[:virial][1] == [1., 0., 0.]u"eV"
        else
            @test_throws KeyError sys[:virial]
        end
        @test_throws KeyError sys[:unknown]
        @test_throws KeyError sys[1, :unknown]
        @test_throws BoundsError sys[0]
        @test_throws BoundsError sys[4]
        @test ismissing(velocity(sys, 1))
        roundtrip = getdpsystem(sys)
        @test pyconvert(Bool, roundtrip.data["nopbc"]) == nonperiodic
        @test pyconvert(Array, roundtrip.data["coords"]) == pyconvert(Array, ls.data["coords"])
        @test pyconvert(Vector{String}, roundtrip.data["atom_names"]) == ["H", "O"]
    end
    for unit in (u"Å", u"nm", u"bohr")
        sys = atomic_system(; unit)
        data = getdpsystem(sys).data
        @test pyconvert(Array, data["coords"])[1, 2, :] ≈ [1., 0., 0.]
        @test pyconvert(Array, data["cells"])[1, 2, :] ≈ [1., 10., 0.]
        @test pyconvert(Vector{Int}, data["atom_types"]) == [0, 1, 0]
    end
    source = atomic_system()
    empty_system = FlexibleSystem(collect(source)[1:0], cell_vectors(source), periodicity(source))
    @test_throws ArgumentError getdpsystem(empty_system)
    @test length(getdpsystem([atomic_system(), atomic_system()])) == 2
    @test_throws ArgumentError getdpsystem(AtomsBase.AbstractSystem[])
    @test_throws ArgumentError getdpsystem(atomic_system(; pbc=(true, false, false)))
    @test_throws ArgumentError getdpsystem([atomic_system(), atomic_system(; pbc=(false, false, false))])
    @test_throws ErrorException DPLabeledSystem(labeled_frames(0))
    @test_throws ErrorException DPLabeledSystem(labeled_frames(2))
    @test_throws ErrorException DPLabeledSystem(pydict())
    # LabeledSystem permits absent force labels.
    sys = DPLabeledSystem(labeled_frames(; forces=false))
    @test !haskey(sys, :force)
    @test !hasatomkey(sys, :force)
    @test_throws KeyError sys[:force]
end
