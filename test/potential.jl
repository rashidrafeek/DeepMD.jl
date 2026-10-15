@testset "Real TensorFlow model inference" begin
    mktempdir() do dir
        path = joinpath(dir, "model.pb")
        pyimport("deepmd.tf.utils.convert").convert_pbtxt_to_pb(
            joinpath(@__DIR__, "fixtures", "deeppot.pbtxt"), path)
        pot = DeepPotential(path)
        @test DeepPotential(pot.dp).dp === pot.dp
        @test_throws ErrorException DeepPotential(pydict())
        baseline = nothing
        for pbc in ((true, true, true), (false, false, false))
            for unit in (u"Å", u"nm", u"bohr")
                sys = atomic_system(; unit, pbc)
                e, f = DeepMD.energy_and_force(sys, pot)
                @test isfinite(ustrip(e))
                @test length(f) == length(sys)
                # Independently call the Python API with explicit Å inputs.
                coords = DeepMD.np.array([0. 0. 0.; 1. 0. 0.; 0. 1. 0.])
                cell = all(pbc) ? DeepMD.np.array([10. 0. 0.; 1. 10. 0.; 0. 0. 10.]) : nothing
                types = pyconvert(Vector{String}, pot.dp.get_type_map())
                ids = [findfirst(==(s), types) - 1 for s in ["H", "O", "H"]]
                pe, pf, pv = pot.dp.eval(coords, cell, DeepMD.np.array(ids))
                @test ustrip(u"eV", e) ≈ pyconvert(Float64, pe[0, 0])
                expected_f = pyconvert(Matrix{Float64}, pf[0])
                @test stack([ustrip.(u"eV/Å", v) for v in f]; dims=1) ≈ expected_f
                vi = DeepMD.virial_stress(sys, pot)
                expected_v = pyconvert(Matrix{Float64}, pv.reshape((3, 3)))
                @test ustrip.(u"eV", vi) ≈ expected_v[[1, 5, 9, 6, 3, 2]]
                if unit == u"Å"
                    baseline = (e, f)
                else
                    @test e ≈ baseline[1]
                    @test force_values(f) ≈ force_values(baseline[2])
                end
            end
        end
        source = atomic_system()
        empty_system = FlexibleSystem(collect(source)[1:0], cell_vectors(source), periodicity(source))
        @test_throws ArgumentError DeepMD.predict_data(empty_system, pot)
        h = 1e-5
        plus = atomic_system(; coords=[[h, 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        minus = atomic_system(; coords=[[-h, 0., 0.], [1., 0., 0.], [0., 1., 0.]])
        derivative = (ustrip(u"eV", DeepMD.energy_and_force(plus, pot).e) -
                      ustrip(u"eV", DeepMD.energy_and_force(minus, pot).e)) / (2h)
        @test ustrip(u"eV/Å", DeepMD.energy_and_force(atomic_system(), pot).f[1][1]) ≈ -derivative rtol=1e-5 atol=1e-6
        @test_throws ArgumentError DeepMD.predict_data(atomic_system(; symbols=[:He, :O, :H]), pot)
        @test_throws ArgumentError DeepMD.predict_data(atomic_system(; pbc=(true, false, false)), pot)
    end
end
