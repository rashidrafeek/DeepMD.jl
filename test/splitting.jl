@testset "Dataset splitting" begin
    energies(ls) = pyconvert(Vector{Float64}, ls.data["energies"])
    for n in (0, 1, 2, 3, 10, 21), ratios in ((0.7, 0.2, 0.1), (1., 0., 0.), (0., 0.5, 0.5), (0.7, 0.3, 0.))
        ls = labeled_frames(n)
        parts = DeepMD.split_trainvalidtest_data(ls, ratios; rng=MersenneTwister(42))
        ids = energies.(values(parts))
        @test sum(length, values(parts)) == n
        @test sort(vcat(ids...)) == collect(0.:(n - 1))
        @test length(unique(vcat(ids...))) == n
        @test all(i -> !iszero(ratios[i]) || isempty(ids[i]), 1:3)
        again = DeepMD.split_trainvalidtest_data(ls, ratios; rng=MersenneTwister(42))
        @test energies.(values(again)) == ids
    end
    ls = labeled_frames(10)
    blocks = DeepMD.split_trainvalidtest_data(ls; shuffle=false)
    @test energies(blocks.ls_training) == collect(0.:6.)
    @test energies(blocks.ls_validation) == [7., 8.]
    @test energies(blocks.ls_testing) == [9.]
    for bad in ((1.,), (-0.1, 0.5, 0.6), (NaN, 0., 1.), (0., 0., 0.), (Inf, 0., 0.))
        @test_throws ArgumentError DeepMD.split_trainvalidtest_data(ls, bad)
    end
    mktempdir() do dir
        ls.to_deepmd_npy(dir)
        parts = DeepMD.split_trainvalidtest_data(dir; shuffle=false)
        @test energies(parts.ls_training) == collect(0.:6.)
    end
end
