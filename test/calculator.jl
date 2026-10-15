using AtomsCalculators

@testset "AtomsCalculators integration" begin
    mktempdir() do dir
        path = joinpath(dir, "model.pb")
        pyimport("deepmd.tf.utils.convert").convert_pbtxt_to_pb(
            joinpath(@__DIR__, "fixtures", "deeppot.pbtxt"), path)
        pot = DeepPotential(path)
        sys = atomic_system()
        result = AtomsCalculators.energy_forces_virial(sys, pot)
        @test AtomsCalculators.energy_unit(pot) == u"eV"
        @test AtomsCalculators.length_unit(pot) == u"Å"
        @test AtomsCalculators.force_unit(pot) == u"eV/Å"
        @test result.virial isa SMatrix{3, 3}
        @test result.energy ≈ DeepMD.energy_and_force(sys, pot).e
        @test force_values(result.forces) ≈ force_values(DeepMD.energy_and_force(sys, pot).f)
        @test AtomsCalculators.potential_energy(sys, pot) ≈ result.energy
        @test force_values(AtomsCalculators.forces(sys, pot)) ≈ force_values(result.forces)
        @test AtomsCalculators.virial(sys, pot) ≈ result.virial
        combined = AtomsCalculators.energy_forces(sys, pot)
        @test combined.energy ≈ result.energy
        @test force_values(combined.forces) ≈ force_values(result.forces)
        f = AtomsCalculators.zero_forces(sys, pot)
        @test AtomsCalculators.forces!(f, sys, pot) === f
        @test force_values(f) ≈ force_values(result.forces)
        @test_throws DimensionMismatch AtomsCalculators.forces!(f[1:2], sys, pot)
        # Validate matrix orientation against the raw row-major tensor.
        raw = DeepMD.predict_data(sys, pot)[2]
        @test ustrip.(u"eV", result.virial) ≈ pyconvert(Matrix{Float64}, raw.reshape((3, 3)))
        # A nonsymmetric sentinel catches row/column order mistakes that a
        # physical (symmetric) virial cannot expose. Also count backend calls.
        scope = pydict(np=DeepMD.np)
        pyexec("""
count = 0
def evaluate(*args):
    global count
    count += 1
    return np.array([[2.0]]), np.zeros((1, 3, 3)), np.arange(9.0).reshape(1, 9)
""", scope)
        pot.dp.eval = scope["evaluate"]
        sentinel = AtomsCalculators.energy_forces_virial(sys, pot)
        @test pyconvert(Int, scope["count"]) == 1
        @test ustrip.(sentinel.virial) == [0. 1. 2.; 3. 4. 5.; 6. 7. 8.]
        @test ustrip.(u"eV", DeepMD.virial_stress(sys, pot)) ≈ [0., 4., 8., 7., 6., 3.]
    end
end
