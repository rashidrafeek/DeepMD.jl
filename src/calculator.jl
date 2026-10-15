AtomsCalculators.energy_unit(::DeepPotential) = u"eV"
AtomsCalculators.length_unit(::DeepPotential) = u"Å"

function AtomsCalculators.energy_forces_virial(sys::AbstractSystem, pot::DeepPotential; kwargs...)
    e, f, v = predict_data(sys, pot)
    energy = pyconvert(Float64, e[0, 0]) * u"eV"
    forces = pyconvert(Vector{SVector{3, Float64}}, f[0]) .* u"eV/Å"
    virial = SMatrix{3, 3}(pyconvert(Matrix{Float64}, v.reshape((3, 3)))) .* u"eV"
    return (; energy, forces, virial)
end

function AtomsCalculators.energy_forces(sys::AbstractSystem, pot::DeepPotential; kwargs...)
    result = AtomsCalculators.energy_forces_virial(sys, pot; kwargs...)
    return (; result.energy, result.forces)
end

AtomsCalculators.potential_energy(sys::AbstractSystem, pot::DeepPotential; kwargs...) =
    AtomsCalculators.energy_forces_virial(sys, pot; kwargs...).energy
AtomsCalculators.forces(sys::AbstractSystem, pot::DeepPotential; kwargs...) =
    AtomsCalculators.energy_forces_virial(sys, pot; kwargs...).forces
AtomsCalculators.virial(sys::AbstractSystem, pot::DeepPotential; kwargs...) =
    AtomsCalculators.energy_forces_virial(sys, pot; kwargs...).virial

function AtomsCalculators.forces!(f, sys::AbstractSystem, pot::DeepPotential; kwargs...)
    length(f) == length(sys) || throw(DimensionMismatch("Force buffer must have one entry per atom."))
    copyto!(f, AtomsCalculators.forces(sys, pot; kwargs...))
    return f
end
