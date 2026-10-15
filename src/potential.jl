#
# Types
#

export DeepPotential

struct DeepPotential
    dp::Py
    function DeepPotential(dp::Py; assert=true)
        if !pyisinstance(dp, DeepPotPy)
            error("Given potential is not a valid `DeepPot` and has type $(pytype(dp)).")
        end
        new(dp)
    end
end
DeepPotential(path::AbstractString) = DeepPotential(DeepPotPy(path))

#
# Functions
#

function predict_data(sys::AbstractSystem, deeppot::DeepPotential)
    type_map = pyconvert(Vector{Symbol}, deeppot.dp.get_type_map())

    isempty(sys) && throw(ArgumentError("Cannot evaluate an empty system."))
    coord = stack([ustrip.(u"Å", r) for r in position(sys, :)]; dims=1)
    cell = _nonperiodic(sys) ? nothing :
        stack([ustrip.(u"Å", v) for v in cell_vectors(sys)]; dims=1)
    symbols = Symbol.(atomic_symbol(sys, :))
    unknown = setdiff(symbols, type_map)
    isempty(unknown) || throw(ArgumentError("Species absent from model type map: $(join(unknown, ", "))."))
    atype = [findfirst(==(symbol), type_map) - 1 for symbol in symbols]

    return deeppot.dp.eval(np.array(coord), isnothing(cell) ? nothing : np.array(cell), np.array(atype; dtype="int32"))
end

function energy_and_force(sys::AbstractSystem, deeppot::DeepPotential)
    energy, force, _ = predict_data(sys, deeppot)
    e = uconvert(u"hartree", pyconvert(Float64, first(first(energy)))u"eV")
    f = map(pyconvert(Vector{SVector{3, Float64}}, first(force))u"eV/Å") do v
        uconvert.(u"hartree/bohr", v)
    end

    return (; e, f)
end

"""
    virial_stress(sys, potential)

Legacy six-component **virial**, in hartree, ordered `(xx, yy, zz, zy, zx, yx)`.
This is not volume-normalized stress. Prefer `AtomsCalculators.virial` for a matrix.
"""
function virial_stress(sys::AbstractSystem, deeppot::DeepPotential)
    _, _, virial_stress = predict_data(sys, deeppot)
    vi_mat = first(virial_stress.reshape((1,3,3)))
    vi = uconvert.(u"hartree", pyconvert(Matrix{Float64}, vi_mat)u"eV")
    # Taken from "types/empirical_potential.jl:64"
    v = @SVector [vi[1, 1], vi[2, 2], vi[3, 3], vi[3, 2], vi[3, 1], vi[2, 1]]

    return v
end
