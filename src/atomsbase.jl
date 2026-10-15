struct DPLabeledSystem <: AbstractSystem{3}
    ls::Py
    function DPLabeledSystem(ls::Py; assert=true)
        if !pyisinstance(ls, dpdata.LabeledSystem)
            error("Given object is not a valid `LabeledSystem` and has type $(pytype(ls)).")
        end
        if length(ls) != 1
            error("Expected exactly one frame in the provided system.")
        end
        new(ls)
    end
end

function cell_vectors(lssys::DPLabeledSystem)
    ls = lssys.ls
    cellvecs = pyconvert(Vector{Vector{Float64}}, ls.data["cells"][0]) .* u"Å"

    return Tuple(SVector{3}.(cellvecs))
end
periodicity(sys::DPLabeledSystem) = ntuple(_ -> !pyconvert(Bool, sys.ls.data.get("nopbc", false)), 3)
AtomsBase.cell(sys::DPLabeledSystem) = AtomsBase.PeriodicCell(; cell_vectors=cell_vectors(sys), periodicity=periodicity(sys))
Base.length(lssys::DPLabeledSystem) = length(lssys.ls.data["coords"][0])
Base.size(lssys::DPLabeledSystem) = (length(lssys),)

function Base.getindex(sys::DPLabeledSystem, i::Integer)
    1 <= i <= length(sys) || throw(BoundsError(sys, i))
    return AtomView(sys, i)
end
function Base.position(lssys::DPLabeledSystem)
    ls = lssys.ls
    posvecs = pyconvert(Vector{Vector{Float64}}, ls.data["coords"][0]) .* u"Å"

    return SVector{3}.(posvecs)
end
function atomic_symbol(lssys::DPLabeledSystem)
    ls = lssys.ls
    attypes = pyconvert(Vector{Int}, ls.data["atom_types"])
    atnames = pyconvert(Vector{Symbol}, ls.data["atom_names"])
    atom_symbols = atnames[attypes .+ 1]

    return atom_symbols
end
function atomic_number(lssys::DPLabeledSystem)
    ls = lssys.ls
    attypes = pyconvert(Vector{Int}, ls.data["atom_types"])
    atnames = pyconvert(Vector{Symbol}, ls.data["atom_names"])
    atnums = getproperty.(element.(atnames), Ref(:number))
    atom_numbers = atnums[attypes .+ 1]

    return atom_numbers
end
function atomic_mass(lssys::DPLabeledSystem)
    ls = lssys.ls
    attypes = pyconvert(Vector{Int}, ls.data["atom_types"])
    atnames = pyconvert(Vector{Symbol}, ls.data["atom_names"])
    atmasses = getproperty.(element.(atnames), Ref(:atomic_mass))
    atom_masses = atmasses[attypes .+ 1]

    return atom_masses
end
velocity(::DPLabeledSystem) = missing

# System property access
function Base.getindex(lssys::DPLabeledSystem, x::Symbol)
    haskey(lssys, x) || throw(KeyError(x))
    ls = lssys.ls
    if x === :cell_vectors
        cell_vectors(lssys)
    elseif x === :periodicity
        periodicity(lssys)
    elseif x === :energy
        pyconvert(Float64, ls.data["energies"][0]) * u"eV"
    elseif x === :force
        pyconvert(Vector{Vector{Float64}}, ls.data["forces"][0]) .* u"eV/Å"
    elseif x === :virial
        pyconvert(Vector{Vector{Float64}}, ls.data["virials"][0]) .* u"eV"
    else
        throw(KeyError(x))
    end
end
const _label_keys = (:energy => "energies", :force => "forces", :virial => "virials")
Base.keys(sys::DPLabeledSystem) = (
    :cell_vectors, :periodicity,
    (key for (key, field) in _label_keys if pyconvert(Bool, sys.ls.data.__contains__(field)))...,
)
Base.haskey(sys::DPLabeledSystem, key::Symbol) = key in keys(sys)

# AtomsBase 0.5 indexed interface; retain the existing unindexed helpers.
Base.position(sys::DPLabeledSystem, i) = position(sys)[i]
AtomsBase.species(sys::DPLabeledSystem, i) = AtomsBase.ChemicalSpecies.(atomic_symbol(sys))[i]
AtomsBase.mass(sys::DPLabeledSystem, i) = atomic_mass(sys)[i]

# Atom and atom property access
atomkeys(sys::DPLabeledSystem) = (:position, :species, :mass, :atomic_symbol, :atomic_number, :atomic_mass, (haskey(sys, :force) ? (:force,) : ())...)
hasatomkey(system::DPLabeledSystem, x::Symbol) = x in atomkeys(system)
function Base.getindex(lssys::DPLabeledSystem, i::Union{Integer,AbstractVector,Colon}, x::Symbol)
    if x === :position
        position(lssys)[i]
    elseif x === :species
        AtomsBase.species(lssys, i)
    elseif x === :mass
        AtomsBase.mass(lssys, i)
    elseif x === :atomic_symbol
        atomic_symbol(lssys)[i]
    elseif x === :atomic_number
        atomic_number(lssys)[i]
    elseif x === :atomic_mass
        atomic_mass(lssys)[i]
    elseif x === :force
        lssys[:force][i]
    else
        throw(KeyError(x))
    end
end
