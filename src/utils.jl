# DeePMD supports fully periodic or fully nonperiodic systems.
function _nonperiodic(sys::AbstractSystem)
    pbc = periodicity(sys)
    (all(pbc) || !any(pbc)) || throw(ArgumentError("Mixed periodicity is unsupported by DeePMD."))
    return !any(pbc)
end

"""
    _getdpsystemdata(system::AbstractSystem)

Get all data to create a dpdata System object from an AbstractSystem.
"""
function _getdpsystemdata(system::AbstractSystem)
    isempty(system) && throw(ArgumentError("Cannot convert an empty atomic system."))
    cell = np.array(stack(map(cell_vectors(system)) do vec
        ustrip.(u"Å", vec)
    end; dims=1))
    coords = np.array(stack(map(position(system, :)) do vec
        ustrip.(u"Å", vec)
    end; dims=1))

    atsymbols = Symbol.(atomic_symbol(system, :))
    unq_atsymbols = unique(atsymbols)

    atom_names = pylist(string.(unq_atsymbols))
    natoms = pylist(count.(.==(unq_atsymbols), Ref(atsymbols)))
    types = np.array(findfirst.(.==(atsymbols), Ref(unq_atsymbols)) .- 1)

    return atom_names, natoms, types, cell, coords
end

"""
    getdpsystem(system::AbstractSystem)

Obtain a System object (Unlabelled system) object from an `AbstractSystem`
compatible with AtomsBase.
"""
function getdpsystem(system::AbstractSystem)
    atom_names, natoms, types, cell, coords = _getdpsystemdata(system)

    data = pydict()
    (
            data["atom_names"],
            data["atom_numbs"],
            data["atom_types"],
            data["cells"],
            data["coords"],
            data["orig"]
    ) = (
        atom_names,
        natoms,
        types,
        cell[np.newaxis, pycolon, pycolon],
        coords[np.newaxis, pycolon, pycolon],
        np.array([0, 0, 0])
    )

    data["nopbc"] = _nonperiodic(system)
    dpsys = dpdata.System(;data)

    return dpsys
end
function getdpsystem(systems::AbstractVector{<:AbstractSystem})
    isempty(systems) && throw(ArgumentError("Provide at least one system."))
    all(s -> _nonperiodic(s) == _nonperiodic(first(systems)), systems) ||
        throw(ArgumentError("Cannot combine periodic and nonperiodic frames."))
    dpsystems = getdpsystem.(systems)
    dpsys = first(dpsystems)
    nsys = length(dpsystems)
    if nsys > 1
        for i in 2:nsys
            dpsys.append(dpsystems[i])
        end
    end

    return dpsys
end

# Normalize geometry through an ibrav=0 input, then let dpdata's public reader
# parse labels and apply its coordinate/force/virial rotations consistently.
function _read_qe_labels(outname, symbols, coords, box; nopbc=false)
    mktempdir() do dir
        input = joinpath(dir, "geometry.in")
        open(input, "w") do io
            println(io, "&SYSTEM\n  ibrav = 0,\n/\nCELL_PARAMETERS angstrom")
            for row in eachrow(box)
                println(io, join(row, " "))
            end
            println(io, "\nATOMIC_POSITIONS angstrom")
            for (symbol, row) in zip(symbols, eachrow(coords))
                println(io, symbol, " ", join(row, " "))
            end
            println(io)
        end
        ls = dpdata.LabeledSystem(pylist([input, abspath(outname)]); fmt="qe/pw/scf")
        ls.data["nopbc"] = nopbc
        return ls
    end
end

"""
    getlabeledsystemdp_qescf(outname; inname=...)

Read a QE SCF input/output pair. ASE reads the input geometry; dpdata's public
reader handles energy, forces, optional virial, and cell rotation. Labels use
eV, eV/Å, and eV. The default replaces `out` with `in` in the basename, leaving directories intact.
"""
function getlabeledsystemdp_qescf(
        outname; inname=joinpath(dirname(outname), replace(basename(outname), "out" => "in"))
    )
    atoms = pyimport("ase.io").read(inname; format="espresso-in")
    symbols = pyconvert(Vector{String}, atoms.get_chemical_symbols())
    coords = pyconvert(Matrix{Float64}, atoms.get_positions())
    box = pyconvert(Matrix{Float64}, atoms.cell.array)
    return _read_qe_labels(outname, symbols, coords, box)
end

"""
    getdplabeledsystem_qescf(outname, system)

Read QE SCF labels using an AtomsBase system as the input geometry.
"""
function getdplabeledsystem_qescf(outname, system::AbstractSystem)
    nopbc = _nonperiodic(system)
    symbols = string.(Symbol.(atomic_symbol(system, :)))
    coords = stack([ustrip.(u"Å", r) for r in position(system, :)]; dims=1)
    box = stack([ustrip.(u"Å", v) for v in cell_vectors(system)]; dims=1)
    return _read_qe_labels(outname, symbols, coords, box; nopbc)
end

function getlabeledsystemdp_qescf!(ls::Py, outname; kwargs...)
    ls.append(getlabeledsystemdp_qescf(outname; kwargs...))

    return ls
end

function split_trainvalidtest_data(
        dir::String, ratios=(0.7, 0.2, 0.1); fmt="deepmd/npy", kwargs...
    )
    ls = dpdata.LabeledSystem(dir; fmt)

    split_trainvalidtest_data(ls, ratios; kwargs...)
end
"""
    split_trainvalidtest_data(ls, ratios=(0.7, 0.2, 0.1); rng=Random.default_rng(), shuffle=true)

Split frames without overlap. Counts use largest-remainder allocation (ties prefer
training, then validation). Pass an RNG for reproducibility, or `shuffle=false`
to retain contiguous trajectory blocks. Zero ratios always receive zero frames.
"""
function split_trainvalidtest_data(
        ls::Py, ratios=(0.7, 0.2, 0.1); rng=Random.default_rng(), shuffle=true
    )
    length(ratios) == 3 || throw(ArgumentError("Provide three split ratios."))
    all(r -> r isa Real && isfinite(r) && r >= 0, ratios) ||
        throw(ArgumentError("Split ratios must be finite and nonnegative."))
    isapprox(sum(ratios), 1) || throw(ArgumentError("Split ratios must sum to one."))
    nframes = length(ls)
    expected = nframes .* collect(ratios) ./ sum(ratios)
    counts = floor.(Int, expected)
    order = sortperm(1:3; by=i -> (-(expected[i] - counts[i]), i))
    for i in order[1:(nframes - sum(counts))]
        counts[i] += 1
    end
    indices = collect(0:(nframes - 1))
    shuffle && Random.shuffle!(rng, indices)
    ntrain, nvalid, _ = counts
    # Explicit NumPy integer arrays also work for empty selections.
    subset(ids) = ls.sub_system(np.array(ids; dtype="int64"))
    ls_training = subset(indices[1:ntrain])
    ls_validation = subset(indices[(ntrain + 1):(ntrain + nvalid)])
    ls_testing = subset(indices[(ntrain + nvalid + 1):end])
    return (; ls_training, ls_validation, ls_testing)
end
