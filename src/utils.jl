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

function getlabeledsystemdp_qescf(
        outname; inname=replace(outname, "out"=>"in")
    )
    outlines = readlines(outname)
    inlines = readlines(inname)
    fmtobj = dpdata.system.load_format("qe/pw/scf")
    
    cell = aseio.read(inname).cell[]
    atom_names, natoms, types, coords = dpdata.qe.scf.get_coords(inlines, cell)
    energy = dpdata.qe.scf.get_energy(outlines)
    force = dpdata.qe.scf.get_force(outlines, natoms)
    stress = dpdata.qe.scf.get_stress(outlines) * np.linalg.det(cell)

    data = pydict()
    (
            data["atom_names"],
            data["atom_numbs"],
            data["atom_types"],
            data["cells"],
            data["coords"],
            data["energies"],
            data["forces"],
            data["virials"],
    ) = (
        atom_names,
        natoms,
        types,
        cell[np.newaxis, pycolon, pycolon],
        coords[np.newaxis, pycolon, pycolon],
        np.array(energy)[np.newaxis],
        force[np.newaxis, pycolon, pycolon],
        stress[np.newaxis, pycolon, pycolon],
    )

    ls = dpdata.LabeledSystem()
    ls.data.update(data)
    ls.check_data()
    if pyhasattr(fmtobj.from_labeled_system, "post_func")
        for post_f in fmtobj.from_labeled_system.post_func
            ls.post_funcs.get_plugin(post_f)(ls)
        end
    end
    
    return ls
end

function getdplabeledsystem_qescf(
        outname, system::AbstractSystem
    )
    outlines = readlines(outname)
    fmtobj = dpdata.system.load_format("qe/pw/scf")
    
    atom_names, natoms, types, cell, coords = _getdpsystemdata(system)
    energy = dpdata.qe.scf.get_energy(outlines)
    force = dpdata.qe.scf.get_force(outlines, natoms)
    stress = dpdata.qe.scf.get_stress(outlines) * np.linalg.det(cell)

    data = pydict()
    (
            data["atom_names"],
            data["atom_numbs"],
            data["atom_types"],
            data["cells"],
            data["coords"],
            data["orig"],
            data["energies"],
            data["forces"],
            data["virials"],
    ) = (
        atom_names,
        natoms,
        types,
        cell[np.newaxis, pycolon, pycolon],
        coords[np.newaxis, pycolon, pycolon],
        np.array([0, 0, 0]),
        np.array(energy)[np.newaxis],
        force[np.newaxis, pycolon, pycolon],
        stress[np.newaxis, pycolon, pycolon],
    )

    ls = dpdata.LabeledSystem(;data)
    if pyhasattr(fmtobj.from_labeled_system, "post_func")
        for post_f in fmtobj.from_labeled_system.post_func
            ls.post_funcs.get_plugin(post_f)(ls)
        end
    end
    
    return ls
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
