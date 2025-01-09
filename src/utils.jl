"""
    _getdpsystemdata(system::AbstractSystem)

Get all data to create a dpdata System object from an AbstractSystem.
"""
function _getdpsystemdata(system::AbstractSystem)
    cell = np.array(stack(map(cell_vectors(system)) do vec
        ustrip.(u"Å", vec)
    end; dims=1))
    coords = np.array(stack(map(position(system)) do vec
        ustrip.(u"Å", vec)
    end; dims=1))

    atsymbols = atomic_symbol(system)
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

    dpsys = dpdata.System(;data)

    return dpsys
end
function getdpsystem(systems::Vector{<:AbstractSystem})
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
function split_trainvalidtest_data(
        ls::Py, ratios=(0.7, 0.2, 0.1)
    )
    nframes = length(ls)
    
    @assert sum(ratios) ≈ 1 "Total ratios should add to 1"
    
    index_training = sample(0:(nframes-1), Int(ceil(nframes * ratios[1])); replace=false)
    index_remaining = filter(!in(index_training), 0:(nframes-1))
    if iszero(ratios[3]) # Take all other indices if no test data is required
        index_validation = copy(index_remaining)
        index_testing = Int[]
    else
        index_validation = sample(index_remaining, Int(ceil(nframes * ratios[2])); replace=false)
        index_testing = filter(!in(index_validation), index_remaining)
    end 
    
    # @show length.([index_training, index_validation, index_testing])
    ls_training = ls.sub_system(index_training)
    ls_validation = ls.sub_system(index_validation)
    ls_testing = ls.sub_system(index_testing)

    (; ls_training, ls_validation, ls_testing)
end
