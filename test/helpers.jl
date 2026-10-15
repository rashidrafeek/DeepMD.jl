using AtomsBase
using Unitful
using UnitfulAtomic
using StaticArrays
using Random

function labeled_frames(n=1; virials=true, forces=true, nopbc=false)
    np = DeepMD.np
    data = pydict(
        atom_names=pylist(["H", "O"]), atom_numbs=pylist([2, 1]),
        atom_types=np.array([0, 1, 0]), orig=np.zeros(3),
        cells=np.tile(np.eye(3) * 10, (n, 1, 1)),
        coords=np.tile(np.array([0. 0. 0.; 1. 0. 0.; 0. 1. 0.]), (n, 1, 1)),
        energies=np.arange(n; dtype="float64"), nopbc=nopbc)
    forces && (data["forces"] = np.zeros((n, 3, 3)))
    virials && (data["virials"] = np.tile(np.eye(3), (n, 1, 1)))
    return dpdata.LabeledSystem(; data)
end

function atomic_system(; unit=u"Å", pbc=(true, true, true), symbols=[:H, :O, :H],
        coords=[[0., 0., 0.], [1., 0., 0.], [0., 1., 0.]])
    atoms = [Atom(s, uconvert.(unit, r .* u"Å")) for (s, r) in zip(symbols, coords)]
    box = [uconvert.(unit, v .* u"Å") for v in ([10., 0., 0.], [1., 10., 0.], [0., 0., 10.])]
    return FlexibleSystem(atoms, box, pbc)
end

force_values(forces) = stack([ustrip.(u"eV/Å", v) for v in forces]; dims=1)
