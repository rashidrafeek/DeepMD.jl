# DeepMD

[![Build Status](https://github.com/rashidrafeek/DeepMD.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/rashidrafeek/DeepMD.jl/actions/workflows/CI.yml?query=branch%3Amain)

A Julia interface to [DeePMD-kit](https://github.com/deepmodeling/deepmd-kit)
using PythonCall.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/rashidrafeek/DeepMD.jl")
using DeepMD
```

Requires Julia 1.10+. Python and CPU dependencies install automatically on first
use; no separate conda installation is needed. The first load requires internet.
For GPU use, configure a compatible environment through
[PythonCall/CondaPkg](https://juliapy.github.io/PythonCall.jl/stable/pythoncall/#Configuration).
Upstream Linux wheels may print CUDA probe messages during successful CPU use.

## Evaluate a model

Add the example's dependencies with
`Pkg.add(["AtomsBase", "AtomsCalculators", "Unitful"])`.

```julia
using DeepMD, AtomsBase, AtomsCalculators, Unitful

atoms = [Atom(:O, [0., 0., 0.]u"Å"),
         Atom(:H, [0.96, 0., 0.]u"Å"),
         Atom(:H, [-0.24, 0.93, 0.]u"Å")]
box = [[10., 0., 0.], [0., 10., 0.], [0., 0., 10.]] .* u"Å"
system = FlexibleSystem(atoms, box, (true, true, true))
potential = DeepPotential("model.pb") # your trained model
result = AtomsCalculators.energy_forces_virial(system, potential)
result.energy  # eV
result.forces  # vector of 3-vectors, eV/Å
result.virial  # 3×3 matrix, eV; NOT volume-normalized stress
```

Inputs accept any Unitful length unit. Systems must be fully periodic or
nonperiodic, and all elements must appear in the model's type map.

Legacy `DeepMD.energy_and_force` returns `(e, f)` in hartree and hartree/bohr.
`DeepMD.virial_stress` returns a virial in hartree, ordered
`(xx, yy, zz, zy, zx, yx)`.

## Data conversion and splitting

```julia
unlabeled = getdpsystem(system)
frames = dpdata.LabeledSystem("data"; fmt="deepmd/npy")
single_frame = DPLabeledSystem(frames.sub_system(0))
position(single_frame, :)  # Å
single_frame[:energy]     # eV
haskey(single_frame, :virial) # virial and force labels may be absent

using Random
parts = DeepMD.split_trainvalidtest_data(frames; rng=MersenneTwister(42))
parts.ls_training.to_deepmd_npy("training")
```

Use `shuffle=false` for contiguous splits. For correlated MD data, prefer
separate trajectories or sufficiently separated blocks.

QE SCF readers return energy, forces, and optional virial, rotating geometry
and labels together:

```julia
labels = DeepMD.getlabeledsystemdp_qescf("scf.out"; inname="scf.in")
labels = DeepMD.getdplabeledsystem_qescf("scf.out", system)
```
