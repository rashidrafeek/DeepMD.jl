# DeepMD

[![Build Status](https://github.com/rashidrafeek/DeepMD.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/rashidrafeek/DeepMD.jl/actions/workflows/CI.yml?query=branch%3Amain)

A helper package to work with [deepmd-kit](https://github.com/deepmodeling/deepmd-kit)
in Julia using PythonCall.

## Installation

```julia
using Pkg
Pkg.add(url="https://github.com/rashidrafeek/DeepMD.jl")
using DeepMD
```

Julia 1.10 or newer is required. PythonCall and CondaPkg automatically download
an isolated Python environment, DeePMD-kit 3.1, dpdata, ASE, and the TensorFlow
CPU runtime on first use. No separate Python or conda installation is needed.
The first load needs internet access and can take several minutes.
TensorFlow `.pb` models work on CPU and GPU; the default runtime uses CPU.
The tested Linux DeePMD-kit 3.1.3 wheel prints CUDA-library probe messages
on CPU-only systems; CPU inference is tested without installing CUDA.
GPU use requires a compatible GPU-enabled Python environment configured through
[PythonCall/CondaPkg](https://juliapy.github.io/PythonCall.jl/stable/pythoncall/#Configuration)
before loading DeepMD. PyTorch is not installed by default.

## Evaluate a model

For this example, add `AtomsBase`, `AtomsCalculators`, and `Unitful` to your
project with `Pkg.add(["AtomsBase", "AtomsCalculators", "Unitful"])`.

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

Positions and cell vectors may use any Unitful length unit. Fully periodic and
fully nonperiodic systems are supported; mixed periodicity throws an error.
The model's type map must include every element in the system.
The legacy `DeepMD.energy_and_force` returns `(e, f)` in hartree and hartree/bohr.
`DeepMD.virial_stress` retains its historical six-component virial in hartree,
ordered `(xx, yy, zz, zy, zx, yx)`, without a volume division or sign change.

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

Splits use largest-remainder rounding so counts always sum to the frame count;
ties prefer training, then validation. Zero ratios receive no frames. Use
`shuffle=false` for contiguous blocks instead of random splitting. For correlated
MD trajectories, prefer separate trajectories or sufficiently separated blocks;
a random frame split can overestimate generalization.

QE SCF helpers read energy, forces, and optional stress using dpdata's public
reader. ASE handles input geometry, including its supported QE cell conventions:

```julia
labels = DeepMD.getlabeledsystemdp_qescf("scf.out"; inname="scf.in")
labels = DeepMD.getdplabeledsystem_qescf("scf.out", system)
```

The reader applies dpdata's standard cell rotation to coordinates, forces, and
virials together. QE stress is multiplied by cell volume to produce a virial.

## Development

```julia
using Pkg
Pkg.test("DeepMD")
```

Tests include an upstream, LGPL-licensed model fixture under `test/fixtures` and
synthetic QE input/output fixtures. They exercise real TensorFlow inference and
need no separate model downloads. Linux x86-64 is the CI-tested platform; other
platforms depend on availability of upstream Python wheels. GPU execution is
not covered by the CPU tests.
