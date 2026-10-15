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
GPU use requires a compatible GPU-enabled Python environment configured through
[PythonCall/CondaPkg](https://juliapy.github.io/PythonCall.jl/stable/pythoncall/#Configuration)
before loading DeepMD. PyTorch is not installed by default.

