# DeepMD

[![Build Status](https://github.com/rashidrafeek/DeepMD.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/rashidrafeek/DeepMD.jl/actions/workflows/CI.yml?query=branch%3Amain)

A helper package to work with [deepmd-kit](https://github.com/deepmodeling/deepmd-kit)
in Julia using PythonCall.

It is assumed that deepmd-kit is already installed in an existing conda 
environment. This conda environment needs to set using CondaPkg.jl based on the
instructions provided [here](https://github.com/JuliaPy/CondaPkg.jl#conda-environment-path)
before `using DeepMD`.
