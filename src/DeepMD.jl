module DeepMD

using PythonCall
using AtomsBase: AbstractSystem, AtomView, element
import AtomsBase: cell_vectors, periodicity, atomic_symbol, velocity,
                  atomic_number, atomic_mass, atomkeys, hasatomkey
using Unitful: @u_str, ustrip, uconvert
using StatsBase: sample
using StaticArrays: SVector, @SVector
# Not compatible with AtomsBase
# using InteratomicPotentials: AbstractPotential               
# import InteratomicPotentials: energy_and_force, virial_stress

const deepmd = PythonCall.pynew()
const dpdata = PythonCall.pynew()
const tf = PythonCall.pynew()
const aseio = PythonCall.pynew()
const np = PythonCall.pynew()
const pycolon =  PythonCall.pynew()
const DeepPotPy = PythonCall.pynew()

function __init__()
    PythonCall.pycopy!(deepmd, pyimport("deepmd"))
    PythonCall.pycopy!(dpdata, pyimport("dpdata"))
    PythonCall.pycopy!(tf, pyimport("tensorflow"))
    PythonCall.pycopy!(aseio, pyimport("ase.io"))
    PythonCall.pycopy!(np, pyimport("numpy"))
    PythonCall.pycopy!(pycolon, pyslice(pybuiltins.None, pybuiltins.None, pybuiltins.None))
    PythonCall.pycopy!(DeepPotPy, pyimport("deepmd.infer").DeepPot)
    pyimport("deepmd.infer") # So that deepmd.infer.DeepPot works
end

export deepmd, dpdata
export DPLabeledSystem
export getdpsystem

include("atomsbase.jl")
include("utils.jl")

end
