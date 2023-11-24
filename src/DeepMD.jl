module DeepMD

using PythonCall

const deepmd = PythonCall.pynew()
const dpdata = PythonCall.pynew()
const tf = PythonCall.pynew()
function __init__()
    PythonCall.pycopy!(deepmd, pyimport("deepmd"))
    PythonCall.pycopy!(dpdata, pyimport("dpdata"))
    PythonCall.pycopy!(tf, pyimport("tensorflow"))
end

end
