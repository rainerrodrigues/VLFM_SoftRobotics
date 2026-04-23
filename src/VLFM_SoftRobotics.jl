module VLFM_SoftRobotics

# Exporting the submodules so they can be used when someone types `using VLFM_SoftRobotics`
export PhysicsPriors, MPC_Controller

# Including the other files in your src directory
include("PhysicsPriors.jl")
include("MPC_Controller.jl")
include("LatentForceModel.jl")

end