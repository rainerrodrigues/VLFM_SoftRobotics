# src/LatentForceModel.jl
module LatentForceModel

using Turing
using DifferentialEquations
using SciMLSensitivity
using LinearAlgebra
using KernelFunctions # For ScaleTransform and SqExponentialKernel
using ..PhysicsPriors

export build_vlfm_model

# The Turing model infers the unknown physical parameters AND the unmodeled noise
@model function build_vlfm_model(times, observed_positions, control_inputs)
    # Priors for unknown physical parameters
    # Soft robots have highly uncertain stiffness and damping that change over time
    c ~ Uniform(0.1, 1.5)  # Damping must be positive but not massive
    k ~ Uniform(0.5, 5.0)  # Stiffness must be within physical reality
    m = 1.0                       # Mass is assumed known for model identifiability

    # Higher lengthscale (ℓ) makes the latent force smoother
    ℓ ~ Uniform(0.5, 3.0) 
    σ_gp ~ Uniform(0.01, 0.5)

    # Defining the optimized kernel
    kernel = σ_gp^2 * SqExponentialKernel() ∘ ScaleTransform(1.0 / ℓ)

    K = kernelmatrix(kernel, times) + 1e-3 * I
    f_vec ~ MvNormal(zeros(length(times)), Symmetric(K))

    p_vec = [m, c, k]

    # Setting up the continuous-time Forward Pass
    u0 = [observed_positions[1], 0.0] # Initial state: [position, velocity]
    
    function step_dynamics(u_state, p, t)
        mass, damp, stiff = p
        x, v = u_state[1], u_state[2]

        # Get the latent force from the GP at time t
        # f_vec(t) is sampled by Turing as part of the inference
        idx = min(searchsortedlast(times, t), length(f_vec))
        idx = max(1, idx)
        force_residual = f_vec[idx]
        
        # Matching the current time 't' to the correct control input
        idx = min(searchsortedlast(times, t), length(control_inputs))
        idx = max(1, idx)
        u_ctrl = control_inputs[idx]
        
        dx = v
        # Nominal Physics + Learned Parameters
        dv = (u_ctrl + force_residual - damp*v - stiff*x - 0.1*stiff*x^3) / mass
        return [dx, dv]
    end

    prob = ODEProblem(step_dynamics, u0, (times[1], times[end]), p_vec)
    
    # Solving the ODE (Zygote-compatible)
    sol = solve(prob, Tsit5(), saveat=times, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()),reltol=1e-3, abstol=1e-3)
    
    # Conditioning the model on real-world observations (The Likelihood)
    obs_noise ~ InverseGamma(2, 3)
    if sol.retcode == ReturnCode.Success && length(sol.u) == length(times)
        for i in 1:length(times)
            # We observe the position (index 1 of the state vector)
            predicted_pos = sol.u[i][1]
            observed_positions[i] ~ Normal(predicted_pos, obs_noise)
        end
    else
        # Rejecting mathematically impossible trajectories that break the ODE solver
        Turing.@addlogprob! -Inf
    end
end

end 