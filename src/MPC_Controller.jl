# src/MPC_Controller.jl
module MPC_Controller

using Zygote, DifferentialEquations, SciMLSensitivity, LinearAlgebra
using ..PhysicsPriors

export compute_optimal_control

# Cost function for a receding horizon
function mpc_cost(control_sequence, current_state, target_state, gp_model, p_base, dt)
    cost = 0.0
    u = current_state
    
    # Rollout the prediction horizon
    for i in 1:length(control_sequence)
        # The total force is the designed control input PLUS the learned GP latent force
        f_total(t) = control_sequence[i] + gp_model(t)
        
        # Updating parameters with the new force function
        p_current = (p_base[1], p_base[2], p_base[3], f_total)
        
        # Predicting next state over small time step dt
        prob = ODEProblem(soft_segment_dynamics!, u, (0.0, dt), p_current)
        
        # We use ZygoteVJP to allow gradients to flow through the ODE solver
        sol = solve(prob, Tsit5(), sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
        u = sol.u[end]
        
        # Cost: Tracking error + Control effort penalty
        tracking_error = norm(u - target_state)^2
        control_effort = 0.05 * control_sequence[i]^2
        
        cost += tracking_error + control_effort
    end
    return cost
end

function compute_optimal_control(current_state, target_state, gp_model, horizon, dt)
    # Initializing a naive control sequence
    initial_guess = zeros(horizon)
    p_base = (1.0, 0.5, 2.0) # mass, damping, stiffness
    
    # Calculating the gradient of the MPC cost function with respect to the control sequence
    # This is normally impossible/extremely slow in Python without custom backward passes
    grads = Zygote.gradient(seq -> mpc_cost(seq, current_state, target_state, gp_model, p_base, dt), initial_guess)[1]
    
    # In a full implementation, you would pass these gradients to an optimizer (like Optim.jl) 
    # to update the initial_guess. For now, we return the gradients to prove differentiability.
    return grads
end

end # module