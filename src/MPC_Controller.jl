# src/MPC_Controller.jl
module MPC_Controller

using Zygote, DifferentialEquations, SciMLSensitivity, LinearAlgebra
using ..PhysicsPriors

export compute_optimal_control

function mpc_cost(control_sequence, current_state, target_state, gp_model, p_base, dt)
    cost = 0.0
    u = current_state
    
    # Converting parameters to a standard Array (Vector) instead of a Tuple.
    # Zygote requires an array so it can allocate memory for the gradients.
    p_vec = [p_base[1], p_base[2], p_base[3]] 
    
    for i in 1:length(control_sequence)
        # The specific control action for this time step
        u_ctrl = control_sequence[i]
        
        # Using a "closure" to define the dynamics. 
        # This allows the ODE to access `u_ctrl` and `gp_model` directly from 
        # the surrounding scope without putting them inside the `p` array.
        function step_dynamics(u_state, p, t)
            m, c, k = p
            x, v = u_state[1], u_state[2]
            
            # Combining the control input with the learned GP force
            force = u_ctrl + gp_model(t)
            
            dx = v
            dv = (force - c*v - k*x - 0.1*k*x^3) / m
            return [dx, dv]
        end
        
        # Passing the closure and the clean Array to the solver
        prob = ODEProblem(step_dynamics, u, (0.0, dt), p_vec)
        sol = solve(prob, Tsit5(), sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
        
        # Updating state
        u = sol.u[end]
        
        # Calculating cost
        tracking_error = norm(u - target_state)^2
        control_effort = 0.05 * u_ctrl^2
        cost += tracking_error + control_effort
    end
    
    return cost
end

function compute_optimal_control(current_state, target_state, gp_model, horizon, dt)
    initial_guess = zeros(horizon)
    p_base = [1.0, 0.5, 2.0] # Making sure the base parameters are also an Array
    
    grads = Zygote.gradient(seq -> mpc_cost(seq, current_state, target_state, gp_model, p_base, dt), initial_guess)[1]
    
    return grads
end

end # module