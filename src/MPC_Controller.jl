# src/MPC_Controller.jl
module MPC_Controller

using Zygote, LinearAlgebra

export compute_optimal_control

# Fast, Zygote-compatible RK4 integrator for the MPC prediction horizon
function rk4_step(state, p, u_ctrl, dt, gp_model, t)
    m, c, k = p
    
    function f(u)
        x, v = u[1], u[2]
        # Calculating the force using our control input + the learned GP
        force = u_ctrl + gp_model(t)
        return [v, (force - c*v - k*x - 0.1*k*x^3) / m]
    end
    
    # Standard RK4 math with Zygote-compatible operations
    k1 = f(state)
    k2 = f(state .+ 0.5 .* dt .* k1)
    k3 = f(state .+ 0.5 .* dt .* k2)
    k4 = f(state .+ dt .* k3)
    
    return state .+ (dt / 6.0) .* (k1 .+ 2.0.*k2 .+ 2.0.*k3 .+ k4)
end

function mpc_cost(control_sequence, current_state, target_state, gp_model, p_base, dt)
    cost = 0.0
    u_val = current_state
    
    for i in 1:length(control_sequence)
        u_ctrl = control_sequence[i]
        
        # Predicting the next state using the fast RK4 step
        u_val = rk4_step(u_val, p_base, u_ctrl, dt, gp_model, 0.0)

        tracking_error = 10.0 * (u_val[1] - target_state[1])^2 
        control_effort = 0.01 * u_ctrl^2
        
        cost += tracking_error + control_effort
    end
    
    return cost
end

function compute_optimal_control(current_state, target_state, gp_model, p_base, horizon, dt)
    u_seq = zeros(horizon)
    learning_rate = 5.0
    
    for iter in 1:50
        # Calculating gradients smoothly without SciML crashes
        grads_tuple = Zygote.gradient(seq -> mpc_cost(seq, current_state, target_state, gp_model, p_base, dt), u_seq)
        grads = grads_tuple[1]
        
        # Safety catch to prevent explosion
        if grads === nothing || any(isnan.(grads)) || any(isinf.(grads))
            break
        end
        
        u_seq .-= learning_rate .* grads

        u_seq .= clamp.(u_seq, -15.0, 15.0)
    end
    
    return u_seq
end

end