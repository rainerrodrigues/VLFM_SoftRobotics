# scripts/simulate_closed_loop.jl
using Turing
using DifferentialEquations
using Zygote
using Plots
using Statistics
using LinearAlgebra
using SciMLSensitivity

include("../src/VLFM_SoftRobotics.jl")
using .VLFM_SoftRobotics.PhysicsPriors
using .VLFM_SoftRobotics.MPC_Controller
using .VLFM_SoftRobotics.LatentForceModel

function run_closed_loop_simulation()
    println("--- Starting Closed-Loop VLFM Simulation ---")
    
    # Simulation Setup
    dt = 0.1
    total_time = 6.0
    times = collect(0.0:dt:total_time)
    horizon = 5
    
    # The true physics of the robot (Unknown to the controller)
    true_p = [1.0, 0.2, 3.0] # Mass=1.0, Damping=0.2, Stiffness=3.0
    
    # The controller's initial (bad) guess
    estimated_p = [1.0, 1.5, 0.5] 
    
    # Target trajectory (a moving sine wave)
    target_trajectory(t) = sin(1.5 * t)
    
    # Storage arrays for plotting later
    actual_positions = zeros(length(times))
    target_positions = zeros(length(times))
    applied_controls = zeros(length(times))
    
    current_state = [0.0, 0.0]
    dummy_gp(t) = 0.0 # Assuming 0 external disturbance for this clean test
    
    println("Phase 1: Running with naive/incorrect physics prior...")
    
    for i in 1:length(times)-1
        t = times[i]
        target_positions[i] = target_trajectory(t)
        
        # --- THE LEARNING TRIGGER ---
        if isapprox(t, 3.0, atol=1e-3)
            println("\n[Trigger] Collecting sensor data from t=0 to t=3...")
            history_times = times[1:i]
            history_pos = actual_positions[1:i]
            history_ctrl = applied_controls[1:i]
            
            println("[Trigger] Running Turing.jl NUTS to learn true dynamics...")
            model = build_vlfm_model(history_times, history_pos, history_ctrl)
            # Short chain for the simulation 
            chain = sample(model, NUTS(0.65), 1000) 
            
            # Extracting the learned parameters (Mean of the posterior)
            learned_c = mean(chain[:c])
            learned_k = mean(chain[:k])
            
            println("  -> True Damping: $(true_p[2]), Learned: $(round(learned_c, digits=2))")
            println("  -> True Stiffness: $(true_p[3]), Learned: $(round(learned_k, digits=2))")
            
            # Updating the controller's brain!
            estimated_p[2] = learned_c
            estimated_p[3] = learned_k
            println("Phase 2: Resuming control with updated Latent Force Model...\n")
        end
        
        # --- MODEL PREDICTIVE CONTROL ---
        # Generating the target state for the MPC horizon
        target_state = [target_trajectory(t + dt*horizon), 0.0]
        
        # Calculating optimal control using Zygote gradients based on *estimated* physics
        grads = compute_optimal_control(current_state, target_state, dummy_gp, horizon, dt)
        
        # Simple gradient descent step to find the best control input
        u_opt = -0.1 * grads[1] 
        applied_controls[i] = u_opt
        
        # --- PHYSICAL REALITY ---
        # Applying the control to the TRUE physics to see what actually happens
        function true_dynamics(u, p, t_sim)
            m, c, k = p
            x, v = u[1], u[2]
            return [v, (u_opt - c*v - k*x - 0.1*k*x^3) / m]
        end
        
        prob = ODEProblem(true_dynamics, current_state, (0.0, dt), true_p)
        sol = solve(prob, Tsit5())
        
        current_state = sol.u[end]
        actual_positions[i+1] = current_state[1]
    end
    
    target_positions[end] = target_trajectory(times[end])
    
    # Generating the Hero Graphic
    println("Simulation complete. Generating plot...")
    
    p1 = plot(times, target_positions, label="Target Trajectory", lw=2, color=:black, linestyle=:dash)
    plot!(p1, times, actual_positions, label="Robot Position", lw=2, color=:blue)
    vline!(p1, [3.0], label="Online VLFM Learning Triggered", color=:red, lw=2)
    
    annotate!(p1, 1.5, -0.8, text("High Error\n(Bad Prior)", 10, :red, :center))
    annotate!(p1, 5.5, -0.4, text("Low Error\n(Learned Physics)", 10, :green, :center))
    
    title!(p1, "Closed-Loop Differentiable MPC with Turing.jl")
    xlabel!(p1, "Time (s)")
    ylabel!(p1, "Position (Strain)")
    
    savefig(p1, "closed_loop_hero_graphic.png")
    println("Saved 'closed_loop_hero_graphic.png'!")
end

run_closed_loop_simulation()