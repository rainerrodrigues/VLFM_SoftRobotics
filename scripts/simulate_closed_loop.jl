# scripts/simulate_closed_loop.jl
using Turing
using DifferentialEquations
using Zygote
using Plots
using Plots.PlotMeasures
using Statistics
using LinearAlgebra
using SciMLSensitivity

include("../src/VLFM_SoftRobotics.jl")
using .VLFM_SoftRobotics.PhysicsPriors
using .VLFM_SoftRobotics.MPC_Controller
using .VLFM_SoftRobotics.LatentForceModel

function run_closed_loop_simulation()
    println("--- Starting Closed-Loop VLFM Simulation ---")
    
    dt = 0.1
    total_time = 6.0
    times = collect(0.0:dt:total_time)
    horizon = 3 # Shorter horizon is faster and perfectly stable for this ODE
    
    # True physics vs Controller's initial (bad) guess
    true_p = [1.0, 0.2, 3.0] 
    estimated_p = [1.0, 1.5, 0.5] 
    
    target_trajectory(t) = sin(1.5 * t)
    
    actual_positions = zeros(length(times))
    target_positions = zeros(length(times))
    applied_controls = zeros(length(times))
    
    current_state = [0.0, 0.0]
    dummy_gp(t) = 0.0 
    
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
            chain = sample(model, NUTS(100, 0.65), 200) # Fast chain for simulation
            
            learned_c = mean(chain[:c])
            learned_k = mean(chain[:k])
            
            println("  -> True Damping: $(true_p[2]), Learned: $(round(learned_c, digits=2))")
            println("  -> True Stiffness: $(true_p[3]), Learned: $(round(learned_k, digits=2))")
            
            # Updating the controller's brain
            estimated_p[2] = learned_c
            estimated_p[3] = learned_k
            println("Phase 2: Resuming control with updated Latent Force Model...\n")
        end
        
        # --- MODEL PREDICTIVE CONTROL ---
        target_state = [target_trajectory(t + dt), 0.0]
        
        # Pass the 'estimated_p' to the controller so it uses the learned physics!
        u_seq = MPC_Controller.compute_optimal_control(current_state, target_state, dummy_gp, estimated_p, horizon, dt)
        
        # Applying the first step of the optimized control sequence
        u_opt = u_seq[1] 
        applied_controls[i] = u_opt
        
        # --- PHYSICAL REALITY ---
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
    
    # Generating the Graphic
    println("Simulation complete. Generating plot...")
    
    default(fontfamily="Computer Modern", grid=true, gridalpha=0.2, framestyle=:box, linewidth=2)
    
    p1 = plot(times, target_positions, label="Target Trajectory", color=:black, linestyle=:dash, size=(800, 500))
    plot!(p1, times, actual_positions, label="Robot Position", color=:blue)
    vline!(p1, [3.0], label="Online VLFM Learning Triggered", color=:red)
    
    annotate!(p1, 1.5, -0.7, text("High Error\n(Bad Prior)", 12, :red, :center))
    annotate!(p1, 5.5, -0.5, text("Low Error\n(Learned Physics)", 12, :green, :center))
    
    title!(p1, "Closed-Loop Differentiable MPC with Turing.jl")
    xlabel!(p1, "Time (s)")
    ylabel!(p1, "Position (Strain)")
    
    savefig(p1, "closed_loop_hero_graphic.pdf")
    savefig(p1, "closed_loop_hero_graphic.png")
    println("Saved 'closed_loop_hero_graphic.pdf'!")
end

run_closed_loop_simulation()