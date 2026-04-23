# benchmarks/compare_baselines.jl
using BenchmarkTools
using Zygote
using DifferentialEquations
using SciMLSensitivity

include("../src/PhysicsPriors.jl")
using .PhysicsPriors

println("--- Benchmarking Differentiable Latent Force Models ---")

# Setting up common parameters
u0 = [0.0, 0.0]
dt = 0.1
target = [1.0, 0.0]
dummy_gp(t) = 0.0 # Simulating GP mean evaluation

# Using an Array instead of a Tuple for Zygote memory allocation
p_base = [1.0, 0.5, 2.0] 

# Benchmark our SciML Continuous-Time Approach (VLFM)
function sciml_forward_pass(u_input)
    # Using the closure approach exactly like we did in the MPC controller
    function step_dynamics(u_state, p, t)
        m, c, k = p
        x, v = u_state[1], u_state[2]
        
        force = u_input + dummy_gp(t)
        
        dx = v
        dv = (force - c*v - k*x - 0.1*k*x^3) / m
        return [dx, dv]
    end
    
    prob = ODEProblem(step_dynamics, u0, (0.0, dt), p_base)
    sol = solve(prob, Tsit5(), sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    return sum(sol.u[end])
end

println("Benchmarking SciML Differentiable ODE (VLFM)...")
# We limit the samples so the benchmark doesn't take 10 minutes to run
@btime Zygote.gradient(sciml_forward_pass, 1.0) samples=10 evals=1


# Simulated Baseline (Discrete-time transition model)
function discrete_forward_pass(u_input)
    m, c, k = p_base
    v_next = u0[2] + dt * ((u_input + dummy_gp(0.0) - c*u0[2] - k*u0[1]) / m)
    x_next = u0[1] + dt * v_next
    return x_next + v_next
end

println("Benchmarking Standard Discrete-Time Model...")
@btime Zygote.gradient(discrete_forward_pass, 1.0)

println("\nNote for Paper: While the discrete update is technically faster per step, ")
println("it requires significantly smaller time-steps (dt) to remain stable in soft robotics.")
println("The continuous SciML approach allows for longer horizons and stable gradient flow.")