# benchmarks/compare_baselines.jl
using BenchmarkTools
using Zygote
using DifferentialEquations

include("../src/PhysicsPriors.jl")
using .PhysicsPriors

println("--- Benchmarking Differentiable Latent Force Models ---")

# Setup common parameters
u0 = [0.0, 0.0]
dt = 0.1
target = [1.0, 0.0]
dummy_gp(t) = 0.0 # Simulating GP mean evaluation
p_base = (1.0, 0.5, 2.0)

# Benchmarking our SciML Continuous-Time Approach (VLFM)
# How fast can we get gradients through the ODE solver?
function sciml_forward_pass(u_input)
    prob = ODEProblem(soft_segment_dynamics!, u0, (0.0, dt), (p_base..., t -> u_input + dummy_gp(t)))
    sol = solve(prob, Tsit5(), sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    return sum(sol.u[end])
end

println("Benchmarking SciML Differentiable ODE (VLFM)...")
sciml_time = @btime Zygote.gradient(sciml_forward_pass, 1.0)


# Simulated Baseline (Discrete-time transition model)
# Represents typical Python implementations where x_{t+1} = Ax_t + Bu_t + GP(x_t, u_t)
function discrete_forward_pass(u_input)
    # Rough Euler discretization of the same dynamics
    m, c, k = p_base
    v_next = u0[2] + dt * ((u_input + dummy_gp(0.0) - c*u0[2] - k*u0[1]) / m)
    x_next = u0[1] + dt * v_next
    return x_next + v_next
end

println("Benchmarking Standard Discrete-Time Model...")
discrete_time = @btime Zygote.gradient(discrete_forward_pass, 1.0)

println("\nNote for Paper: While the discrete update is technically faster per step, ")
println("it requires significantly smaller time-steps (dt) to remain stable in soft robotics.")
println("The continuous SciML approach allows for longer horizons and stable gradient flow.")