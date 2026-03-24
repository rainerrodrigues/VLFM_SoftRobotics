# test/runtests.jl
using Test
using DifferentialEquations
using Zygote

# Including source files
include("../src/PhysicsPriors.jl")
using .PhysicsPriors

include("../src/MPC_Controller.jl")
using .MPC_Controller

@testset "VLFM Soft Robotics Framework Tests" begin

    @testset "Physics Simulator" begin
        # Test if the ODE solves without error
        u0 = [0.0, 0.0]
        tspan = (0.0, 1.0)
        # Dummy force function returning 1.0
        p = (1.0, 0.5, 2.0, t -> 1.0) 
        
        sol = simulate_nominal_physics(u0, tspan, p)
        @test sol.retcode == ReturnCode.Success
        @test length(sol.t) > 1
    end

    @testset "Differentiable MPC via Zygote" begin
        # Test if we can successfully differentiate through the ODE
        current_state = [0.0, 0.0]
        target_state = [1.0, 0.0]
        horizon = 5
        dt = 0.1
        
        # Dummy GP model returning 0.0 mean prediction for testing
        dummy_gp(t) = 0.0 
        
        grads = compute_optimal_control(current_state, target_state, dummy_gp, horizon, dt)
        
        # If Zygote successfully backpropagated through DifferentialEquations.jl, 
        # grads will be an array of the same length as the horizon, not nothing.
        @test !isnothing(grads)
        @test length(grads) == horizon
    end
end