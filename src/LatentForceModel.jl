using Turing, DifferentialEquations, AbstractGPs, Zygote

# Defining the physical prior (Simplified Soft Robot Segment)
function soft_arm_dynamics!(du, u, p, t)
    # p[1] = damping, p[2] = stiffness, f_t = latent force at time t
    v, x = u[1], u[2]
    f_t = p[3](t) # The GP evaluated at time t
    
    du[1] = -p[1]*v - p[2]*x + f_t  # Acceleration
    du[2] = v                       # Velocity
end

# Defining the Turing Model
@model function vlfm_model(times, observations)
    # Hyperparameters for the physical system
    damping ~ LogNormal(0.0, 1.0)
    stiffness ~ LogNormal(0.0, 1.0)
    
    # GP Hyperparameters for the unknown dynamics (e.g., unmodeled friction/viscoelasticity)
    variance ~ InverseGamma(2, 3)
    lengthscale ~ InverseGamma(2, 3)
    
    # Defining the GP prior
    kernel = variance * SqExponentialKernel() ∘ ScaleTransform(1.0 / lengthscale)
    f_gp ~ GP(kernel)
    
    # Setup ODE with the sampled GP as the forcing function
    u0 = [0.0, 0.0]
    tspan = (minimum(times), maximum(times))
    p = [damping, stiffness, t -> f_gp(t)]
    
    prob = ODEProblem(soft_arm_dynamics!, u0, tspan, p)
    
    # Defining the ODE solver with Zygote-compatible sensitivity (ForwardDiff/ReverseDiff)
    sol = solve(prob, Tsit5(), saveat=times, sensealg=InterpolatingAdjoint(autojacvec=ZygoteVJP()))
    
    # Defining the likelihood of observations given the solved ODE states
    sigma_obs ~ HalfNormal(0.1)
    for i in 1:length(times)
        observations[i] ~ Normal(sol.u[i][2], sigma_obs)
    end
end