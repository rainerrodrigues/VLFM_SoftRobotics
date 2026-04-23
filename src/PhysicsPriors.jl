# src/PhysicsPriors.jl
module PhysicsPriors

using DifferentialEquations

export soft_segment_dynamics, simulate_nominal_physics

# The nominal (simplified) physics of a soft robotic segment
# modeled as a nonlinear spring-mass-damper system.
function soft_segment_dynamics(du, u, p, t)
    # States: u[1] = position (strain), u[2] = velocity
    # Parameters: p[1] = mass, p[2] = damping, p[3] = stiffness, p[4] = control input + GP Force
    m, c, k, f_ext = p
    
    x, v = u[1], u[2]
    
    dx = v
    dv = (f_ext(t) - c*v - k*x - 0.1*k*x^3) / m
    return [dx, dv]
end

function simulate_nominal_physics(u0, tspan, p)
    prob = ODEProblem(soft_segment_dynamics, u0, tspan, p)
    # Tsit5 is an efficient non-stiff solver standard in SciML
    return solve(prob, Tsit5(), saveat=0.01) 
end

end 