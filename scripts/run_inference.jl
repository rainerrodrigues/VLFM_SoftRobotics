# scripts/run_inference.jl
using Turing
using Plots
using StatsPlots # Required for Turing's built-in MCMC chain plotting
# Including your package's main module
include("../src/VLFM_SoftRobotics.jl")
using .VLFM_SoftRobotics.LatentForceModel

println("--- Setting up VLFM Inference ---")

# Generating Synthetic Data
# For the actual paper, you will replace this block with CSV.read() to load 
# the real positional tracking data from your soft robot's camera/sensors.
println("Generating simulated sensor data...")
times = collect(0.0:0.25:5.0)

# Simulating a control step input at t = 1.0s
control_inputs = [t < 1.0 ? 0.0 : 2.0 for t in times]

# Simulating the robot's physical response (a damped oscillation) with sensor noise
true_positions = [t < 1.0 ? 0.0 : 1.0 - exp(-0.5*(t-1))*cos(2*(t-1)) for t in times]
observed_positions = true_positions .+ 0.05 .* randn(length(times))

# Instantiate the Turing Model
println("Compiling the Latent Force Model...")
model = build_vlfm_model(times, observed_positions, control_inputs)

# target_acceptance: 0.80 or 0.85 is more robust for SciML models
# n_adapts: Ensure the sampler has enough time to learn the geometry
println("Starting NUTS sampling. This will take a moment to compile gradients...")
chain = sample(model, NUTS(500, 0.8), 1000)

# Saving and Interpret the Results
println("\n--- Inference Complete ---")
display(chain) # Prints the statistical summary to the terminal

using Plots.PlotMeasures
default(
    fontfamily="Computer Modern", 
    grid=true, 
    gridalpha=0.2,                
    framestyle=:box,
    linewidth=1.5,
    titlefontsize=14,  # Larger text for better readability in print
    guidefontsize=12,
    tickfontsize=10
)

# Creating FIGURE A: The Physical Parameters
println("Generating Figure A (Physics)...")
physics_chain = final_chain[["c", "k"]]
plot_physics = plot(physics_chain,
    size=(800, 500), # Very spacious for just 2 rows
    margin=8Plots.mm,
    left_margin=12Plots.mm
)
savefig(plot_physics, "figure_A_physics_posterior.pdf")
savefig(plot_physics, "figure_A_physics_posterior.png")

# Creating FIGURE B: The GP Hyperparameters
println("Generating Figure B (Hyperparameters)...")
gp_chain = final_chain[["ℓ", "σ_gp", "obs_noise"]]
plot_gp = plot(gp_chain,
    size=(800, 750), # Spacious for 3 rows
    margin=8Plots.mm,
    left_margin=12Plots.mm
)
savefig(plot_gp, "figure_B_GP_posterior.pdf")
savefig(plot_gp, "figure_B_GP_posterior.png")

println("Done! Check 'figure_A' and 'figure_B' in your folder.")