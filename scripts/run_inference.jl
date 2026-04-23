# scripts/run_inference.jl
using Turing
using Plots
using StatsPlots # Required for Turing's built-in MCMC chain plotting
# Including your package's main module
include("../src/VLFM_SoftRobotics.jl")
using .VLFM_SoftRobotics.LatentForceModel

println("--- Setting up VLFM Inference ---")

# Generating Synthetic Data
# For your actual paper, you will replace this block with CSV.read() to load 
# the real positional tracking data from your soft robot's camera/sensors.
println("Generating simulated sensor data...")
times = collect(0.0:0.1:5.0)

# Simulating a control step input at t = 1.0s
control_inputs = [t < 1.0 ? 0.0 : 2.0 for t in times]

# Simulating the robot's physical response (a damped oscillation) with sensor noise
true_positions = [t < 1.0 ? 0.0 : 1.0 - exp(-0.5*(t-1))*cos(2*(t-1)) for t in times]
observed_positions = true_positions .+ 0.05 .* randn(length(times))

# Instantiate the Turing Model
println("Compiling the Latent Force Model...")
model = build_vlfm_model(times, observed_positions, control_inputs)

# Executing the No-U-Turn Sampler (NUTS)
# We request 500 samples here for testing. 
# For the final paper, increase this to 2000 or 4000 for smoother distributions.
# The 0.65 is the target acceptance rate for the sampler's step-size adaptation.
println("Starting NUTS sampling. This will take a moment to compile gradients...")
chain = sample(model, NUTS(0.65), 500)

# Saving and Interpret the Results
println("\n--- Inference Complete ---")
display(chain) # Prints the statistical summary to the terminal

# Generating the trace and density plots
println("Generating posterior plots...")
posterior_plot = plot(chain)

# Saving the plot to your directory for inclusion in your LaTeX document
savefig(posterior_plot, "posterior_distributions.png")
println("Saved publication graph as 'posterior_distributions.png'.")