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
chain = sample(model, NUTS(100,0.65), MCMCThreads(), 200, 4)

# Saving and Interpret the Results
println("\n--- Inference Complete ---")
display(chain) # Prints the statistical summary to the terminal

# Generating the trace and density plots
println("Generating posterior plots...")
using Plots.PlotMeasures # Required for margin adjustments
default(
    fontfamily="Computer Modern", # The standard LaTeX font
    grid=true, 
    gridalpha=0.2,                # Makes the grid lines subtle
    framestyle=:box               # Encloses the plots in a neat box
)

posterior_plot = plot(chain,
    size=(900, 700),              # Larger, cleaner aspect ratio
    dpi=300,                      # 300 DPI is the minimum for journal printing
    linewidth=1.5,                # Slightly thicker lines for the trace plots
    titlefontsize=12,
    guidefontsize=11,
    tickfontsize=9,
    margin=4Plots.mm,             # Adds breathing room between the subplots so text doesn't overlap
    left_margin=8Plots.mm         # Extra room on the left so Y-axis labels aren't cut off
)

# Saving the plot to your directory for inclusion in your LaTeX document
savefig(posterior_plot, "posterior_distributions.png")
savefig(posterior_plot, "posterior_distributions.pdf")
println("Saved publication graph as 'posterior_distributions.png'.")