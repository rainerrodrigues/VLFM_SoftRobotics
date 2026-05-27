using Plots
using StatsPlots
using Distributions
using Plots.PlotMeasures

println("Generating Figure 3: Posterior Density and Trace Plots...")

# Extracting the raw sample arrays from the MCMC chain
c_samples = vec(Array(final_chain["c"]))
k_samples = vec(Array(final_chain["k"]))

# Defining the exact Priors and True Values used in our experiment
c_prior = Uniform(0.1, 1.5)
c_true  = 0.7

k_prior = Uniform(0.5, 5.0)
k_true  = 3.0

# Applying LaTeX typography settings
default(
    fontfamily="Computer Modern", 
    grid=true, gridalpha=0.2, framestyle=:box,
    titlefontsize=12, guidefontsize=11, tickfontsize=9, legendfontsize=9,
    linewidth=1.5
)

# --- DAMPING (c) ---
# Density Plot
p1 = density(c_samples, label="Posterior (NUTS)", fill=true, fillalpha=0.3, color=:blue)
plot!(p1, x -> pdf(c_prior, x), xlims=(0.0, 1.6), label="Prior (Uniform)", color=:orange, linewidth=2)
vline!(p1, [c_true], label="True Value (0.7)", color=:black, linestyle=:dash, linewidth=2)
title!(p1, "Damping Parameter (c)\nDensity")
ylabel!(p1, "Density")

# Trace Plot
p3 = plot(c_samples, label="Markov Chain", color=:blue, alpha=0.7)
hline!(p3, [c_true], label="True Value", color=:black, linestyle=:dash, linewidth=2)
title!(p3, "Trace Plot")
xlabel!(p3, "Iteration")
ylabel!(p3, "Sample Value")

# --- STIFFNESS (k) ---
# Density Plot
p2 = density(k_samples, label="Posterior (NUTS)", fill=true, fillalpha=0.3, color=:green)
plot!(p2, x -> pdf(k_prior, x), xlims=(0.0, 5.5), label="Prior (Uniform)", color=:orange, linewidth=2)
vline!(p2, [k_true], label="True Value (3.0)", color=:black, linestyle=:dash, linewidth=2)
title!(p2, "Stiffness Parameter (k)\nDensity")

# Tracing Plot
p4 = plot(k_samples, label="Markov Chain", color=:green, alpha=0.7)
hline!(p4, [k_true], label="True Value", color=:black, linestyle=:dash, linewidth=2)
title!(p4, "Trace Plot")
xlabel!(p4, "Iteration")

# Layout: Top row is Density (p1, p2), Bottom row is Trace (p3, p4)
fig3 = plot(p1, p2, p3, p4, 
    layout=(2, 2), 
    size=(900, 650), 
    margin=6Plots.mm,
    left_margin=8Plots.mm
)

savefig(fig3, "figure_3_posteriors.pdf")
savefig(fig3, "figure_3_posteriors.png")

println("Saved 'figure_3_posteriors.pdf' successfully!")