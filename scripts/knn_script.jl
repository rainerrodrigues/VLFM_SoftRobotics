using Plots
using Plots.PlotMeasures

println("Generating KNN Hyperparameter Tuning Plot...")

# Simulating the Data (K = 1 to 20)
K_values = 1:20

# Minimum is explicitly set at K = 8
mse_values = [
    0.450, 0.380, 0.330, 0.290, 0.260, 0.240, 0.225, 0.210, 0.215, 0.225, 
    0.240, 0.255, 0.270, 0.285, 0.300, 0.315, 0.330, 0.345, 0.360, 0.375
]

# Applying my paper's standard LaTeX typography
default(
    fontfamily="Computer Modern", 
    grid=true, 
    gridalpha=0.3,                
    framestyle=:box,
    linewidth=2,
    titlefontsize=14,
    guidefontsize=12,
    tickfontsize=10,
    legendfontsize=10
)

# Plotting the main MSE line with markers
p = plot(K_values, mse_values, 
    label="Validation MSE", 
    color=:blue, 
    marker=:circle,       # Adding dots at each K value for readability
    markersize=5,
    size=(700, 450),      # Standard half-page width for a paper
    margin=6Plots.mm
)

# Adding the vertical dashed marker at K = 8
vline!(p, [8], 
    label="Optimal K = 8", 
    color=:red, 
    linestyle=:dash, 
    linewidth=2
)

# Formatting the axes
title!(p, "KNN Hyperparameter Sensitivity Analysis")
xlabel!(p, "Number of Neighbors (K)")
ylabel!(p, "Mean Squared Error (MSE)")

# Forcing the X-axis to show neat integer ticks from 1 to 20
xticks!(p, 1:2:20) 

# Saving as a vector graphic
savefig(p, "figure_2_knn_tuning.pdf")
savefig(p, "figure_2_knn_tuning.png")

println("Saved 'figure_2_knn_tuning.pdf' successfully!")