import matplotlib.pyplot as plt

# Data extracted from the scaling study log:
# Number of MPI processes (cores)
cores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 
         11, 12, 13, 14, 15, 16, 17, 18, 19, 20]

# Corresponding total elapsed times (in seconds) for each run
elapsed_times = [
    0.0508,  # 1 core
    0.0177,  # 2 cores
    0.0189,  # 3 cores
    0.0121,  # 4 cores
    0.0091,  # 5 cores
    0.0087,  # 6 cores
    0.0074,  # 7 cores
    0.0041,  # 8 cores
    0.0039,  # 9 cores
    0.0028,  # 10 cores
    0.0029,  # 11 cores
    0.0031,  # 12 cores
    0.0029,  # 13 cores
    0.0021,  # 14 cores
    0.0048,  # 15 cores
    0.0022,  # 16 cores
    0.0019,  # 17 cores
    0.0198,  # 18 cores
    0.0016,  # 19 cores
    0.0013   # 20 cores
]

# Create the plot
plt.figure(figsize=(10, 6))
plt.plot(cores, elapsed_times, marker='o', linestyle='-', color='b')

# Labeling axes and title
plt.xlabel("Number of MPI Processes (Cores)")
plt.ylabel("Total Elapsed Time (seconds)")
plt.title("Scaling Study: Computation Time vs. Number of MPI Processes")
plt.xticks(cores)  # Show integer ticks for each core count
plt.grid(True)


# Save the plot to the current directory as a PNG file
plt.savefig("q1b_plot.png", dpi=300, bbox_inches="tight")