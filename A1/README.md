# Assignment 1
**Name:** Paul Wang  
**ID:** zw2685

## Question 1

### (1.a) Performance Comparison: Original vs. Ahead-of-Time Compiled Function

In this section, two versions of the code are compared:

- **Original Version:**  
  The script is executed as provided, without any modifications for performance.  
  **Script:** [q1a_original.py](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q1/q1a_original.py)

- **Pre-Compiled Version with AOT:**  
  A portion of the code is refactored into a separate function and pre-compiled ahead of time using Numba. This version makes use of ahead-of-time (AOT) compilation to speed up execution.  
  **Scripts:**  
  - [q1a_precompiled.py](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q1/q1a_precompiled.py)  
  - [q1a_aot.py](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q1/q1a_aot.py)

- **Results:**  
  Running both versions on the Midway cluster produced the following timing measurements (as shown in [q1a.out](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q1/q1a.out)):
  - **Without AOT compilation:** 2.9518 seconds  
  - **With AOT compilation:** 0.0313 seconds

This significant speedup clearly demonstrates the benefits of pre-compiling the computationally intensive function with Numba.

- **Execution Script:**  
  The job was submitted using the following SBATCH file: [q1a.sh](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q1/q1a.sh).

### (1.b) Timing 20 Simulation Runs

For this part, a series of 20 simulation runs was executed using the script designed for rank 0.

- **Script Used:**  
  [q1b_rank0.py](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q1/q1b_rank0.py), [q1b_plot.py]https://github.com/PaulWang-Uchicago/30123/blob/main/A1/q1/q1b_plot.py). 

- **Job Submission Script:**  
  The simulations were run via the following SBATCH file: [q1b.sh](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q1/q1b.sh). I record the timing data for the 20 simulation runs in [q1b_times.out](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q1/q1b_times.out). 

- **Visualization:**  
  A plot summarizing the timing data for the 20 simulation runs is available here: [q1b_plot.png](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q1/q1b_plot.png).

The plot provides a clear visualization of the run-to-run timing consistency and overall performance improvement achieved with the changes.

### (1.c) 
The speedup is not linear because according to Amdahl’s Law, the presence of even a small fraction of sequential code in a program restricts the maximum achievable speedup, regardless of the number of cores used. 

---

## Question 2

This question uses an embarrassingly parallel grid search to find the value of $\rho$ that maximizes the average number of periods until the first occurrence of a non-positive health value. The procedure involves running the same mixture of shocks across all simulation runs while precompiling the computationally intensive portion of the code using Numba.

### (2.a) Grid Search Setup and Parallel Processing

- **Code Implementation:**  
  The grid search is implemented in [q2a.py](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q2/q2a.py).  
  In this script, $\rho$ is varied over a designated range (-0.95 to 0.95), with 200 grid points.  
  Each simulation run is executed on a separate process to leverage embarrassingly parallel processing (using MPI).  
  The computationally intensive function is precompiled ahead-of-time using Numba as implemented in [q2a_aot.py](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q2/q2a_aot.py).

- **Parallel Execution:**  
  The simulations were run in parallel on the Midway cluster using the provided SBATCH file, [q2a.sh](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q2/q2a.sh).

### (2.b) Visualization of Simulation Outcomes

- **Plot Description:**  
  After collecting the simulation results from the grid search, the average number of periods until the first negative or zero health value was plotted, with the $y$-axis representing the average periods and the $x$-axis representing $\rho$.
  
- **Plot Output:**  
  The resulting visualization is available in [grid_search_results.png](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q2/grid_search_results.png).  
  The same [q2a.py](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q2/q2a.py) file contains the code used for this analysis.

### (2.c) Optimal Parameter and Performance Metrics

- **Optimal Parameter:**  
  The grid search found that the best value is **$\rho$ = -0.0334**, which yields an average of **754.25** periods until the first occurrence of $H \leq 0$.

- **Performance:**  
  The total elapsed time for the grid search when run in parallel was **0.0914 seconds**.  
  This result demonstrates a significant improvement due to parallelization and precompilation, indicating that the embarrassingly parallel approach scaled very well compared to a sequential execution.

- **Output Details:**  
  Detailed results and timing information can be found in [q2a.out](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q2/q2a.out).

---

## Question 3

### (3.a) GPU Computation of NDVI and CPU Comparison

For this portion of the assignment, we compute the NDVI (Normalized Difference Vegetation Index) from **bands 4 and 5** of the provided Landsat scene.

- **Code:**  
  The GPU-based calculation is implemented in [q3a.py](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q3/q3a.py).
- **Comparison with CPU:**  
  I measure and compare the execution time of the GPU version versus a CPU-only version for computing NDVI over the entire scene.
- **Visualization:**  
  The resulting NDVI map is shown in [ndvi_gpu.png](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q3/ndvi_gpu.png), which closely matches the reference figure from the assignment prompt.

### (3.b) Timing Results and Execution Details

- **Code and Output Files:**  
  - GPU timing and CPU timing results are both captured in the output file [Q3A.out](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q3/Q3A.out).  
  - The SLURM job submission script is provided in [q3a.sh](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q3/q3a.sh).
- **Findings:**  
  The experiment shows a noticeable speedup when using the GPU implementation compared to the CPU. Detailed timing data, including GPU configuration, can be found in the output file.

### (3.c) Scaling to Larger Scenes or Multiple Scenes

- **Extended Code:**  
  For larger-scale experiments—such as multiple Landsat scenes or higher-resolution data—the same NDVI computation is adapted in [q3c.py](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q3/q3c.py).  
- **Performance Observations:**  
  The corresponding results are documented in [Q3C.out](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q3/Q3C.out). As scene size (or the number of scenes) increases, the GPU approach continues to demonstrate substantial time savings, though GPU memory management and kernel configuration (thread/block dimensions) become more critical for maintaining efficiency.
- **Job Submission:**  
  The script [q3c.sh](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q3/q3c.sh) was used for automated submission to the cluster.

---
