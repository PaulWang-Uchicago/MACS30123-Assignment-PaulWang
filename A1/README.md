# Assignment 1
**Name:** Paul Wang  
**ID:** zw2685

---

## Question 1

### (1.a) Performance Comparison: Original vs. Ahead-of-Time Compiled Function

In this section, two versions of the code are compared:

- **Original Version:**  
  The script is executed as provided, without any modifications for performance.  
  **Script:** [q1a_original.py](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q1/q1a_original.py)

- **Pre-Compiled Version with AOT:**  
  A portion of the code is refactored into a separate function and pre-compiled ahead-of-time using Numba to boost performance.  
  **Scripts:**  
  - [q1a_precompiled.py](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q1/q1a_precompiled.py)  
  - [q1a_aot.py](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q1/q1a_aot.py)

- **Results:**  
  Running both versions on the Midway cluster produced the following timing measurements (as shown in [q1a.out](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q1/q1a.out)):
  - **Without AOT compilation:** 2.9518 seconds  
  - **With AOT compilation:** 0.0313 seconds

  This significant speedup clearly demonstrates the benefit of pre-compiling the computationally intensive function with Numba.

- **Execution Script:**  
  The job was submitted using the SBATCH file: [q1a.sh](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q1/q1a.sh).

### (1.b) Timing 20 Simulation Runs

For this part, a series of 20 simulation runs was executed using a script designed for recording the time at rank 0.

- **Script Used:**  
  [q1b_rank0.py](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q1/q1b_rank0.py) and [q1b_plot.py](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q1/q1b_plot.py)

- **Job Submission Script:**  
  The simulations were run via the SBATCH file: [q1b.sh](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q1/q1b.sh). Timing data for the 20 runs was recorded in [q1b_times.out](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q1/q1b_times.out).

- **Visualization:**  
  A plot summarizing the timing data for the 20 simulation runs is available here: [q1b_plot.png](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q1/q1b_plot.png).

### (1.c) Discussion of Speedup

The observed speedup is not linear. According to Amdahl’s Law, even a small portion of sequential code can limit the maximum achievable speedup regardless of the number of cores used.

---

## Question 2

This question uses an embarrassingly parallel grid search to find the value of $\rho$ that maximizes the average number of periods until the first occurrence of a non-positive health value. The procedure involves running the same mixture of shocks across all simulation runs while precompiling the computationally intensive portions of the code using Numba.

### (2.a) Grid Search Setup and Parallel Processing

- **Code Implementation:**  
  The grid search is implemented in [q2a.py](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q2/q2a.py).  
  In this script, $\rho$ is varied over a designated range (from -0.95 to 0.95) using 200 grid points.  
  Each simulation run is executed on a separate process (using MPI) to leverage embarrassingly parallel processing.  
  The computationally intensive function is precompiled ahead-of-time using Numba as implemented in [q2a_aot.py](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q2/q2a_aot.py).

- **Parallel Execution:**  
  The simulations were run in parallel on the Midway cluster using the SBATCH file: [q2a.sh](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q2/q2a.sh).

### (2.b) Visualization of Simulation Outcomes

- **Plot Description:**  
  After collecting the simulation results from the grid search, the average number of periods until the first negative or zero health value is plotted, with the $y$-axis representing the average periods and the $x$-axis representing $\rho$.

- **Plot Output:**  
  The visualization is available in [grid_search_results.png](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q2/grid_search_results.png).
    ![grid_search_results.png](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q2/grid_search_results.png)
  The [q2a.py](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q2/q2a.py) script contains the code for this analysis.

### (2.c) Optimal Parameter and Performance Metrics

- **Optimal Parameter:**  
  The grid search identified the best value as **$\rho = -0.0334$**, yielding an average of **754.25** periods until the first occurrence of $H \leq 0$.

- **Performance:**  
  The total elapsed time for the grid search when run in parallel was **0.0914 seconds**.  

- **Output Details:**  
  Detailed results and timing data are available in [q2a.out](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q2/q2a.out).

---

## Question 3

### (3.a) GPU Computation of NDVI and CPU Comparison

For this portion of the assignment, we compute the NDVI (Normalized Difference Vegetation Index) from **bands 4 and 5** of a provided Landsat scene.

- **Code:**  
  The GPU-based calculation is implemented in [q3a.py](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q3/q3a.py).

- **Comparison with CPU:**  
  Execution times for the GPU version are compared against a CPU-only version for computing NDVI over the entire scene.

- **Visualization:**  
  The resulting NDVI map is shown in [ndvi_gpu.png](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q3/ndvi_gpu.png), which closely matches the reference figure provided in the assignment prompt.
  ![ndvi_gpu.png](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q3/ndvi_gpu.png)

### (3.b) Timing Results and Execution Details

- **Code and Output Files:**  
  Both GPU and CPU timing results are captured in the output file [Q3A.out](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q3/Q3A.out). GPU NDVI computation time is **0.063545 seconds**, while 
  CPU NDVI computation time is **0.034748 seconds**.
 
  The SLURM job submission script used is [q3a.sh](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q3/q3a.sh).

- **Findings:**  
  For small data sizes, the GPU can be slower because the overhead, such as kernel launch and data transfer between CPU and GPU, consumes more time than the actual computation.

### (3.c) Scaling Experiment

- **Extended Code:**  
  The NDVI computation has been extended and adapted for larger datasets in [q3c.py](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q3/q3c.py).

- **Performance Observations:**  
  Experiments were conducted using various scaling factors with the following results:

  - **Scaling Factor: 20x**  
    - GPU NDVI computation time: **0.188254 seconds**  
    - CPU NDVI computation time: **0.743359 seconds**

  - **Scaling Factor: 50x**  
    - GPU NDVI computation time: **0.500031 seconds**  
    - CPU NDVI computation time: **1.862294 seconds**

  - **Scaling Factor: 100x**  
    - GPU NDVI computation time: **1.010000 seconds**  
    - CPU NDVI computation time: **3.719078 seconds**

  - **Scaling Factor: 150x**  
    - GPU NDVI computation time: **1.579718 seconds**  
    - CPU NDVI computation time: **5.612612 seconds**

  These results are documented in [Q3C.out](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q3/Q3C.out). The data clearly illustrates that as the data size increases, the GPU approach delivers substantial time savings compared to the CPU version.
  
- **Job Submission:**  
  The experiments were executed using the SLURM job submission script [q3c.sh](https://github.com/PaulWang-Uchicago/MACS30123-Assignment-PaulWang/blob/main/A1/q3/q3c.sh).

---
