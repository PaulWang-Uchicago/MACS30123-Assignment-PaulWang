import numpy as np
import scipy.stats as sts
import time
from mpi4py import MPI
import q2a_precompiled

def main():
    # Initialize MPI
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # parameters
    T = 4160
    S = 1000
    mu  = 3.0
    z_0 = mu
    
    rho_values = np.linspace(-0.95, 0.95, 200)

    # Rank 0 creates shock matrix
    if rank == 0:
        np.random.seed(0)
        eps_mat = sts.norm.rvs(loc=0.0, scale=1.0, size=(T, S)).astype(np.float64)
    else:
        eps_mat = None

    eps_mat = comm.bcast(eps_mat, root=0)

    # Split rho_values among ranks
    rho_subarrays = np.array_split(rho_values, size)
    local_rho_values = rho_subarrays[rank]

    # Start timing on rank 0
    if rank == 0:
        global_start = time.time()

    # Each process computes over its subset of rho
    local_results = []
    for rho in local_rho_values:
        val = q2a_precompiled.health_processing(eps_mat, rho, mu, z_0)
        local_results.append((rho, val))

    # Convert to a structured array
    local_results = np.array(local_results, dtype=[('rho','f8'), ('avg_periods','f8')])

    # Gather results on rank 0
    all_data = comm.gather(local_results, root=0)

    # Rank 0 processes final results
    if rank == 0:
        all_results = np.concatenate(all_data)

        best_index = np.argmax(all_results['avg_periods'])
        best_rho   = all_results[best_index]['rho']
        best_val   = all_results[best_index]['avg_periods']

        global_end = time.time()
        elapsed = global_end - global_start

        print(f"Grid search done over {len(all_results)} rho-values.")
        print(f"Best rho = {best_rho:.4f}, giving an average of {best_val:.2f} periods.")
        print(f"Total elapsed time: {elapsed:.4f} seconds.")

        import matplotlib.pyplot as plt

        # Extract rho values and average periods from the structured array
        rho_plot = all_results['rho']
        avg_periods_plot = all_results['avg_periods']

        plt.figure(figsize=(8, 6))
        plt.plot(rho_plot, avg_periods_plot, marker='o', linestyle='-', label='Average Periods')
        plt.xlabel('Persistence parameter (rho)')
        plt.ylabel('Average periods to first negative (zₜ ≤ 0)')
        plt.title('Grid Search Results: Average Periods vs. rho')
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        # Save the figure
        plt.savefig("grid_search_results.png")
        plt.show()

if __name__ == "__main__":
    main()
