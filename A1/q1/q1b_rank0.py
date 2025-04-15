import numpy as np
import scipy.stats as sts
import time
import q1a_aot  # the precompiled module
from mpi4py import MPI

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

# Set model parameters
rho = 0.5
mu = 3.0
sigma = 1.0
z_0 = mu

# Set simulation parameters
group = 1000   # Total number of lives to simulate
T = 4160         # Number of periods for each simulation

# Determine the number of simulations per process
group = group // size
if rank < group % size:
    group += 1

np.random.seed(rank)

# Each process creates its own shock matrix for its portion
eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, group)).astype(np.float64)

start_time = time.time()
local_results = q1a_aot.simulate_z_mat(eps_mat, rho, mu, z_0, T, group)
end_time = time.time()
local_elapsed = end_time - start_time

# Replace the reduction operation with MPI.SUM
total_elapsed = comm.reduce(local_elapsed, op=MPI.SUM, root=0)

if rank == 0:
    mean_elapsed = total_elapsed / size
    print(f"Average elapsed time (across processes): {mean_elapsed:.4f} seconds")