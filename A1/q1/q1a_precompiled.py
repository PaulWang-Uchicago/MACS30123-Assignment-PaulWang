import numpy as np
import scipy.stats as sts
import time
import q1a_aot  # the precompiled module

# Set model parameters
rho = 0.5
mu = 3.0
sigma = 1.0
z_0 = mu

# Set simulation parameters
S = 1000       # number of lives to simulate
T = 4160       # number of periods for each simulation
np.random.seed(25)

# Draw all idiosyncratic random shocks
eps_mat = sts.norm.rvs(loc=0, scale=sigma, size=(T, S)).astype(np.float64)

# Run and time the simulation using the precompiled function
start_time = time.time()
z_mat = q1a_aot.simulate_z_mat(eps_mat, rho, mu, z_0, T, S)
end_time = time.time()
elapsed = end_time - start_time
print(f"Elapsed time with ahead-of-time compiled function: {elapsed:.4f} seconds")
