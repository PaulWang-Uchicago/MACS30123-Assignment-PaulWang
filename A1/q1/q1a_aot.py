from numba.pycc import CC
import numpy as np

cc = CC("q1a_aot")

@cc.export("simulate_z_mat", "float64[:,:](float64[:,:], float64, float64, float64, int64, int64)")
def simulate_z_mat(eps_mat, rho, mu, z_0, T, S):
    z_mat = np.zeros((T, S), dtype=np.float64)
    for s_ind in range(S):
        z_tm1 = z_0
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            z_mat[t_ind, s_ind] = z_t
            z_tm1 = z_t
    return z_mat

if __name__ == "__main__":
    cc.compile()
