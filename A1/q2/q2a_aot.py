from numba.pycc import CC
import numpy as np

cc = CC("q2a_precompiled")

@cc.export("health_processing", "float64(float64[:,:], float64, float64, float64)")
def health_processing(eps_mat, rho, mu, z_0):
    T, S = eps_mat.shape
    total_periods = 0.0

    for s_ind in range(S):
        z_tm1 = z_0
        # If the health index never falls to or below zero, count T periods.
        count_periods = float(T)
        for t_ind in range(T):
            e_t = eps_mat[t_ind, s_ind]
            z_t = rho * z_tm1 + (1 - rho) * mu + e_t
            if z_t <= 0.0:
                count_periods = float(t_ind)
                break
            z_tm1 = z_t
        total_periods += count_periods

    return total_periods / S

if __name__ == "__main__":
    cc.compile()
