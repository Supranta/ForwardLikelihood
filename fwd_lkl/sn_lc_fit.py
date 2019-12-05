"""
This class uses forward likelihood method on simple distance data using a Gaussian likelihood for
P(r|data).
"""
import numpy as np
from fwd_lkl.fwd_lkl import fwd_lkl

class sn_lc_fit(fwd_lkl):
    def __init__(self, v_data, v_field, delta_field, coord_system, vary_sig_v, start_index, rescale_distance, N_POINTS=500):
        super().__init__(v_data, v_field, delta_field, coord_system, vary_sig_v)
        self.mB = v_data[3]
        self.c = v_data[4]
        self.x1 = v_data[5]
        self.e_mB = v_data[6]
        self.e_c = v_data[7]
        self.e_x1 = v_data[8]
        self.start_index = start_index
        self.end_index = start_index + self.num_params()

    def num_params(self):
        N_PARAMS = 4        # M, alpha, beta_sn, sigma_int
        return N_PARAMS

    def pos0(self):
        theta_init_mean = []
        theta_init_spread = []
        if(self.rescale_distance):
            theta_init_mean.append([-18.0, 0.162, 3.126, 0.1])
            theta_init_spread.append([0.01, 0.001, 0.01, 0.01])
        return theta_init_mean, theta_init_spread

    def p_r(self, catalog_theta):
        M, alpha, beta_sn, sigma_int = catalog_theta

        mu = self.mB - M + alpha * self.x1 - beta_sn * self.c_sn
        r, V_r, delta = self.precomputed

        cartesian_pos_r = (np.expand_dims(self.r_hat.T, axis=1)*np.tile(np.expand_dims(r, axis=0),(1,1,3)))
        density_term = (1.0 + delta).T

        delta_d = (r-d)
        return np.exp(-0.5*delta_d*delta_d / sigma_d / sigma_d)*density_term

    def catalog_lnprior(self, catalog_params):
        if(self.rescale_distance):
            htilde = catalog_params
            if(htilde < 0.5):
                return -np.inf
        return 0.0
