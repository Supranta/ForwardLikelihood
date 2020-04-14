"""
This class uses forward likelihood method on simple distance data using a Gaussian likelihood for
P(r|data).
"""
import numpy as np
from fwd_lkl.fwd_lkl import fwd_lkl
from .tools.cosmological_funcs import r_from_mu

class sn_lc_fit(fwd_lkl):
    def __init__(self, v_data, v_field, delta_field, coord_system, vary_sig_v, add_monopole, start_index, lognormal, N_POINTS=500):
        super().__init__(v_data, v_field, delta_field, coord_system, vary_sig_v, add_monopole, lognormal)
        self.mB = v_data[3]
        self.c_sn = v_data[4]
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
        theta_init_mean = theta_init_mean+[-18.0, 0.162, 3.126, 0.1]
        theta_init_spread = theta_init_spread + [0.01, 0.001, 0.01, 0.005]
        return theta_init_mean, theta_init_spread

    def d_sigmad(self, catalog_theta):
        M, alpha, beta_sn, sigma_int = catalog_theta

        mu = self.mB - M + alpha * self.x1 - beta_sn * self.c_sn
        d = r_from_mu(mu)

        e_mu = np.sqrt(self.e_mB**2 + sigma_int**2)
        sigma_d = e_mu  * (np.log(10)/5.0) * d

        return d, sigma_d, e_mu

    def catalog_lnprior(self, catalog_params):
        M, alpha, beta_sn, sigma_int = catalog_params
        if(sigma_int < 0.0):
            return -np.inf
        return 0.0
