"""
This class uses forward likelihood method on simple distance data using a Gaussian likelihood for
P(r|data).
"""
import numpy as np
from fwd_lkl.fwd_lkl import fwd_lkl
from .tools.cosmological_funcs import r_from_mu

class TF(fwd_lkl):
    def __init__(self, v_data, v_field, delta_field, coord_system, vary_sig_v, start_index, N_POINTS=500):
        super().__init__(v_data, v_field, delta_field, coord_system, vary_sig_v)
        self.i = v_data[3]
        self.eta = v_data[4]
        self.e_i = v_data[5]
        self.e_eta = v_data[6]
        self.start_index = start_index
        self.end_index = start_index + self.num_params()

    def num_params(self):
        N_PARAMS = 3        # ainv, binv, sigma_int
        return N_PARAMS

    def pos0(self):
        theta_init_mean = []
        theta_init_spread = []
        theta_init_mean = theta_init_mean+[-21., -6., 0.3]
        theta_init_spread = theta_init_spread + [0.01, 0.01, 0.01]
        return theta_init_mean, theta_init_spread

    def d_sigmad(self, catalog_theta):
        ainv, binv, sigma_int = catalog_theta
        mu = self.i - (ainv + binv * self.eta)
        d = r_from_mu(mu)
        e_mu = np.sqrt(self.e_i**2 + (binv*self.e_eta)**2 + sigma_int**2)
        sigma_d = e_mu  * (np.log(10)/5.0) * d
        return d, sigma_d

    def catalog_lnprior(self, catalog_params):
        ainv, binv, sigma_int = catalog_params
        if(sigma_int < 0.0):
            return -np.inf
        return 0.0
