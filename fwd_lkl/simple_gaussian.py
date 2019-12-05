"""
This class uses forward likelihood method on simple distance data using a Gaussian likelihood for
P(r|data).
"""
import numpy as np
from fwd_lkl.fwd_lkl import fwd_lkl

class simple_gaussian(fwd_lkl):
    def __init__(self, v_data, v_field, delta_field, coord_system, vary_sig_v, start_index, rescale_distance, add_sigma_int, N_POINTS=500):
        super().__init__(v_data, v_field, delta_field, coord_system, vary_sig_v)
        self.rhMpc = v_data[3]
        self.e_rhMpc = v_data[4]
        self.e_V = self.e_rhMpc*100
        self.rescale_distance = rescale_distance
        self.add_sigma_int = add_sigma_int
        self.start_index = start_index
        self.end_index = start_index + self.num_params()

    def num_params(self):
        N_PARAMS = 0
        if(self.rescale_distance):
            N_PARAMS += 1
        if(self.add_sigma_int):
            N_PARAMS += 1
        return N_PARAMS

    def pos0(self):
        theta_init_mean = []
        theta_init_spread = []
        if(self.rescale_distance):
            theta_init_mean.append(1.0)
            theta_init_spread.append(0.01)
        if(self.add_sigma_int):
            theta_init_mean.append(0.1)
            theta_init_spread.append(0.005)
        return theta_init_mean, theta_init_spread

    def d_sigmad(self, catalog_theta):
        if(self.rescale_distance):
            d = catalog_theta[0] * self.rhMpc
        else:
            d = self.rhMpc
        if(self.add_sigma_int):
            if(self.rescale_distance):
                sigma_int = catalog_theta[1]
            else:
                sigma_int = catalog_theta[0]
            sigma_int_d = sigma_int * np.log(10.)/5 * self.rhMpc
            sigma_d = np.sqrt(self.e_rhMpc**2 + sigma_int_d**2)
        else:
            sigma_d = self.e_rhMpc
        return d, sigma_d

    def catalog_lnprior(self, catalog_params):
        if(self.rescale_distance):
            htilde = catalog_params[0]
            if(htilde < 0.5):
                return -np.inf
        if(self.add_sigma_int):
            if(self.rescale_distance):
                sigma_int = catalog_params[1]
            else:
                sigma_int = catalog_params[0]
            if(sigma_int < 0.0):
                return -np.inf
        return 0.0
