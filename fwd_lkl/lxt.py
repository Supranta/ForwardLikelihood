"""
This class uses forward likelihood method on the LX-T relation for X-ray clusters
P(r|data).
"""
import numpy as np
from math import pi
from fwd_lkl.fwd_lkl import fwd_lkl
from .tools.cosmological_funcs import r_from_mu

tenpc_in_Mpc = 3.085678e+19
k = 1e+28/tenpc_in_Mpc

class LXT(fwd_lkl):
    def __init__(self, v_data, v_field, delta_field, coord_system, vary_sig_v, start_index, lognormal, N_POINTS=1000):
        super().__init__(v_data, v_field, delta_field, coord_system, vary_sig_v, lognormal)
        self.T    = v_data[3]
        self.flux = v_data[4]
        self.e_T = v_data[5]
        self.start_index = start_index
        self.end_index = start_index + self.num_params()

    def num_params(self):
        N_PARAMS = 3        # AX, BX, sigma_int
        return N_PARAMS

    def pos0(self):
        theta_init_mean = []
        theta_init_spread = []
        theta_init_mean = theta_init_mean+[1.0, 2.0, 0.3]
        theta_init_spread = theta_init_spread + [0.0025, 0.01, 0.0015]
        return theta_init_mean, theta_init_spread

    def d_sigmad(self, catalog_theta):
        AX, BX, sigma_int = catalog_theta

        mu = 2.5*(AX + BX * np.log10(self.T/4.) - np.log10(4*pi*self.flux)) + 5*np.log10(k)
        d = r_from_mu(mu)
        e_mu = np.sqrt((2.5 * (BX / np.log(10)) * self.e_T/self.T)**2 + sigma_int**2)
        sigma_d = e_mu  * (np.log(10)/5.0) * d
        return d, sigma_d, e_mu

    def catalog_lnprior(self, catalog_params):
        AX, BX, sigma_int = catalog_params
        if(sigma_int < 0.0):
            return -np.inf
        return 0.0
