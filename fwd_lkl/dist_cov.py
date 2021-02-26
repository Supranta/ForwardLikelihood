import numpy as np
import jax.numpy as jnp
from jax import grad
from jax.config import config
config.update("jax_enable_x64", True)

from .tools.cosmological_funcs import z_cos, c, r_from_mu
from .tools.utils import direction_vector

cosmo_pars = [0.315, 0.685]

class DistCov():
    def __init__(self, v_data, v_field, delta_field, coord_system, dist_cov=None):
        self.RA           = v_data[0]
        self.DEC          = v_data[1]
        self.z_obs        = v_data[2]
        self.mu           = v_data[3]
        self.e_mu         = v_data[4]

        self.N_OBJ = len(self.RA)

        self.dist_cov     = dist_cov
        if self.dist_cov is not None:
            self.dist_cov_inv = np.linalg.inv(dist_cov)

        self.r_hat = direction_vector(self.RA, self.DEC, coord_system)

        N_POINTS = 500

        V_x_field, V_y_field, V_z_field = v_field
        r = np.linspace(0.01, 198., N_POINTS).reshape(N_POINTS, 1)
        cartesian_pos_r = (np.expand_dims(self.r_hat.T, axis=1)*np.tile(np.expand_dims(r, axis=0),(1,1,3)))

        V_r = (V_x_field(cartesian_pos_r)*np.expand_dims(self.r_hat[0], 1)
        + V_y_field(cartesian_pos_r)*np.expand_dims(self.r_hat[1], 1)
        + V_z_field(cartesian_pos_r)*np.expand_dims(self.r_hat[2], 1)).T

        delta = delta_field(cartesian_pos_r)
        self.precomputed = [r, V_r, delta]

    def d_sigmad(self, mu_sys):
        mu = self.mu + mu_sys
        d = r_from_mu(mu)
        sigma_d = self.e_mu * (np.log(10)/5.) * d
        return d, sigma_d, self.e_mu

    def p_r(self, mu_sys):
        d, sigma_d, e_mu = self.d_sigmad(mu_sys)
        r, V_r, delta = self.precomputed
        cartesian_pos_r = (np.expand_dims(self.r_hat.T, axis=1)*np.tile(np.expand_dims(r, axis=0),(1,1,3)))
        density_term = (1. + delta).T
        delta_mu = 5*jnp.log10(r/d)
        return r * r * jnp.exp(-0.5*(delta_mu/e_mu)**2) * density_term

    def catalog_lnprob(self, flow_params, mu_sys):
        sig_v = 150.
        beta = flow_params[0]

        v_bulk = jnp.array([flow_params[1:]]).reshape((3,1))

        r, V_r, delta = self.precomputed
        v_bulk_r = jnp.sum((v_bulk * self.r_hat), axis=0, keepdims=True)
        z_pred_r = ((1 + (v_bulk_r + beta*V_r)/c)*(1 + z_cos(r, cosmo_pars)) - 1.0)
        delta_z_sig_v = c*(z_pred_r - self.z_obs)/(sig_v)

        pr = self.p_r(mu_sys)
        pr_norm = jnp.trapz(pr, r, axis=0)

        lnprob = jnp.sum(jnp.log(jnp.trapz((1.0/np.sqrt(2*np.pi*sig_v*sig_v))*jnp.exp(-0.5*delta_z_sig_v**2) * pr / pr_norm, axis=0)))

        return lnprob

    def mu_sys_prior(self, mu_sys):
        return -0.5 * jnp.sum(mu_sys.T @ self.dist_cov_inv @ mu_sys)

    def psi(self, theta):
        if self.dist_cov is None:
            flow_params = theta
            prior = 0.
            mu_sys = 0.
        else:
            flow_params = theta[:4]
            mu_sys      = theta[4:]
            prior = self.mu_sys_prior(mu_sys)
        lkl = self.catalog_lnprob(flow_params, mu_sys)
        return -prior - lkl

    def grad_psi(self, theta):
        return grad(self.psi, 0)(theta)
