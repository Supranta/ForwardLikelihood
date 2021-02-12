import numpy as np
from .tools.cosmological_funcs import z_cos, c
from .tools.utils import direction_vector
"""
This class uses forward likelihood method on simple distance data using a Gaussian likelihood for
P(r|data).
"""
def num_flow_params(fix_V_ext, vary_sig_v, add_quadrupole, radial_beta):
    num_params = 4
    if(fix_V_ext):
        num_params = 1
    if(vary_sig_v):
        num_params += 1
    if(add_quadrupole):
        num_params += 5
    if(radial_beta):
        num_params += 1
    return num_params

def flow_params_pos0(fix_V_ext, vary_sig_v, add_quadrupole, radial_beta):
    theta_init_mean   = [1., 0., 0., 0.]
    theta_init_spread = [0.005, 5.0, 5.0, 5.0]
    labels = [r'$\beta$', r'$V_x$', r'$V_y$', r'$V_z$']
    simple_labels = ['beta', 'V_x', 'V_y', 'V_z']
    if(fix_V_ext):
        theta_init_mean   = [1.]
        theta_init_spread = [0.005]
        labels = [r'$\beta$']
        simple_labels = ['beta']
    if(radial_beta):
        theta_init_mean.insert(0, 0.)
        theta_init_spread.insert(0, 1.)
        labels.insert(0,r'$\beta_1$')
        simple_labels.insert(0,'beta_slope')
    if(vary_sig_v):
        theta_init_mean.insert(0, 100.)
        theta_init_spread.insert(0, 5.)
        labels.insert(0,r'$\sigma_v$')
        simple_labels.insert(0,'sigma_v')
    if(add_quadrupole):
        for n in range(5):
            theta_init_mean.insert(-1, 0.)
            theta_init_spread.insert(-1, 0.2)
            labels.insert(-1,r'$U_'+str(n)+'$')
            simple_labels.insert(-1,'U_'+str(n))
    return theta_init_mean, theta_init_spread, labels, simple_labels

class fwd_lkl:
    def __init__(self, v_data, v_field, delta_field, coord_system,
                        fix_V_ext, vary_sig_v, add_quadrupole, radial_beta,
                        lognormal, N_POINTS=500):
        self.RA           = v_data[0]
        self.DEC          = v_data[1]
        self.z_obs        = v_data[2]
        self.vary_sig_v   = vary_sig_v
        self.fix_V_ext    = fix_V_ext
        self.add_quadrupole = add_quadrupole
        self.radial_beta  = radial_beta
        self.lognormal    = lognormal
        self.num_flow_params = num_flow_params(fix_V_ext, vary_sig_v, add_quadrupole, radial_beta)
        self.r_hat = direction_vector(self.RA, self.DEC, coord_system)

        V_x_field, V_y_field, V_z_field = v_field
        r = np.linspace(0.01, 198., N_POINTS).reshape(N_POINTS, 1)
        cartesian_pos_r = (np.expand_dims(self.r_hat.T, axis=1)*np.tile(np.expand_dims(r, axis=0),(1,1,3)))

        V_r = (V_x_field(cartesian_pos_r)*np.expand_dims(self.r_hat[0], 1)
        + V_y_field(cartesian_pos_r)*np.expand_dims(self.r_hat[1], 1)
        + V_z_field(cartesian_pos_r)*np.expand_dims(self.r_hat[2], 1)).T

        delta = delta_field(cartesian_pos_r)

        self.precomputed = [r, V_r, delta]

    def set_fixed_V_ext(self, V_ext_fixed):
        self.Vx_fixed = V_ext_fixed[0]
        self.Vy_fixed = V_ext_fixed[1]
        self.Vz_fixed = V_ext_fixed[2]

    def p_r(self, catalog_theta):
        d, sigma_d, e_mu = self.d_sigmad(catalog_theta)
        r, V_r, delta = self.precomputed

        cartesian_pos_r = (np.expand_dims(self.r_hat.T, axis=1)*np.tile(np.expand_dims(r, axis=0),(1,1,3)))
        density_term = (1.0 + delta).T

        if(self.lognormal):
            delta_mu = 5*np.log10(r/d)
            return r * r * np.exp(-0.5*(delta_mu/e_mu)**2) * density_term
        else:
            delta_d = (r-d)
            return r * r * np.exp(-0.5*delta_d*delta_d / sigma_d / sigma_d) * density_term

    def catalog_lnprob(self, params, cosmo_pars):
        flow_params = params[:self.num_flow_params]
        if(self.add_quadrupole):
            quadrupole = flow_params[-5:]
            U_xx, U_yy, U_xy, U_xz, U_yz = quadrupole
            U_zz = -(U_xx + U_yy)
            quadrupole_matrix = np.array([[U_xx, U_xy, U_xz],[U_xy, U_yy, U_yz],[U_xz, U_yz, U_zz]])
            flow_params = flow_params[:-5]
        if(self.vary_sig_v):
            sig_v = flow_params[0]
            flow_params = flow_params[1:]
        else:
            sig_v = 150.
        if(self.radial_beta):
            beta_slope = flow_params[0]
            flow_params = flow_params[1:]

        if(self.fix_V_ext):
            beta = flow_params
            V_x  = self.Vx_fixed
            V_y  = self.Vy_fixed
            V_z  = self.Vz_fixed
        else:
            beta, V_x, V_y, V_z = flow_params

        v_bulk = np.array([V_x, V_y, V_z]).reshape((3,1))
        r, V_r, delta = self.precomputed

        v_bulk_r = np.sum((v_bulk * self.r_hat), axis=0, keepdims=True)
        if(self.add_quadrupole):
            v_bulk_quad = np.sum(self.r_hat*np.matmul(quadrupole_matrix, self.r_hat),axis=0,keepdims=True)*r
            v_bulk_r = v_bulk_r + v_bulk_quad
        N_GAL = self.r_hat.shape[1]
        if(self.radial_beta):
            # print(r.shape, V_r.shape)
            z_pred_r = ((1 + (v_bulk_r + (beta + beta_slope * (r/100.))*V_r)/c)*(1 + z_cos(r, cosmo_pars)) - 1.0)
        else:
            z_pred_r = ((1 + (v_bulk_r + beta*V_r)/c)*(1 + z_cos(r, cosmo_pars)) - 1.0)
        delta_z_sig_v = c*(z_pred_r - self.z_obs)/(sig_v)

        catalog_params = params[self.start_index:self.end_index]
        pr = self.p_r(catalog_params)
        pr_norm = np.trapz(pr, r, axis=0)

        lnprob = np.sum(np.log(np.trapz((1.0/np.sqrt(2*np.pi*sig_v*sig_v))*np.exp(-0.5*delta_z_sig_v**2) * pr / pr_norm, axis=0)))
        if(np.isnan(lnprob)):
            return -np.inf
        return lnprob + self.catalog_lnprior(catalog_params)
