import numpy as np
from .utils import to_cartesian, v_field

#Define Constants
c = 300000

def z_cos(r_hMpc, cosmo_pars):
    Omega_m, Omega_L = cosmo_pars
    q0 = Omega_m/2.0 - Omega_L
    return (1.0 - np.sqrt(1 - 2*r_hMpc*100*(1 + q0)/c))/(1.0 + q0)

def z_pred(r, r_hat, cosmo_pars, interpolating_funcs, V_ext, beta):
    cartesian_pos = r.reshape(len(r), 1) * r_hat
    v_field_arr = v_field(cartesian_pos, interpolating_funcs).T
    v_ext_r = np.sum(r_hat * V_ext, axis=1)
    v_r = np.sum(r_hat * v_field_arr, axis=1)
    return (1.0 + z_cos(r, cosmo_pars))*(1.0 + (beta * v_r + v_ext_r)/c ) - 1.0

def z_obs(r_hMpc, V):
    return (r_hMpc*100 + V)/c
