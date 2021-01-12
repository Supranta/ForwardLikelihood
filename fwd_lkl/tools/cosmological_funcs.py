import numpy as np
from .utils import to_cartesian, v_field

#Define Constants
speed_of_light = 299792
c = speed_of_light

def z_cos(r_hMpc, cosmo_pars):
    Omega_m, Omega_L = cosmo_pars
    q0 = Omega_m/2.0 - Omega_L
    return (1.0 - np.sqrt(1 - 2*r_hMpc*100*(1 + q0)/speed_of_light))/(1.0 + q0)

def z_pred(r, r_hat, cosmo_pars, interpolating_funcs, V_ext, beta):
    cartesian_pos = r.reshape(len(r), 1) * r_hat
    v_field_arr = v_field(cartesian_pos, interpolating_funcs).T
    v_ext_r = np.sum(r_hat * V_ext, axis=1)
    v_r = np.sum(r_hat * v_field_arr, axis=1)
    return (1.0 + z_cos(r, cosmo_pars))*(1.0 + (beta * v_r + v_ext_r)/speed_of_light ) - 1.0

def z_obs(r_hMpc, V):
    return (r_hMpc*100 + V)/speed_of_light

def r_from_mu(mu, cosmo_pars=[0.3,0.7]):
    dL = 10**(mu/5.0 - 5.0)
    r_hMpc = dL
    for i in range(4):
        r_hMpc = dL/(1.0 + z_cos(r_hMpc, cosmo_pars))
    return r_hMpc

def r_from_DA(dA, cosmo_pars=[0.3,0.7]):
    r_hMpc = dA
    for i in range(4):
        r_hMpc = dA * (1.0 + z_cos(r_hMpc, cosmo_pars))
    return r_hMpc
