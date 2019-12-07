import numpy as np
from astropy import units as u
from astropy.coordinates import SkyCoord, Galactic, cartesian_to_spherical

# Coordinate transform helper functions

def to_cartesian(r, angular_pos, coord_system='equatorial'):
    RA, DEC = angular_pos
    c = SkyCoord(ra=RA*u.degree, dec=DEC*u.degree, distance=r)
    if(coord_system == 'equatorial'):
        cartesian_pos = np.array([c.cartesian.x, c.cartesian.y, c.cartesian.z])
    if(coord_system == 'galactic'):
        l, b = c.galactic.l.deg, c.galactic.b.deg
        galactic_c = Galactic(l=l*u.degree, b=b*u.degree, distance=r)
        cartesian_pos = np.array([galactic_c.cartesian.x, galactic_c.cartesian.y, galactic_c.cartesian.z])
    return cartesian_pos

def direction_vector(RA, DEC, coord_system):
    cartesian_pos = to_cartesian(1., [RA, DEC], coord_system)
    return cartesian_pos/np.linalg.norm(cartesian_pos, axis=0)

def v_field(cartesian_pos, V_interpolating_funcs):
    v_x_interp, v_y_interp, v_z_interp = V_interpolating_funcs
    v_x = v_x_interp(cartesian_pos)
    v_y = v_y_interp(cartesian_pos)
    v_z = v_z_interp(cartesian_pos)
    v_field_arr = np.array([v_x, v_y, v_z])
    return v_field_arr

def v_sample_direction_norm(V_samples, coord_system='equatorial'):
    V_norm = np.linalg.norm(V_samples, axis=1)
    V_hat = V_samples/V_norm.reshape((len(V_norm), 1))
    v_x_hat, v_y_hat, v_z_hat = V_hat.T
    _, alpha, beta = cartesian_to_spherical(v_x_hat, v_y_hat, v_z_hat)
    if(coord_system=='equatorial'):
        c = SkyCoord(ra=beta, dec=alpha)
        l, b = c.galactic.l.deg, c.galactic.b.deg
    if(coord_system=='galactic'):
        l, b = beta.deg, alpha.deg
    return V_norm, l, b
