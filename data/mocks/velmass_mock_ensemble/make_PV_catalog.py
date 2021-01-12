import numpy as np
import sys

sys.path.append('../../..')
from fwd_lkl.tools.cosmological_funcs import z_cos, speed_of_light

mock_num = sys.argv[1]
savedir = 'mock'+mock_num

print("Reading halo catalog...")
halos = np.load(savedir+'/halos.npy')
print("Total halos: %d"%(len(halos)))
halos = halos[halos['PID']==-1]
print("Total parent halos: %d"%(len(halos)))

X = halos['X']
Y = halos['Y']
Z = halos['Z']

VX = halos['VX']
VY = halos['VY']
VZ = halos['VZ']

from astropy.coordinates import cartesian_to_spherical

coord = cartesian_to_spherical(x=X, y=Y, z=Z)

r_true = np.array(coord[0])
DEC    = coord[1].deg
RA     = coord[2].deg

pos = np.array([X, Y, Z])
vel = np.array([VX, VY, VZ])

def z_obs(pos, vel):
    r = np.linalg.norm(pos, axis=0)
    r_hat = pos / r
    V_r = np.sum(vel * r_hat, axis=0)
    z_cos_arr = z_cos(r, [0.315, 0.685])
    return z_cos_arr + (1. + z_cos_arr) * V_r / speed_of_light

z_obs_arr = z_obs(pos, vel)

def r2mu(r):
    return 5. * np.log10(r) + 25.

M = -19.

delta_mu = 5. / np.log(10) * 0.1
mu_true = r2mu(r_true)
mu_obs  = mu_true + delta_mu * np.random.normal(size=len(mu_true))
m_obs   = M + mu_obs

m_select = (m_obs < 16.5)
N_tot = np.sum(m_select)
select_2k = np.random.choice(np.arange(N_tot), size=2000, replace=False)

mu = mu_obs[m_select][select_2k]
e_mu = delta_mu * np.ones(len(mu))
RA_select = RA[m_select][select_2k]
DEC_select = DEC[m_select][select_2k]
z_obs_select = z_obs_arr[m_select][select_2k]

import pandas as pd

PV_df = pd.DataFrame()

PV_df['mu'] = mu
PV_df['e_mu'] = e_mu
PV_df['RA'] = RA_select
PV_df['DEC'] = DEC_select
PV_df['zCMB'] = z_obs_select

PV_df.to_csv(savedir+'/PV_mock_2k.csv')
