"""
Script to do the hmc for joint sampling of the mu_sys and flow parameters.
Philosophy:
- Unlike fit.py, I don't want to make this general. Instead we want this script to do the inference for a very specific case.
- Define an associated class in fwd_lkl/hmc_cov.py
- Still keep the general structure as similar to fit.py as possible.
"""
import numpy as np
import os, sys, time
import h5py as h5

from fwd_lkl.tools.data import process_reconstruction_data
from fwd_lkl.sampler import HMCSampler
from fwd_lkl.dist_cov import DistCov
from fwd_lkl.tools.config_dist_cov import config_io, config_PV_data, config_box, config_mcmc

configfile = sys.argv[1]

N_SAVE = 5
# cosmo_pars = [0.315, 0.685]

output_dir, reconstruction_data_file   = config_io(configfile)
v_data, dist_cov                       = config_PV_data(configfile)
coord_system, box_size, corner, N_grid = config_box(configfile)
N_MCMC, dt, N_LEAPFROG                 = config_mcmc(configfile)

no_cov = False

home_dir                 = os.path.abspath('.')
output_dir               = home_dir+'/'+output_dir
reconstruction_data_file = home_dir+'/'+reconstruction_data_file

delta_field, v_field = process_reconstruction_data(reconstruction_data_file, box_size, corner, N_grid)

PVCatalog = DistCov(v_data, v_field, delta_field, coord_system, dist_cov)

flow_param_cov = np.array([0.04, 20., 20., 20.])**2
flow_param_mass = np.diag(1./flow_param_cov)
flow_theta0 = np.array([1.0, 0.,0.,0.])
if dist_cov is None:
    N_DIM = 4
    mass_matrix = flow_param_mass
    x0 = flow_theta0
else:
    N_DIM = (PVCatalog.N_OBJ + 4)
    mass_matrix = np.zeros((N_DIM,N_DIM))
    mass_matrix[4:,4:] = PVCatalog.dist_cov_inv
    mass_matrix[:4,:4] = flow_param_mass
    mu_sys0 = np.random.multivariate_normal(np.zeros(PVCatalog.N_OBJ), cov=PVCatalog.dist_cov)
    x0 = np.hstack((flow_theta0, mu_sys0))

sampler = HMCSampler(N_DIM, PVCatalog.psi, PVCatalog.grad_psi, mass_matrix)

x = x0
n_accepted = 0

for i in range(N_MCMC):
        print("MCMC Iteration: %d"%(i))
        x, lnprob, acc, KE = sampler.sample_one_step(x, dt, N_LEAPFROG)
        if(acc):
                n_accepted += 1
        print("Acceptance rate: %2.3f"%(n_accepted/(i+1)))
        print("Flow_params: "+str(x[:4]))
        if(i%N_SAVE==0):
            with h5.File(output_dir+'/mcmc_%d.h5'%(i//N_SAVE),'w') as f:
                f['flow_params'] = x[:4]
                """
                if dist_cov is None:
                    f['flow_params'] = x
                else:
                    mu_sys_prior = PVCatalog.mu_sys_prior(x[4:])
                    f['mu_sys_prior'] = mu_sys_prior
                    f['flow_params'] = x[:4]
                    f['mu_sys']      = x[4:]
                """
                f['lnprob']      = lnprob
                f['KE']          = KE
