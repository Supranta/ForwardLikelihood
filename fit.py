import numpy as np
import os, sys, time
from fwd_lkl.fwd_lkl import fwd_lkl, num_flow_params, flow_params_pos0
from fwd_lkl.tools.config import config_fwd_lkl
from fwd_lkl.tools.create_obj import create_catalog_obj
from fwd_lkl.tools.data import process_reconstruction_data
from fwd_lkl.tools.fitting import uncertainty, sample, write_catalog_parameters

configfile = sys.argv[1]

cosmo_pars = [0.315, 0.685]

def fwd_lnprob(theta, catalog_objs):
        lnprob = 0
        flow_params = theta[:N_FLOW_PARAMS]
        if(vary_sig_v):
                sig_v, beta, V_x, V_y, V_z = flow_params
                if(sig_v < 0.0):
                        return -np.inf
        for catalog_obj in catalog_objs:
                lnprob += catalog_obj.catalog_lnprob(theta, cosmo_pars)
        return lnprob

def fwd_objective(theta, catalog_objs):
        return -fwd_lnprob(theta, catalog_objs)

NCAT, fit_method,\
        vary_sig_v, output_dir,\
        data_file, coord_system, box_size, corner, N_grid,\
        N_MCMC, N_WALKERS, N_THREADS, \
            catalogs = config_fwd_lkl(configfile)

home_dir = os.path.abspath('.')
output_dir = home_dir+'/'+output_dir
data_file = home_dir+'/'+data_file
delta_field, v_field = process_reconstruction_data(data_file, box_size, corner, N_grid)

N_FLOW_PARAMS = num_flow_params(vary_sig_v)
N = N_FLOW_PARAMS
theta_init_mean, theta_init_spread, flow_model_labels, simple_labels = flow_params_pos0(vary_sig_v)

catalog_objs = []

for i, catalog in enumerate(catalogs):
        v_data_type, rescale_distance, add_sigma_int, v_data_file, lognormal = catalog
        obj = create_catalog_obj(v_data_type, v_data_file,\
            N, vary_sig_v, v_field, delta_field, coord_system, lognormal, rescale_distance, add_sigma_int)
        catalog_objs.append(obj)
        cat_init_mean, cat_init_spread = obj.pos0()
        N += obj.num_params()
        theta_init_mean += cat_init_mean
        theta_init_spread += cat_init_spread

if(fit_method=='mcmc'):
        import emcee

        print('Sampling parameters.....')
        pos0 = [theta_init_mean + theta_init_spread*np.random.randn(N) for i in range(N_WALKERS)]
        sampler = emcee.EnsembleSampler(N_WALKERS, N, fwd_lnprob, args=(catalog_objs,), threads=N_THREADS)

        sample(sampler, pos0, N_MCMC, output_dir)

elif(fit_method=='optimize'):
        print('Finding optimal parameters.....')

        from scipy.optimize import fmin_powell

        opt_value = fmin_powell(fwd_objective, theta_init_mean, args=(catalog_objs,))
        uncertainty_arr = uncertainty(fwd_objective, opt_value, (catalog_objs,), theta_init_spread)
        print(opt_value, uncertainty_arr)

        with open(output_dir+'/results.txt','w') as f:
                print('Writing results to file.....')
                for i in range(N_FLOW_PARAMS):
                        f.write(simple_labels[i]+': %2.3f +/- %2.3f \n' %(opt_value[i],uncertainty_arr[i]))
                n = N_FLOW_PARAMS
                for i in range(NCAT):
                        n = write_catalog_parameters(f, i, n, opt_value, uncertainty_arr, catalog_objs)
