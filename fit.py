import numpy as np
import os, sys, time

os.environ["OMP_NUM_THREADS"] = "1"

from multiprocessing import Pool

from fwd_lkl.fwd_lkl import fwd_lkl, num_flow_params, flow_params_pos0
from fwd_lkl.tools.config import config_fwd_lkl, config_fixed_V_ext
from fwd_lkl.tools.create_obj import create_catalog_obj
from fwd_lkl.tools.data import process_reconstruction_data
from fwd_lkl.tools.fitting import uncertainty, sample, write_catalog_parameters
from fwd_lkl.sampler import MHSampler

configfile = sys.argv[1]

cosmo_pars = [0.315, 0.685]

def fwd_lnprob(theta, catalog_objs):
        lnprob = 0
        flow_params = theta[:N_FLOW_PARAMS]
        if(vary_sig_v):
                sig_v = flow_params[0]
                if(sig_v < 0.0):
                        return -np.inf
        for catalog_obj in catalog_objs:
            lnprob = catalog_obj.catalog_lnprob(theta, cosmo_pars)
        return lnprob

def fwd_objective(theta, catalog_objs):
        return -fwd_lnprob(theta, catalog_objs)

NCAT, fit_method,\
        fix_V_ext, vary_sig_v, add_quadrupole, radial_beta,\
        output_dir, czlow, czhigh,\
        data_file, coord_system, box_size, corner, N_grid,\
        N_MCMC, N_WALKERS, sampler_type,\
            catalogs = config_fwd_lkl(configfile)

if(fix_V_ext):
        V_ext_fixed = config_fixed_V_ext(configfile)
        print("Fixed V_ext: "+str(V_ext_fixed))
home_dir = os.path.abspath('.')
output_dir = home_dir+'/'+output_dir
data_file = home_dir+'/'+data_file
delta_field, v_field = process_reconstruction_data(data_file, box_size, corner, N_grid)

N_FLOW_PARAMS = num_flow_params(fix_V_ext, vary_sig_v, add_quadrupole, radial_beta)
N = N_FLOW_PARAMS
theta_init_mean, theta_init_spread, flow_model_labels, simple_labels = flow_params_pos0(fix_V_ext, vary_sig_v, add_quadrupole, radial_beta)

catalog_objs = []

for i, catalog in enumerate(catalogs):
        v_data_type, rescale_distance, add_sigma_int, v_data_file, lognormal, dist_cov = catalog
        obj = create_catalog_obj(v_data_type, v_data_file, czlow, czhigh,\
            N, fix_V_ext, vary_sig_v, add_quadrupole, radial_beta, v_field, delta_field, coord_system, lognormal, rescale_distance, add_sigma_int, dist_cov)
        if(fix_V_ext):
                obj.set_fixed_V_ext(V_ext_fixed)
        catalog_objs.append(obj)
        cat_init_mean, cat_init_spread = obj.pos0()
        N += obj.num_params()
        theta_init_mean += cat_init_mean
        theta_init_spread += cat_init_spread

if(fit_method=='mcmc'):
    print('Sampling parameters.....')
    if(sampler_type=='emcee'):
        import emcee
        print('Using emcee sampler.....')
        pos0 = [theta_init_mean + theta_init_spread*np.random.randn(N) for i in range(N_WALKERS)] 
        with Pool() as pool:
            sampler = emcee.EnsembleSampler(N_WALKERS, N, fwd_lnprob, args=(catalog_objs,), pool=pool)
            sample(sampler, pos0, N_MCMC, output_dir, catalog_objs)
    if(sampler_type=='mh'):
        print("Using MH sampler...")
        theta = theta_init_mean
        sampler = MHSampler(fwd_lnprob, lnprob_args=(catalog_objs,))
        theta_list = []
        cov = np.diag(np.array([0.04, 20., 20., 20.])**2)
        accepted = 0
        for ii in range(N_MCMC):
            theta, acc, lnprob0 = sampler.sample_one_step(theta, cov)
            theta_list.append(theta)
            if(acc):
                accepted += 1
            if(ii%10==0):
                print("MCMC iteration: %d"%(ii))
                print("theta: "+str(theta))
                print("Acceptance rate: %2.3f"%(accepted/(ii+1)))
                np.save(output_dir+'/mh_chain.npy',np.array(theta_list))
            
elif(fit_method=='optimize'):
        print('Finding optimal parameters.....')

        from scipy.optimize import fmin_powell

        opt_value = fmin_powell(fwd_objective, theta_init_mean, args=(catalog_objs,))
        if(len(theta_init_mean)==1):
                opt_value = [opt_value]
        uncertainty_arr = uncertainty(fwd_objective, opt_value, (catalog_objs,), theta_init_spread)
        print(opt_value, uncertainty_arr)

        with open(output_dir+'/results.txt','w') as f:
                print('Writing results to file.....')
                for i in range(N_FLOW_PARAMS):
                        f.write(simple_labels[i]+': %2.3f +/- %2.3f \n' %(opt_value[i],uncertainty_arr[i]))
                n = N_FLOW_PARAMS
                for i in range(NCAT):
                        n = write_catalog_parameters(f, i, n, opt_value, uncertainty_arr, catalog_objs)
