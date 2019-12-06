import numpy as np
import os, sys, time
from fwd_lkl.fwd_lkl import fwd_lkl, num_flow_params, flow_params_pos0
from fwd_lkl.tools.config import config_fwd_lkl
from fwd_lkl.tools.create_obj import catalog_obj
from fwd_lkl.tools.data import process_reconstruction_data
from fwd_lkl.tools.utils import uncertainty
import emcee

configfile = sys.argv[1]

cosmo_pars = [0.3, 0.7]

NCAT, fit_method,\
        vary_sig_v, output_dir,\
        data_file, coord_system, box_size, corner, N_grid,\
        N_MCMC, N_WALKERS, N_THREADS, \
            catalogs = config_fwd_lkl(configfile)

home_dir = os.path.abspath('.')
output_dir = home_dir+'/'+output_dir
data_file = home_dir+'/'+data_file
delta_field, v_field = process_reconstruction_data(data_file, box_size, corner, N_grid)

num_params = num_flow_params(vary_sig_v)

theta_init_mean, theta_init_spread, flow_model_labels, simple_labels = flow_params_pos0(vary_sig_v)

catalog_objs = []

for i, catalog in enumerate(catalogs):
        if catalog[0]=='simple_gaussian':
                v_data_type, rescale_distance, add_sigma_int, v_data_file = catalog
                v_data_file = home_dir+'/'+v_data_file
                obj = catalog_obj(v_data_type, v_data_file,\
                            num_params,\
                            vary_sig_v,\
                            v_field, delta_field, coord_system,\
                            rescale_distance, add_sigma_int)
        else:
                v_data_type, v_data_file = catalog
                obj = catalog_obj(v_data_type, v_data_file,\
                    num_params,\
                    vary_sig_v,\
                    v_field, delta_field, coord_system)
        catalog_objs.append(obj)
        cat_init_mean, cat_init_spread = obj.pos0()
        num_params += obj.num_params()
        theta_init_mean += cat_init_mean
        theta_init_spread += cat_init_spread
        print(theta_init_mean, theta_init_spread)

N_FLOW_PARAMS = num_flow_params(vary_sig_v)

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

if(fit_method=='mcmc'):
        i = 0
        pos0 = [theta_init_mean + theta_init_spread*np.random.randn(num_params) for i in range(N_WALKERS)]
        sampler = emcee.EnsembleSampler(N_WALKERS, num_params, fwd_lnprob, args=(catalog_objs,), threads=N_THREADS)
        start_time = time.time()
        for result in sampler.sample(pos0, iterations=N_MCMC):
                end_time = time.time()
                print("Current Iteration: "+str(i)+", Time Taken: %2.2f \n"%(end_time - start_time))
                i += 1
                start_time = time.time()
                if(i%10==0):
                    np.save(output_dir+'/chain.npy',sampler.chain[:,:i,:])
        np.save(output_dir+'/chain.npy',sampler.chain)
elif(fit_method=='optimize'):
        from scipy.optimize import fmin_powell
        print(theta_init_mean)
        opt_value = fmin_powell(fwd_objective, theta_init_mean, args=(catalog_objs,), xtol=0.001)
        uncertainty_arr = np.zeros(len(theta_init_mean))
        for i in range(len(theta_init_mean)):
                uncertainty_arr[i] = uncertainty(fwd_objective, opt_value, i, (catalog_objs,), 0.05)
        print(opt_value, uncertainty_arr)
        f = open(output_dir+'/results.txt','w')
        print('Writing results')
        for i in range(N_FLOW_PARAMS):
                f.write(simple_labels[i]+': %2.3f +/- %2.3f \n' %(opt_value[i],uncertainty_arr[i]))
        n = N_FLOW_PARAMS
        for i in range(NCAT):
                f.write('\n Catalog %d: \n'%(i))
                catalog_obj = catalog_objs[i]
                for j in range(catalog_obj.num_params()):
                        f.write('parameter%d : %2.4f +/- %2.4f\n'%(j,opt_value[n+j],uncertainty_arr[n+j]))
                n += catalog_obj.num_params()
        f.close()
