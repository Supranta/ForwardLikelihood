import numpy as np
import sys
from fwd_lkl.fwd_lkl import num_flow_params, flow_params_pos0
from fwd_lkl.tools.config import analyze_fwd_lkl
from fwd_lkl.tools.plotting import chain_plot
import corner
import matplotlib.pyplot as plt

plt.rc('text',usetex=True)
plt.rc('font',family='serif')

configfile = sys.argv[1]
N_BURN_IN = int(sys.argv[2])

NCAT, vary_sig_v, output_dir, \
        plot_chain, plot_lkl, plot_corner, catalogs = analyze_fwd_lkl(configfile)

output_dir = '/Users/boruah/Desktop/research/bayesian_fwd_modelling/pec_vel_fwd_lkl_module/output/'+output_dir

N_FLOW_PARAMS = num_flow_params(vary_sig_v)
print(N_FLOW_PARAMS)
_, _, flow_model_labels, simple_labels = flow_params_pos0(vary_sig_v)
chain = np.load(output_dir+'/chain.npy')
print(vary_sig_v, flow_model_labels)

N_DIM = chain.shape[-1]
samples = chain[:,N_BURN_IN:,:].reshape((-1,N_DIM))

f = open(output_dir+'/results.txt','w')
print('Writing results')
for i in range(N_FLOW_PARAMS):
        f.write(simple_labels[i]+': %2.3f +/- %2.3f \n' %(np.mean(samples[:,i]),np.std(samples[:,i])))
for i in range(NCAT):
        f.write('\n Catalog %d: \n'%(i))
        catalog = catalogs[i]
        v_data_type, rescale_distance, _, _ = catalog
        if(rescale_distance):
                f.write('htilde: %2.3f +/- %2.3f \n' %(np.mean(samples[:,i+N_FLOW_PARAMS]),np.std(samples[:,i+N_FLOW_PARAMS])))
f.close()

if(plot_chain):
        print('Plotting Flow model chain....')
        flow_params = chain[:,:,:N_FLOW_PARAMS]
        chain_plot(flow_params, labels=flow_model_labels, savefig=output_dir+'/chain.png', show=False)

if(plot_corner):
        print('Plotting Flow model corner plot....')
        flow_samples = samples[:,:N_FLOW_PARAMS]
        corner.corner(flow_samples, labels=flow_model_labels)
        plt.savefig(output_dir+'/corner.png',dpi=150)
