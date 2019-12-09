import numpy as np
import sys, os
from fwd_lkl.fwd_lkl import num_flow_params, flow_params_pos0
from fwd_lkl.tools.config import analyze_fwd_lkl
from fwd_lkl.tools.plotting import chain_plot
#from fwd_lkl.tools.create_obj import create_catalog_obj
from fwd_lkl.tools.utils import v_sample_direction_norm
import corner
import matplotlib.pyplot as plt

plt.rc('text',usetex=True)
plt.rc('font',family='serif')

def num_params(v_data_type, rescale_distance=False):
    if(v_data_type=='simple_gaussian'):
        if(rescale_distance):
            return 1
        else:
            return 0
    elif(v_data_type=='tf'):
        return 3
    elif(v_data_type=='sn_lc_fit'):
        return 4
configfile = sys.argv[1]
N_BURN_IN = int(sys.argv[2])

coord_system, NCAT, vary_sig_v, output_dir, \
        plot_chain, plot_lkl, plot_corner, catalogs = analyze_fwd_lkl(configfile)

home_dir = os.path.abspath('.')
output_dir = home_dir+'/'+output_dir

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
if(plot_chain):
    print('Plotting Flow model chain....')
    flow_params = chain[:,:,:N_FLOW_PARAMS]
    chain_plot(flow_params, labels=flow_model_labels, savefig=output_dir+'/flow_model_chain.png', show=False)
if(plot_corner):
    print('Plotting Flow model corner plot....')
    flow_samples = samples[:,:N_FLOW_PARAMS]
    V_samples = samples[:,N_FLOW_PARAMS-3:N_FLOW_PARAMS]
    corner.corner(flow_samples, labels=flow_model_labels)
    plt.savefig(output_dir+'/flow_model_corner.png',dpi=150)
    V_tot, l, b = v_sample_direction_norm(V_samples, coord_system)
    corner.corner(np.array([V_tot, l, b]).T, labels=[r'$V_{\textrm{ext}}$', '$l$', '$b$'])
    plt.savefig(output_dir+'/bulk_flow.png',dpi=150)
    #l[l < 120] += 360
    f.write('V_ext : %2.1f + %2.1f - %2.1f \n' %(np.mean(V_tot), np.percentile(V_tot, 84)-np.mean(V_tot), np.mean(V_tot)-np.percentile(V_tot, 16)))
    f.write('l : %2.1f + %2.1f - %2.1f \n' %(np.mean(l), np.percentile(l, 84)-np.mean(l), np.mean(l)-np.percentile(l, 16)))
    f.write('b : %2.1f + %2.1f - %2.1f \n' %(np.mean(b), np.percentile(b, 84)-np.mean(b), np.mean(b)-np.percentile(b, 16)))

N = N_FLOW_PARAMS

for i in range(NCAT):
    f.write('\n Catalog %d: \n'%(i))
    catalog = catalogs[i]
    v_data_type, rescale_distance, add_sigma_int, v_data_file = catalog
    N_PARAMS = num_params(v_data_type, rescale_distance)
    for j in range(N_PARAMS):
        f.write('parameter %d: %2.3f +/- %2.3f \n' %(j, np.mean(samples[:,i+N]),np.std(samples[:,i+N])))
    if(N_PARAMS != 0):
        if(plot_chain):
            print('Plotting catalog %d chain....'%(i))
            catalog_params = chain[:,:,N:N+N_PARAMS]
            chain_plot(catalog_params, savefig=output_dir+'/catalog_'+str(i)+'_chain.png', show=False)
        if(plot_corner):
            print('Plotting catalog %d corner plot....'%(i))
            catalog_samples = samples[:,N:N+N_PARAMS]
            corner.corner(catalog_samples)
            plt.savefig(output_dir+'/catalog_'+str(i)+'_corner.png',dpi=150)
    N += N_PARAMS
f.close()


