from ..simple_gaussian import simple_gaussian
from ..sn_lc_fit import sn_lc_fit
from ..tf import TF
import numpy as np
import pandas as pd
from .cosmological_funcs import r_from_mu

def catalog_obj(distance_indicator, v_data_file, \
            start_index,\
            vary_sig_v,\
            v_field, delta_field, coord_system,\
            rescale_distance=None, add_sigma_int=None):

    df = pd.read_csv(v_data_file)
    RA = np.array(df['RA'])
    DEC = np.array(df['DEC'])
    zCMB = np.array(df['zCMB'])
    if(distance_indicator == 'simple_gaussian'):
        try:
            r_hMpc = np.array(df['rhMpc'])
        except:
            mu = np.array(df['mu'])
            r_hMpc = r_from_mu(mu)
        try:
            e_r_hMpc = np.array(df['e_rhMpc'])
        except:
            e_mu = np.array(df['e_mu'])
            e_r_hMpc = e_mu  * (np.log(10)/5.0) * r_hMpc
        v_data = [RA, DEC, zCMB, r_hMpc, e_r_hMpc]
        obj = simple_gaussian(v_data, v_field, delta_field, coord_system,\
                                vary_sig_v, start_index,\
                                rescale_distance, add_sigma_int)
    elif(distance_indicator == 'sn_lc_fit'):
        mB = np.array(df['mB'])
        c_sn = np.array(df['c'])
        x1 = np.array(df['x1'])
        e_mB = np.array(df['e_mB'])
        e_c = np.array(df['e_c'])
        e_x1 = np.array(df['e_x1'])
        v_data = [RA, DEC, zCMB, mB, c_sn, x1, e_mB, e_c, e_x1]
        obj = sn_lc_fit(v_data, v_field, delta_field, coord_system,\
                                vary_sig_v, start_index)
    elif(distance_indicator == 'tf'):
        i = np.array(df['mag'])
        eta = np.array(df['eta'])
        e_i = np.array(df['e_mag'])
        e_eta = np.array(df['e_eta'])
        v_data = [RA, DEC, zCMB, i, eta, e_i, e_eta]
        obj = TF(v_data, v_field, delta_field, coord_system,\
                                vary_sig_v, start_index)
    return obj
