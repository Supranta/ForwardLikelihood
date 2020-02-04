from ..simple_distance import simple_distance
from ..sn_lc_fit import sn_lc_fit
from ..tf import TF
from ..fp import FP
import numpy as np
import pandas as pd
from .cosmological_funcs import r_from_mu

def create_catalog_obj(distance_indicator, v_data_file, \
            start_index,\
            vary_sig_v,\
            v_field, delta_field, coord_system, lognormal,\
            rescale_distance=None, add_sigma_int=None):

    print('Entering create_catalog_obj....')
    df = pd.read_csv(v_data_file)
    RA = np.array(df['RA'])
    DEC = np.array(df['DEC'])
    zCMB = np.array(df['zCMB'])

    if(distance_indicator == 'simple_distance'):
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
        obj = simple_distance(v_data, v_field, delta_field, coord_system,\
                                vary_sig_v, start_index,\
                                rescale_distance, add_sigma_int, lognormal)
    elif(distance_indicator == 'sn_lc_fit'):
        mB = np.array(df['mB'])
        c_sn = np.array(df['c'])
        x1 = np.array(df['x1'])
        e_mB = np.array(df['e_mB'])
        e_c = np.array(df['e_c'])
        e_x1 = np.array(df['e_x1'])
        v_data = [RA, DEC, zCMB, mB, c_sn, x1, e_mB, e_c, e_x1]
        obj = sn_lc_fit(v_data, v_field, delta_field, coord_system,\
                                vary_sig_v, start_index, lognormal)
    elif(distance_indicator == 'tf'):
        i = np.array(df['mag'])
        eta = np.array(df['eta'])
        e_i = np.array(df['e_mag'])
        e_eta = np.array(df['e_eta'])
        v_data = [RA, DEC, zCMB, i, eta, e_i, e_eta]
        obj = TF(v_data, v_field, delta_field, coord_system,\
                                vary_sig_v, start_index, lognormal)
    elif(distance_indicator == 'fp'):
        s = np.array(df['s'])
        i = np.array(df['i'])
        r = np.array(df['r'])
        e_s = np.array(df['e_s'])
        e_i = np.array(df['e_i'])
        e_r = np.array(df['e_r'])
        v_data = [RA, DEC, zCMB, s, i, r, e_s, e_i, e_r]
        obj = FP(v_data, v_field, delta_field, coord_system,\
                                vary_sig_v, start_index)
    print('Exiting create_catalog_obj....')
    return obj
