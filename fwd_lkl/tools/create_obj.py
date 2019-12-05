from ..simple_gaussian import simple_gaussian
import numpy as np
import pandas as pd

v_data_directory = '/Users/boruah/Desktop/research/bayesian_fwd_modelling/pec_vel_fwd_lkl_module/data/peculiar_velocity_catalog/'

def catalog_obj(distance_indicator, v_data_file, \
            rescale_distance, start_index,\
            vary_sig_v,\
            v_field, delta_field, coord_system, file_format='csv'):
    if(file_format=='csv'):
        df = pd.read_csv(v_data_directory+v_data_file)
        RA = np.array(df['RA'])
        DEC = np.array(df['DEC'])
        zCMB = np.array(df['zCMB'])
        r_hMpc = np.array(df['rhMpc'])
        e_r_hMpc = np.array(df['e_rhMpc'])
    v_data = [RA, DEC, zCMB, r_hMpc, e_r_hMpc]

    if(distance_indicator == 'simple_gaussian'):
        obj = simple_gaussian(v_data, v_field, delta_field, coord_system,\
                                vary_sig_v, start_index,\
                                rescale_distance)

    return obj
