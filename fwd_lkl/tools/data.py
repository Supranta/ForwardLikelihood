from scipy.interpolate import RegularGridInterpolator
import numpy as np
import h5py as h5

data_repository = '/Users/boruah/Desktop/research/bayesian_fwd_modelling/pec_vel_fwd_lkl_module/data/reconstruction_fields/'
def process_reconstruction_data(data_file, box_size, corner, N_grid):
    l = box_size/N_grid

    X = np.linspace(corner, corner+box_size, N_grid)
    Y = np.linspace(corner, corner+box_size, N_grid)
    Z = np.linspace(corner, corner+box_size, N_grid)

    f = h5.File(data_repository+data_file,'r')

    v_data = np.array(f['velocity'])

    v_x_data, v_y_data, v_z_data = v_data

    v_x_interp = RegularGridInterpolator((X, Y, Z), v_x_data)
    v_y_interp = RegularGridInterpolator((X, Y, Z), v_y_data)
    v_z_interp = RegularGridInterpolator((X, Y, Z), v_z_data)

    delta_data = np.array(f['density'])
    delta_interp = RegularGridInterpolator((X, Y, Z), delta_data)

    return delta_interp, [v_x_interp, v_y_interp, v_z_interp]
