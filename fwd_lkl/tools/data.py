from scipy.interpolate import RegularGridInterpolator
import numpy as np
import h5py as h5

def process_reconstruction_data(data_file, box_size, corner, N_grid):
    print('Entering reconstruction data interpolation....')
    l = box_size/N_grid

    X = np.linspace(corner+0.5*l, corner+box_size-0.5*l, N_grid)
    Y = np.linspace(corner+0.5*l, corner+box_size-0.5*l, N_grid)
    Z = np.linspace(corner+0.5*l, corner+box_size-0.5*l, N_grid)

    f = h5.File(data_file,'r')

    v_data = np.array(f['velocity'])

    v_x_data, v_y_data, v_z_data = v_data

    v_x_interp = RegularGridInterpolator((X, Y, Z), v_x_data)
    v_y_interp = RegularGridInterpolator((X, Y, Z), v_y_data)
    v_z_interp = RegularGridInterpolator((X, Y, Z), v_z_data)

    delta_data = np.array(f['density'])
    delta_interp = RegularGridInterpolator((X, Y, Z), delta_data)

    print('Exiting reconstruction data interpolation....')
    return delta_interp, [v_x_interp, v_y_interp, v_z_interp]
