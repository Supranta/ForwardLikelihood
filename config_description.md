# Configuration file description

You can find an example of the config file in the `config/inverse` folder.

### Default options

There are two options you need to enter for the default options:

```ini
[DEFAULT]
NCAT=3                     # Number of peculiar velocity catalogs in use
fit_method = optimize      # Either optimize or mcmc. mcmc samples from the posterior.
```

### io options

```ini
[io]
output_dir = output/carrick/all_combined      # The output directory where to save the results (e.g, mcmc chains, plots, etc)
```

### flow_model options

```ini
[flow_model]
vary_sig_v = False            # Whether fit sigma_v or not. If True, fit sigma_v as a parameter, If False, sigma_v fixed to 150
```

### reconstruction options

```ini
[reconstruction]
data_file = data/reconstruction_fields/carrick.h5     # File containing the reconstructed density and velocity fields
coord_system = galactic                               # Coordinates in which the reconstructed fields are given
box_size = 400.                                       # Size of each side of the cubic box containing the fields
corner = -200.                                        # Coordinate of the corner point
N_GRID = 257                                          # Number of grids
```

### MCMC options

This is required only if `fit_method = mcmc`
```ini
[MCMC]
N_MCMC = 500                                # Number of MCMC steps
N_WALKERS = 64                              # Number of emcee walkers
N_THREADS = 64                              # Number of threads to use
```

### Analyze options

This is processed by `analyze.py` file to process the resulting mcmc chains. Some of the options (e.g, autocorr) are not completely checked.
The mcmc chains are processed from the `output_dir` to produce plots of the chain, posterior as a function of the chain, a corner plot of parameter constraints
```ini
[analyze]
plot_chain = True                       # If True, plot the trace plots for the mcmc chains
plot_lkl = True                         # If True, plot the posterior as a function of the chains. Useful for checking burn-in
plot_corner = True                      # If True, plot the corner plot for parameter constraints 
autocorr = False                        # If True, plot the autocorrelation function for each emcee walker  
```

### Catalog descriptions

At the moment, this code can only read csv files for the peculiar velocity fields.

There are additional parameters for each catalog which we may need to infer (e.g, for `v_data_type=tf`, the code also infers a_TF, b_TH and sigma_int).

If `lognormal = True`, the distance prior is Gaussian in the log-distance, not in distance.

In this case there are 3 catalogs:

- SFI++ field galaxy sample, where the Tully-Fisher parameters are inferred as well (Hence `v_data_type=tf`).
- SFI++ group sample, where the given distances are taken as given (Hence `v_data_type=simple_distance`). 
There is however a factor h_tilde, which rescales the distance (Hence, `rescale_distance = True`).
There may be catalogs in which an additional intrinsic scatter may be needed. In such cases, use `add_sigma_int = True`
- Foundation sample: The global parameters for the Tripp formula are inferred as well. 

```ini
[catalog_0]
v_data_type = tf
data_file = data/peculiar_velocity_catalog/sfi_gals_tf.csv
lognormal = False

[catalog_1]
v_data_type = simple_distance
data_file = data/peculiar_velocity_catalog/sfi_grps.csv
rescale_distance = True
add_sigma_int = False
lognormal = False

[catalog_2]
v_data_type = sn_lc_fit
data_file = data/peculiar_velocity_catalog/foundation.csv
lognormal=False
```

#### What are the required fields for the peculiar velocity data file?

For all datasets, you need the following fields in the csv file, 

- `RA`: Right Ascension in degrees
- `DEC`: Declination in degrees
- `zCMB`: The redshift in the CMB frame. 

If `v_data_type = simple_distance`, the distance can be in one of the following:
- `rhMpc`: the distances to the object in h^{-1} Mpc
- `mu`: Distance modulus + `5log10(h)`. Note that the distance modulus is usually calculated using the luminosity distance. You will need to convert it to comoving distance.
For the errors, use:
- `e_rhMpc`
- `e_mu`

If `v_data_type = sn_lc_fit`, you need the following fields in the csv file as well:
- `c` and `e_c`: The color parameter and the error
- `x1` and `e_x1`: The stretch parameter and the error
- `mB` and `e_mB`: The magnitude in the B-band and the error

If `v_data_type = tf`, you need the following fields in the csv file as well:
- `mag` and `e_mag`: The apparent magnitude and its error
- `eta` and `e_eta`: eta and its error
