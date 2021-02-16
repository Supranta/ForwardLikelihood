import configparser
import numpy as np

def config_fixed_V_ext(configfile):
    print('Entering config_fwd_lkl....')
    config = configparser.ConfigParser()
    config.read(configfile)

    V_ext_fixed      = np.array(config['flow_model']['V_ext_fixed'].split(',')).astype(float)

    return V_ext_fixed

def config_fwd_lkl(configfile):
    """
    configfile is the filename of the config file
    """
    print('Entering config_fwd_lkl....')
    config = configparser.ConfigParser()
    config.read(configfile)

    NCAT = int(config['default']['NCAT'])
    fit_method = config['default']['fit_method']

    fix_V_ext      = bool(config['flow_model']['fix_V_ext']=="True")
    vary_sig_v     = bool(config['flow_model']['vary_sig_v']=="True")
    add_quadrupole = bool(config['flow_model']['add_quadrupole']=="True")
    radial_beta    = bool(config['flow_model']['radial_beta']=="True")

    try:
        czlow = float(config['redshift_select']['czlow'])
        czhigh = float(config['redshift_select']['czhigh'])
    except:
        czlow = 0.0
        czhigh = 25000.

    output_dir = config['io']['output_dir']

    data_file = config['reconstruction']['data_file']
    coord_system = config['reconstruction']['coord_system']
    box_size = float(config['reconstruction']['box_size'])
    corner   = float(config['reconstruction']['corner'])
    N_GRID   = int(config['reconstruction']['N_GRID'])

    N_MCMC = int(config['MCMC']['N_MCMC'])
    N_WALKERS = int(config['MCMC']['N_WALKERS'])
    try:
        sampler_type = config['MCMC']['sampler_type']
    except:
        sampler_type = 'emcee'

    catalogs = []

    for i in range(NCAT):
        catalog_i = catalog_parser(config, i)
        catalogs.append(catalog_i)

    print('Exiting config_fwd_lkl....')
    return NCAT, fit_method, \
            fix_V_ext, vary_sig_v, add_quadrupole, radial_beta, \
            output_dir, czlow, czhigh, \
            data_file, coord_system, box_size, corner, N_GRID, \
            N_MCMC, N_WALKERS, sampler_type, catalogs

def catalog_parser(config, i):
    catalog_str = 'catalog_'+str(i)
    v_data_type = config[catalog_str]['v_data_type']
    data_file = config[catalog_str]['data_file']
    assert v_data_type == 'simple_distance' or v_data_type == 'sn_lc_fit' or v_data_type == 'tf' or v_data_type == 'fp' or v_data_type == 'lxt'
    rescale_distance = None
    add_sigma_int    = None
    if(v_data_type=='simple_distance'):
        rescale_distance = bool(config[catalog_str]['rescale_distance'] == 'True')
        add_sigma_int = bool(config[catalog_str]['add_sigma_int'] == 'True')
    try:
        dist_cov_path = config[catalog_str]['dist_cov']
        dist_cov = np.load(dist_cov_path)
    except:
        dist_cov = None
    lognormal = bool(config[catalog_str]['lognormal'] == "True")
    return [v_data_type, rescale_distance, add_sigma_int, data_file, lognormal, dist_cov]

def analyze_fwd_lkl(configfile):
    """
    configfile is the filename of the config file
    """
    config = configparser.ConfigParser()
    config.read(configfile)

    coord_system = config['reconstruction']['coord_system']
    NCAT = int(config['default']['NCAT'])

    vary_sig_v = bool(config['flow_model']['vary_sig_v']=="True")

    output_dir = config['io']['output_dir']

    plot_chain = bool(config['analyze']['plot_chain']=='True')
    plot_lkl = bool(config['analyze']['plot_lkl']=='True')
    plot_corner = bool(config['analyze']['plot_corner']=='True')

    catalogs = []

    for i in range(NCAT):
        catalog_i = catalog_parser(config, i)
        catalogs.append(catalog_i)

    return coord_system, NCAT, vary_sig_v, output_dir, \
            plot_chain, plot_lkl, plot_corner, catalogs
