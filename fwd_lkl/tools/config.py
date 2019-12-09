import configparser

def config_fwd_lkl(configfile):
    """
    configfile is the filename of the config file
    """
    print('Entering config_fwd_lkl....')
    config = configparser.ConfigParser()
    config.read(configfile)

    NCAT = int(config['default']['NCAT'])
    fit_method = config['default']['fit_method']

    vary_sig_v = bool(config['flow_model']['vary_sig_v']=="True")

    output_dir = config['io']['output_dir']

    data_file = config['reconstruction']['data_file']
    coord_system = config['reconstruction']['coord_system']
    box_size = float(config['reconstruction']['box_size'])
    corner   = float(config['reconstruction']['corner'])
    N_GRID   = int(config['reconstruction']['N_GRID'])

    N_MCMC = int(config['MCMC']['N_MCMC'])
    N_WALKERS = int(config['MCMC']['N_WALKERS'])
    N_THREADS = int(config['MCMC']['N_THREADS'])

    catalogs = []

    for i in range(NCAT):
        catalog_i = catalog_parser(config, i)
        catalogs.append(catalog_i)

    print('Exiting config_fwd_lkl....')
    return NCAT, fit_method, \
            vary_sig_v, output_dir, \
            data_file, coord_system, box_size, corner, N_GRID, \
            N_MCMC, N_WALKERS, N_THREADS, \
                catalogs

def catalog_parser(config, i):
    v_data_type = config['catalog_'+str(i)]['v_data_type']
    data_file = config['catalog_'+str(i)]['data_file']
    assert v_data_type == 'simple_gaussian' or v_data_type == 'sn_lc_fit' or v_data_type == 'tf' or v_data_type == 'fp'
    rescale_distance = None
    add_sigma_int = None
    if(v_data_type=='simple_gaussian'):
        rescale_distance = bool(config['catalog_'+str(i)]['rescale_distance'] == 'True')
        add_sigma_int = bool(config['catalog_'+str(i)]['add_sigma_int'] == 'True')
    return [v_data_type, rescale_distance, add_sigma_int, data_file]

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
