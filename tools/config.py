import configparser

def config_fwd_lkl(configfile):
    """
    configfile is the filename of the config file
    """
    config = configparser.ConfigParser()
    config.read(configfile)

    NCAT = int(config['default']['NCAT'])
    fit_method = config['default']['fit_method']

    vary_sig_v = bool(config['flow_model']['vary_sig_v']=="True")
    fix_beta = bool(config['flow_model']['fix_beta']=="True")

    data_file = config['reconstruction']['data_file']
    coord_system = config['reconstruction']['coord_system']
    box_size = float(config['reconstruction']['box_size'])
    corner   = float(config['reconstruction']['corner'])

    N_MCMC = int(config['MCMC']['N_MCMC'])
    N_WALKERS = int(config['MCMC']['N_WALKERS'])
    N_THREADS = int(config['MCMC']['N_THREADS'])

    catalogs = []

    for i in range(NCAT):
        catalog_i = catalog_parser(config, i)
        catalogs.append(catalog_i)

    return NCAT, fit_method, \
            vary_sig_v, fix_beta, \
            data_file, coord_system, box_size, corner, \
            N_MCMC, N_WALKERS, N_THREADS, \
                catalogs

def catalog_parser(config, i):
    distance_indicator = config['catalog_'+str(i)]['distance_indicator']
    rescale_distance = bool(config['catalog_'+str(i)]['rescale_distance'] == 'True')
    file_format = config['catalog_'+str(i)]['file_format']
    data_file = config['catalog_'+str(i)]['data_file']
    assert distance_indicator == 'simple'
    assert file_format == 'fits'
    return [distance_indicator, rescale_distance, file_format, data_file]
