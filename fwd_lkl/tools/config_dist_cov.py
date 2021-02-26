import configparser
import numpy as np
import pandas as pd

def config_io(configfile):
    print("Entering config_io....")
    config = configparser.ConfigParser()
    config.read(configfile)

    output_dir                = config['IO']['output_dir']
    reconstruction_data_file  = config['IO']['reconstruction_data_file']

    print("Exiting config_io....")
    return output_dir, reconstruction_data_file


def config_PV_data(configfile):
    print("Entering config_data....")
    config = configparser.ConfigParser()
    config.read(configfile)

    dist_cov_path = config['PV_DATA']['dist_cov']
    dist_cov = np.load(dist_cov_path)

    PV_datafile = config['PV_DATA']['datafile']

    df = pd.read_csv(PV_datafile)
    RA   = np.array(df['RA'])
    DEC  = np.array(df['DEC'])
    zCMB = np.array(df['zCMB'])
    mu   = np.array(df['mu'])
    e_mu = np.array(df['e_mu'])
    v_data = [RA, DEC, zCMB, mu, e_mu]

    print("Exiting config_data....")
    return v_data, dist_cov

def config_box(configfile):
    print("Entering config_box....")
    config = configparser.ConfigParser()
    config.read(configfile)

    coord_system = config['BOX']['coord_system']
    box_size     = float(config['BOX']['box_size'])
    corner       = float(config['BOX']['corner'])
    N_GRID       = int(config['BOX']['N_GRID'])

    print("Exiting config_box....")

    return coord_system, box_size, corner, N_GRID

def config_mcmc(configfile):
    print("Entering config_mcmc....")
    config = configparser.ConfigParser()
    config.read(configfile)


    N_MCMC     = int(config['MCMC']['N_MCMC'])
    dt         = float(config['MCMC']['dt'])
    N_LEAPFROG = int(config['MCMC']['N_LEAPFROG'])

    print("Exiting config_mcmc....")

    return N_MCMC, dt, N_LEAPFROG
