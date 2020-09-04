import pandas as pd
import argparse
import numpy as np
import os
import utilities

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='outdir', default = 'out')
parser.add_argument('-n', type=int, dest='N', default = 300)
parser.add_argument('-s', type=int, dest='samples', default=1)
parser.add_argument('--seed', type=int)
parser.add_argument('-c', required=True, dest='catalog')
args = parser.parse_args()

if not os.path.isdir(args.outdir):
    os.mkdir(args.outdir)

if args.catalog.endswith('.csv'):
    halos = pd.read_csv(args.catalog, index_col=0)
else:
    from astropy.cosmology import FlatLambdaCDM
    cosmo = FlatLambdaCDM(H0=100, Om0=0.315)
    c = 299792.458 #km/s
    halos = utilities.fitres_to_pd(args.catalog)
    halos.rename(columns={'RA_GAL': 'RA', 'DEC_GAL': 'DEC'}, inplace=True)
    halos['zCMB'] = (halos['ZTRUE'] + 1.)*(halos['VPEC']/c + 1.) - 1.
    halos['mu'] = cosmo.distmod(halos['ZTRUE']).value \
                + np.random.default_rng().normal(0.0, 0.08, len(halos['ZTRUE']))
    halos['e_mu'] = 0.08

bg = np.random.PCG64(args.seed)

for i in range(args.samples):
    df = halos.sample(args.N, random_state=bg)
    df.to_csv('{}/{}.csv'.format(args.outdir, i), index=False)

