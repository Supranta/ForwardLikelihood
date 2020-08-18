import pandas as pd
import argparse
import numpy as np
import os

parser = argparse.ArgumentParser()
parser.add_argument('-d', dest='outdir', default = 'out')
parser.add_argument('-n', type=int, dest='N', default = 300)
parser.add_argument('-s', type=int, dest='samples', default=1)
parser.add_argument('--seed', type=int)
args = parser.parse_args()

if not os.path.isdir(args.outdir):
    os.mkdir(args.outdir)

halos = pd.read_csv('corrected_halo_catalog.csv', index_col=0)
bg = np.random.PCG64(args.seed)

for i in range(args.samples):
    df = halos.sample(args.N, random_state=bg)
    df.to_csv('{}/{}.csv'.format(args.outdir, i), index=False)

