import sys
import numpy as np
import cPickle as pickle

fname = sys.argv[1]

with open(fname, 'r') as f:
    hparams = pickle.load(f)
    print "cov_main", hparams['cov_main']
    print "cov_fic", hparams['cov_fic']
    print "noise_var", hparams['noise_var']
