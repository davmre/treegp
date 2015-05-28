from treegp.bcm.multi_shared_bcm import MultiSharedBCM, Blocker, sample_synthetic
from treegp.bcm.local_regression import BCM
from treegp.bcm.bcmopt import SampledData

from treegp.gp import GPCov, GP, mcov, prior_sample, dgaussian
from treegp.util import mkdir_p
import numpy as np
import scipy.stats
import scipy.optimize
import time
import os
import sys

import cPickle as pickle

def load_log(run_name):
    d = "/home/dmoore/python/sigvisa/experiments/bcmopt/" + run_name
    log = os.path.join(d, "log.txt")
    logArray = np.loadtxt(log)
    steps = np.asarray(logArray[:, 0], dtype=int)
    times = logArray[:, 1]
    lls = logArray[:, 2]
    return steps, times, lls

def load_Xs(run_name, interval):
    d = "/home/dmoore/python/sigvisa/experiments/bcmopt/" + run_name

    Xs = []
    i = 0
    steps = []
    while True:
        fname = os.path.join(d, "step_%05d.npy" % i)
        if not os.path.exists(fname): break
        X = np.load(fname)
        Xs.append(X)
        steps.append(i)

        i += interval
    return steps, Xs

def plot_ll(run_name):
    steps, times, lls = load_log(run_name)


def plot_predictive_lik(run_name, interval=10):
    steps, times, lls = load_log(run_name)
    isteps, Xs = load_Xs(run_name)

def plot_mean_abs_deviation(run_name, interval=10):
    pass

def results_from_step(data_fname, step_fname):

    with open(data_fname, 'rb') as f:
        sdata = pickle.load(f)

    XX = np.load(step_fname)
    x = XX.flatten()

    print "MAD error %.04f" % (sdata.mean_abs_err(x))
    if XX.shape[0] < 1000:
        print ("GP predictive likelihood  %.3f (true %.3f)" % (sdata.prediction_error_gp(x),sdata.prediction_error_gp(sdata.SX.flatten())))
    print "BCM predictive likelihood  %.3f (true %.3f)" % (sdata.prediction_error_bcm(x),sdata.prediction_error_bcm(sdata.SX.flatten()))

def main():
    results_from_step(sys.argv[1], sys.argv[2])

if __name__ =="__main__":
    main()
