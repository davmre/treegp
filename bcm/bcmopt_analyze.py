from sigvisa.models.spatial_regression.multi_shared_bcm import MultiSharedBCM, Blocker, sample_synthetic
from sigvisa.models.spatial_regression.local_regression import BCM

from sigvisa.treegp.gp import GPCov, GP, mcov, prior_sample, dgaussian
from sigvisa.utils.fileutils import mkdir_p
import numpy as np
import scipy.stats
import scipy.optimize
import time
import os

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
